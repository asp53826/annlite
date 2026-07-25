"""End to end tests for the HTTP service."""

import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import server as srv  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "DATA_DIR", str(tmp_path))
    srv.collections.clear()
    with TestClient(srv.app) as c:
        yield c


def vecs(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32).tolist()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["simd"] in ("neon", "avx", "scalar")


def test_create_insert_query(client):
    assert client.post("/collections", json={"name": "docs", "dim": 8}).status_code == 201
    v = vecs(200, 8)
    r = client.post("/collections/docs/points", json={"vectors": v})
    assert r.status_code == 200 and r.json()["inserted"] == 200

    r = client.post("/collections/docs/query", json={"vectors": [v[3]], "k": 5, "ef": 64})
    hits = r.json()["results"][0]
    assert len(hits) == 5
    assert hits[0]["id"] == 3 and hits[0]["distance"] < 1e-3
    assert [h["distance"] for h in hits] == sorted(h["distance"] for h in hits)


def test_duplicate_collection_is_a_conflict(client):
    client.post("/collections", json={"name": "a", "dim": 4})
    assert client.post("/collections", json={"name": "a", "dim": 4}).status_code == 409


def test_missing_collection_is_404(client):
    assert client.post("/collections/nope/query", json={"vectors": [[1, 2]]}).status_code == 404
    assert client.get("/collections/nope/stats").status_code == 404


def test_dimension_mismatch_is_400(client):
    client.post("/collections", json={"name": "d", "dim": 8})
    r = client.post("/collections/d/points", json={"vectors": [[1.0, 2.0]]})
    assert r.status_code == 400


def test_explicit_ids_are_honoured(client):
    client.post("/collections", json={"name": "e", "dim": 4})
    client.post("/collections/e/points", json={"vectors": vecs(3, 4), "ids": [77, 88, 99]})
    r = client.post("/collections/e/query", json={"vectors": vecs(3, 4)[:1], "k": 3, "ef": 32})
    assert set(h["id"] for h in r.json()["results"][0]) == {77, 88, 99}


def test_id_count_must_match(client):
    client.post("/collections", json={"name": "f", "dim": 4})
    r = client.post("/collections/f/points", json={"vectors": vecs(3, 4), "ids": [1]})
    assert r.status_code == 400


def test_delete_then_query(client):
    client.post("/collections", json={"name": "g", "dim": 8})
    v = vecs(100, 8)
    client.post("/collections/g/points", json={"vectors": v})
    r = client.request("DELETE", "/collections/g/points", json={"ids": [0, 1, 2]})
    assert r.json() == {"requested": 3, "removed": 3, "size": 97}

    hits = client.post("/collections/g/query",
                       json={"vectors": [v[0]], "k": 10, "ef": 64}).json()["results"][0]
    assert not ({0, 1, 2} & {h["id"] for h in hits})


def test_deleting_unknown_ids_reports_zero(client):
    client.post("/collections", json={"name": "h", "dim": 4})
    client.post("/collections/h/points", json={"vectors": vecs(5, 4)})
    assert client.request("DELETE", "/collections/h/points",
                          json={"ids": [999]}).json()["removed"] == 0


def test_exact_mode_is_ground_truth(client):
    """Exact search must never miss what approximate search finds."""
    client.post("/collections", json={"name": "i", "dim": 16})
    v = vecs(500, 16, seed=2)
    client.post("/collections/i/points", json={"vectors": v})
    q = {"vectors": [v[10]], "k": 10}
    approx = client.post("/collections/i/query", json={**q, "ef": 128}).json()
    exact = client.post("/collections/i/query", json={**q, "exact": True}).json()
    a = {h["id"] for h in approx["results"][0]}
    e = {h["id"] for h in exact["results"][0]}
    assert len(a & e) >= 9
    assert exact["exact"] is True and approx["distance_computations"] > 0


def test_persist_and_restore_roundtrip(client):
    client.post("/collections", json={"name": "j", "dim": 8})
    v = vecs(300, 8, seed=5)
    client.post("/collections/j/points", json={"vectors": v})
    before = client.post("/collections/j/query",
                         json={"vectors": v[:5], "k": 5, "ef": 64}).json()["results"]

    assert client.post("/collections/j/persist").json()["bytes"] > 0
    client.delete("/collections/j")
    assert "j" not in client.get("/collections").json()

    assert client.post("/collections/j/restore").json()["size"] == 300
    after = client.post("/collections/j/query",
                        json={"vectors": v[:5], "k": 5, "ef": 64}).json()["results"]
    assert before == after


def test_restore_without_a_snapshot_is_404(client):
    assert client.post("/collections/ghost/restore").status_code == 404


def test_stats_and_listing(client):
    client.post("/collections", json={"name": "k", "dim": 8})
    client.post("/collections/k/points", json={"vectors": vecs(50, 8)})
    st = client.get("/collections/k/stats").json()
    assert st["elements"] == 50 and st["dim"] == 8
    assert client.get("/collections").json()["k"]["size"] == 50


def test_empty_vector_list_is_rejected(client):
    client.post("/collections", json={"name": "l", "dim": 4})
    assert client.post("/collections/l/points", json={"vectors": []}).status_code == 400
