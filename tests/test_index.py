"""Correctness tests for the C++ index, driven from Python.

The one that matters most is recall against exact brute force. An ANN index
that silently returns slightly wrong neighbours passes every smoke test and
quietly degrades whatever is built on top of it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import _annlite as annlite  # noqa: E402

# padding marker for result rows shorter than k
NO_LABEL = (1 << 64) - 1


def make_data(n=2000, dim=32, seed=0, clustered=True):
    rng = np.random.default_rng(seed)
    if not clustered:
        return rng.standard_normal((n, dim)).astype(np.float32)
    centers = rng.standard_normal((max(n // 200, 4), dim)).astype(np.float32) * 5
    pick = rng.integers(0, len(centers), n)
    return (centers[pick] + rng.standard_normal((n, dim))).astype(np.float32)


def build(X, metric="l2", M=16, ef_construction=200, labels=None):
    idx = annlite.Index(dim=X.shape[1], metric=metric, M=M, ef_construction=ef_construction)
    idx.add(X, list(range(len(X))) if labels is None else labels)
    return idx


def recall(got, truth, k):
    return np.mean([len(set(g[:k].tolist()) & set(t[:k].tolist())) / k
                    for g, t in zip(got, truth)])


# ---------------------------------------------------------------- basics

def test_simd_backend_is_reported():
    assert annlite.simd_backend() in ("neon", "avx", "scalar")


def test_size_and_dim():
    X = make_data(500, 16)
    idx = build(X)
    assert len(idx) == 500 and idx.dim == 16 and idx.size == 500


def test_search_returns_k_neighbours_sorted_by_distance():
    X = make_data(1000, 16)
    idx = build(X)
    labels, dists, comps = idx.search(X[:5], k=10, ef=64)
    assert labels.shape == (5, 10) and dists.shape == (5, 10)
    assert comps > 0
    for row in dists:
        assert np.all(np.diff(row) >= -1e-5), "distances must be non decreasing"


def test_a_vector_finds_itself_first():
    """The nearest neighbour of an indexed point is itself, at distance 0.
    If this fails the graph is disconnected or the metric is wrong."""
    X = make_data(1000, 24)
    idx = build(X)
    labels, dists, _ = idx.search(X, k=1, ef=128)
    assert (labels[:, 0] == np.arange(len(X))).mean() > 0.99
    assert np.allclose(dists[:, 0], 0.0, atol=1e-3)


# ---------------------------------------------------------------- accuracy

@pytest.mark.parametrize("dim", [8, 32, 128])
def test_recall_against_exact_brute_force(dim):
    X = make_data(2000, dim)
    idx = build(X)
    Q = make_data(100, dim, seed=99)
    got, _, _ = idx.search(Q, k=10, ef=128)
    truth, _ = idx.brute_force(Q, k=10)
    assert recall(got, truth, 10) >= 0.97


def test_brute_force_distances_match_numpy():
    """Validates the SIMD kernel end to end, including the tail loop for
    dimensions that aren't a multiple of the vector width."""
    for dim in (7, 16, 33, 128):
        X = make_data(200, dim, seed=dim)
        idx = build(X)
        Q = make_data(5, dim, seed=dim + 1)
        labels, dists = idx.brute_force(Q, k=5)
        for qi in range(len(Q)):
            for j, lab in enumerate(labels[qi]):
                expected = float(((Q[qi] - X[lab]) ** 2).sum())
                assert dists[qi][j] == pytest.approx(expected, rel=1e-4, abs=1e-3)


def test_higher_ef_never_hurts_recall():
    X = make_data(3000, 32)
    idx = build(X)
    Q = make_data(100, 32, seed=7)
    truth, _ = idx.brute_force(Q, k=10)
    scores = []
    for ef in (10, 20, 50, 100, 200):
        got, _, _ = idx.search(Q, k=10, ef=ef)
        scores.append(recall(got, truth, 10))
    # monotone up to float noise: ef is the accuracy knob and it must behave
    assert all(b >= a - 0.02 for a, b in zip(scores, scores[1:])), scores
    assert scores[-1] > scores[0]


def test_more_ef_costs_more_distance_computations():
    X = make_data(2000, 32)
    idx = build(X)
    Q = make_data(50, 32, seed=3)
    _, _, cheap = idx.search(Q, k=10, ef=10)
    _, _, dear = idx.search(Q, k=10, ef=200)
    assert dear > cheap


def test_beats_brute_force_on_work_done():
    X = make_data(5000, 64)
    idx = build(X)
    Q = make_data(50, 64, seed=5)
    _, _, comps = idx.search(Q, k=10, ef=64)
    assert comps / len(Q) < len(X) / 4, "graph search should skip most of the index"


# ---------------------------------------------------------------- metrics

def test_cosine_ignores_magnitude():
    """Scaling a vector must not change its cosine neighbours."""
    X = make_data(500, 16)
    idx = build(X, metric="cosine")
    q = X[3:4]
    a, _, _ = idx.search(q, k=5, ef=64)
    b, _, _ = idx.search(q * 17.0, k=5, ef=64)
    assert list(a[0]) == list(b[0])


def test_inner_product_prefers_large_vectors():
    X = np.array([[1.0, 0.0], [10.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    idx = annlite.Index(dim=2, metric="ip", M=4)
    idx.add(X, [0, 1, 2])
    labels, _, _ = idx.search(np.array([[1.0, 0.0]], dtype=np.float32), k=1, ef=16)
    assert labels[0][0] == 1  # magnitude matters for inner product


def test_unknown_metric_rejected():
    with pytest.raises(Exception):
        annlite.Index(dim=4, metric="manhattan")


# ---------------------------------------------------------------- mutation

def test_delete_removes_from_results():
    X = make_data(1000, 16)
    idx = build(X)
    victim = int(idx.search(X[0:1], k=1, ef=64)[0][0][0])
    assert idx.remove(victim)
    assert victim not in idx
    labels, _, _ = idx.search(X[0:1], k=10, ef=128)
    assert victim not in labels[0]
    assert len(idx) == 999 and idx.deleted == 1


def test_delete_is_idempotent_and_reports_misses():
    X = make_data(200, 8)
    idx = build(X)
    assert idx.remove(5) is True
    assert idx.remove(5) is False
    assert idx.remove(99999) is False


def test_deleting_many_keeps_the_rest_findable():
    """Tombstones must not disconnect the graph for surviving nodes, since
    deleted nodes are still traversed as waypoints."""
    X = make_data(2000, 32)
    idx = build(X)
    rng = np.random.default_rng(1)
    doomed = set(rng.choice(2000, 800, replace=False).tolist())
    for label in doomed:
        idx.remove(int(label))
    assert len(idx) == 1200

    survivors = [i for i in range(2000) if i not in doomed]
    Q = X[survivors[:100]]
    labels, _, _ = idx.search(Q, k=1, ef=128)
    assert not (set(labels.ravel().tolist()) & doomed), "returned a deleted vector"
    found = (labels[:, 0] == np.array(survivors[:100])).mean()
    assert found > 0.95


def test_readding_a_label_replaces_the_old_vector():
    X = make_data(300, 8)
    idx = build(X)
    new = np.full((1, 8), 42.0, dtype=np.float32)
    idx.add(new, [7])
    assert len(idx) == 300  # replaced, not appended
    assert np.allclose(idx.get_vector(7), new[0])
    labels, _, _ = idx.search(new, k=1, ef=64)
    assert labels[0][0] == 7


def test_get_vector_roundtrips_and_raises_on_missing():
    X = make_data(100, 8)
    idx = build(X)
    assert np.allclose(idx.get_vector(4), X[4], atol=1e-6)
    with pytest.raises(Exception):
        idx.get_vector(12345)


def test_incremental_adds_match_a_bulk_build():
    X = make_data(1000, 16)
    bulk = build(X)
    inc = annlite.Index(dim=16, metric="l2", M=16, ef_construction=200)
    for i in range(0, 1000, 100):
        inc.add(X[i : i + 100], list(range(i, i + 100)))
    Q = make_data(50, 16, seed=11)
    truth, _ = bulk.brute_force(Q, k=10)
    a, _, _ = bulk.search(Q, k=10, ef=128)
    b, _, _ = inc.search(Q, k=10, ef=128)
    assert recall(b, truth, 10) >= recall(a, truth, 10) - 0.05


# ---------------------------------------------------------------- persistence

def test_save_load_returns_identical_results(tmp_path):
    X = make_data(1500, 32)
    idx = build(X)
    Q = make_data(60, 32, seed=13)
    before, dbefore, _ = idx.search(Q, k=10, ef=100)

    path = str(tmp_path / "idx.ann")
    idx.save(path)
    again = annlite.Index.load(path)

    after, dafter, _ = again.search(Q, k=10, ef=100)
    assert np.array_equal(before, after), "reloaded index gave different neighbours"
    assert np.allclose(dbefore, dafter, atol=1e-5)
    assert len(again) == len(idx) and again.dim == idx.dim


def test_save_load_preserves_deletions(tmp_path):
    X = make_data(500, 16)
    idx = build(X)
    for label in range(0, 100):
        idx.remove(label)
    path = str(tmp_path / "d.ann")
    idx.save(path)
    again = annlite.Index.load(path)
    assert len(again) == 400 and again.deleted == 100
    labels, _, _ = again.search(X[:50], k=10, ef=128)
    assert not (set(labels.ravel().tolist()) & set(range(100)))


def test_load_rejects_a_non_index_file(tmp_path):
    bad = tmp_path / "junk.ann"
    bad.write_bytes(b"this is not an index" * 10)
    with pytest.raises(Exception):
        annlite.Index.load(str(bad))


def test_save_is_atomic_no_partial_file_left(tmp_path):
    X = make_data(200, 8)
    idx = build(X)
    path = str(tmp_path / "a.ann")
    idx.save(path)
    idx.save(path)  # overwrite an existing index
    assert not os.path.exists(path + ".tmp")
    assert len(annlite.Index.load(path)) == 200


# ---------------------------------------------------------------- edges

def test_empty_index_returns_nothing():
    idx = annlite.Index(dim=8)
    labels, dists, _ = idx.search(make_data(3, 8), k=5, ef=16)
    assert (labels == NO_LABEL).all()
    assert np.isinf(dists).all()


def test_k_larger_than_index_pads_the_rows():
    X = make_data(5, 8)
    idx = build(X)
    labels, dists, _ = idx.search(X[:1], k=20, ef=64)
    real = [x for x in labels[0].tolist() if x != NO_LABEL]
    assert len(real) == 5
    assert np.isinf(dists[0][5:]).all()


def test_single_element_index():
    X = make_data(1, 8)
    idx = build(X)
    labels, dists, _ = idx.search(X, k=1, ef=16)
    assert labels[0][0] == 0 and dists[0][0] == pytest.approx(0.0, abs=1e-4)


def test_dimension_mismatch_is_an_error():
    idx = build(make_data(100, 16))
    with pytest.raises(Exception):
        idx.search(make_data(2, 8), k=5)
    with pytest.raises(Exception):
        idx.add(make_data(2, 8), [200, 201])


def test_label_count_must_match_vector_count():
    idx = annlite.Index(dim=8)
    with pytest.raises(Exception):
        idx.add(make_data(5, 8), [1, 2])


def test_duplicate_vectors_do_not_break_the_graph():
    """Identical points make every distance a tie, which is where a sloppy
    neighbour heuristic collapses the graph."""
    X = np.repeat(make_data(20, 8), 25, axis=0)
    idx = build(X)
    Q = X[:10]
    got, _, _ = idx.search(Q, k=10, ef=128)
    truth, _ = idx.brute_force(Q, k=10)
    assert recall(got, truth, 10) >= 0.9


def test_non_contiguous_input_is_handled():
    X = make_data(200, 32)
    idx = build(X)
    view = np.asfortranarray(X[:10])  # forces a copy through forcecast
    labels, _, _ = idx.search(view, k=5, ef=64)
    assert labels.shape == (10, 5)


def test_arbitrary_labels_not_just_indices():
    X = make_data(300, 16)
    labels_in = [10_000_000 + i * 7 for i in range(300)]
    idx = build(X, labels=labels_in)
    got, _, _ = idx.search(X[:5], k=1, ef=64)
    assert [int(g[0]) for g in got] == labels_in[:5]


def test_concurrent_searches_agree_with_serial():
    """The GIL is released during search, so this genuinely runs in parallel
    and would surface a data race in the shared visited pool."""
    X = make_data(3000, 32)
    idx = build(X)
    Q = make_data(200, 32, seed=21)
    serial, ds, _ = idx.search(Q, k=10, ef=100, threads=1)
    parallel, dp, _ = idx.search(Q, k=10, ef=100, threads=8)
    assert np.array_equal(serial, parallel)
    assert np.allclose(ds, dp, atol=1e-5)


def test_ef_search_property_is_the_default():
    X = make_data(1000, 16)
    idx = build(X)
    idx.ef_search = 200
    assert idx.ef_search == 200
    Q = make_data(30, 16, seed=31)
    truth, _ = idx.brute_force(Q, k=10)
    got, _, _ = idx.search(Q, k=10)  # no explicit ef, uses the property
    assert recall(got, truth, 10) >= 0.97


def test_stats_report_a_sane_graph():
    X = make_data(2000, 32)
    idx = build(X, M=16)
    st = idx.stats()
    assert st["elements"] == 2000
    assert st["levels"] >= 2, "a 2000 element index should build more than one layer"
    # layer 0 allows 2*M edges per node, and a healthy graph uses most of them
    assert 8 <= st["mean_degree_layer0"] <= 32
    assert st["memory_bytes"] > X.nbytes


def test_M_of_one_is_rejected():
    with pytest.raises(Exception):
        annlite.Index(dim=8, M=1)


def test_zero_dim_is_rejected():
    with pytest.raises(Exception):
        annlite.Index(dim=0)
