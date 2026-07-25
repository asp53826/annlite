#!/usr/bin/env python3
"""HTTP service around the index.

Named collections, insert/query/delete, and persistence. Writes are serialized
per collection because the C++ side takes an exclusive lock on insert anyway,
while reads run concurrently in a thread pool, which is the whole point of
releasing the GIL in the bindings.

  uvicorn python.server:app --port 8080
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _annlite as annlite  # noqa: E402

DATA_DIR = os.environ.get("ANNLITE_DATA", os.path.join(os.getcwd(), "annlite-data"))
DEFAULT_THREADS = int(os.environ.get("ANNLITE_THREADS", "4"))


class CreateCollection(BaseModel):
    name: str
    dim: int = Field(gt=0)
    metric: str = "l2"
    M: int = Field(default=16, ge=2)
    ef_construction: int = Field(default=200, ge=1)
    ef_search: int = Field(default=50, ge=1)


class InsertRequest(BaseModel):
    vectors: list[list[float]]
    ids: list[int] | None = None


class QueryRequest(BaseModel):
    vectors: list[list[float]]
    k: int = Field(default=10, ge=1)
    ef: int = Field(default=0, ge=0)
    exact: bool = False  # brute force, for checking recall in production


class DeleteRequest(BaseModel):
    ids: list[int]


class Collection:
    def __init__(self, spec: CreateCollection):
        self.spec = spec
        self.index = annlite.Index(
            dim=spec.dim, metric=spec.metric, M=spec.M,
            ef_construction=spec.ef_construction, ef_search=spec.ef_search,
        )
        # one writer at a time. reads don't take this, they rely on the
        # shared_mutex in the C++ index.
        self.write_lock = threading.Lock()
        self.next_id = 0

    def allocate_ids(self, n: int) -> list[int]:
        start = self.next_id
        self.next_id += n
        return list(range(start, start + n))


collections: dict[str, Collection] = {}
registry_lock = threading.Lock()

app = FastAPI(title="annlite", version="0.1.0")


def get(name: str) -> Collection:
    c = collections.get(name)
    if c is None:
        raise HTTPException(404, f"no collection named {name}")
    return c


def to_array(vectors: list[list[float]], dim: int) -> np.ndarray:
    if not vectors:
        raise HTTPException(400, "no vectors given")
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise HTTPException(400, f"expected vectors of dimension {dim}, got shape {arr.shape}")
    return arr


@app.get("/health")
async def health():
    return {"status": "ok", "simd": annlite.simd_backend(), "collections": len(collections)}


@app.get("/collections")
async def list_collections():
    return {
        name: {"size": len(c.index), "dim": c.spec.dim, "metric": c.spec.metric}
        for name, c in collections.items()
    }


@app.post("/collections", status_code=201)
async def create_collection(spec: CreateCollection):
    with registry_lock:
        if spec.name in collections:
            raise HTTPException(409, f"collection {spec.name} already exists")
        try:
            collections[spec.name] = Collection(spec)
        except Exception as e:
            raise HTTPException(400, str(e))
    return {"name": spec.name, "dim": spec.dim, "metric": spec.metric}


@app.delete("/collections/{name}")
async def drop_collection(name: str):
    with registry_lock:
        if collections.pop(name, None) is None:
            raise HTTPException(404, f"no collection named {name}")
    return {"dropped": name}


@app.get("/collections/{name}/stats")
async def stats(name: str):
    c = get(name)
    return {"name": name, **c.index.stats(), "ef_search": c.index.ef_search}


@app.post("/collections/{name}/points")
async def insert(name: str, req: InsertRequest):
    c = get(name)
    arr = to_array(req.vectors, c.spec.dim)
    if req.ids is not None and len(req.ids) != len(arr):
        raise HTTPException(400, f"{len(arr)} vectors but {len(req.ids)} ids")

    def work():
        with c.write_lock:
            ids = req.ids if req.ids is not None else c.allocate_ids(len(arr))
            c.index.add(arr, ids)
            return ids

    t0 = time.perf_counter()
    ids = await asyncio.to_thread(work)
    return {"inserted": len(ids), "ids": ids, "took_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.post("/collections/{name}/query")
async def query(name: str, req: QueryRequest):
    c = get(name)
    arr = to_array(req.vectors, c.spec.dim)

    def work():
        if req.exact:
            labels, dists = c.index.brute_force(arr, k=req.k)
            return labels, dists, 0
        return c.index.search(arr, k=req.k, ef=req.ef, threads=DEFAULT_THREADS)

    t0 = time.perf_counter()
    labels, dists, comps = await asyncio.to_thread(work)
    took = (time.perf_counter() - t0) * 1000

    sentinel = (1 << 64) - 1
    results = []
    for row_l, row_d in zip(labels.tolist(), dists.tolist()):
        hits = [{"id": i, "distance": d} for i, d in zip(row_l, row_d) if i != sentinel]
        results.append(hits)
    return {
        "results": results,
        "took_ms": round(took, 3),
        "distance_computations": int(comps),
        "exact": req.exact,
    }


@app.delete("/collections/{name}/points")
async def delete(name: str, req: DeleteRequest):
    c = get(name)

    def work():
        with c.write_lock:
            return sum(1 for i in req.ids if c.index.remove(i))

    removed = await asyncio.to_thread(work)
    return {"requested": len(req.ids), "removed": removed, "size": len(c.index)}


@app.post("/collections/{name}/persist")
async def persist(name: str):
    c = get(name)
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.ann")

    def work():
        with c.write_lock:
            c.index.save(path)
        return os.path.getsize(path)

    size = await asyncio.to_thread(work)
    return {"saved": path, "bytes": size}


@app.post("/collections/{name}/restore")
async def restore(name: str):
    path = os.path.join(DATA_DIR, f"{name}.ann")
    if not os.path.exists(path):
        raise HTTPException(404, f"no snapshot at {path}")

    def work():
        return annlite.Index.load(path)

    index = await asyncio.to_thread(work)
    with registry_lock:
        spec = CreateCollection(name=name, dim=index.dim)
        c = collections.get(name)
        if c is None:
            c = Collection(spec)
            collections[name] = c
        c.index = index
        c.next_id = 0
    return {"restored": name, "size": len(index), "dim": index.dim}


def main():
    import uvicorn

    uvicorn.run(app, host=os.environ.get("ANNLITE_HOST", "127.0.0.1"),
                port=int(os.environ.get("ANNLITE_PORT", "8080")))


if __name__ == "__main__":
    main()
