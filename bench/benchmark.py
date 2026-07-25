#!/usr/bin/env python3
"""Recall vs latency, against brute force and against FAISS.

A single recall number means nothing on its own. Any ANN index can hit 0.99
recall by searching wider, and any index can be fast by searching badly. The
only honest way to compare two of them is to sweep the accuracy knob and plot
the whole curve, which is what this does.

Ground truth is exact, either from the dataset's own file (SIFT ships one) or
computed by brute force here.

  python bench/benchmark.py                    # sift, vs faiss if installed
  python bench/benchmark.py --dataset random --n 50000 --dim 128
  python bench/benchmark.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import _annlite as annlite  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIFT = os.path.join(ROOT, "data", "siftsmall")


def read_fvecs(path: str) -> np.ndarray:
    """fvecs: each row is a little endian int32 dim followed by dim float32."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:].copy().view(np.float32)


def read_ivecs(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    return raw.reshape(-1, dim + 1)[:, 1:].copy()


def load_dataset(name: str, n: int, dim: int, nq: int, seed: int):
    if name == "sift":
        if not os.path.isdir(SIFT):
            sys.exit(f"{SIFT} missing, run scripts/fetch_data.sh")
        base = read_fvecs(os.path.join(SIFT, "siftsmall_base.fvecs"))
        query = read_fvecs(os.path.join(SIFT, "siftsmall_query.fvecs"))
        gt = read_ivecs(os.path.join(SIFT, "siftsmall_groundtruth.ivecs"))
        return base, query, gt, "sift-128-euclidean"

    rng = np.random.default_rng(seed)
    if name == "random":
        base = rng.standard_normal((n, dim)).astype(np.float32)
        query = rng.standard_normal((nq, dim)).astype(np.float32)
    elif name == "clustered":
        # closer to real embeddings than uniform noise, which is unrealistically
        # hard for a graph index because there is no structure to exploit
        centers = rng.standard_normal((max(n // 500, 8), dim)).astype(np.float32) * 4
        pick = rng.integers(0, len(centers), n)
        base = (centers[pick] + rng.standard_normal((n, dim)) * 0.7).astype(np.float32)
        qpick = rng.integers(0, len(centers), nq)
        query = (centers[qpick] + rng.standard_normal((nq, dim)) * 0.7).astype(np.float32)
    else:
        sys.exit(f"unknown dataset {name}")
    return base, query, None, f"{name}-{n}x{dim}"


def exact_ground_truth(base: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Chunked so a big base doesn't allocate an nq x n distance matrix."""
    out = np.zeros((len(query), k), dtype=np.int32)
    bsq = (base * base).sum(axis=1)
    step = max(1, 2_000_000 // max(len(base), 1))
    for i in range(0, len(query), step):
        q = query[i : i + step]
        d = bsq[None, :] - 2.0 * (q @ base.T)  # query norm is constant per row
        out[i : i + step] = np.argpartition(d, k, axis=1)[:, :k][
            np.arange(len(q))[:, None],
            np.argsort(np.take_along_axis(d, np.argpartition(d, k, axis=1)[:, :k], 1), axis=1),
        ]
    return out


def recall_at_k(got: np.ndarray, truth: np.ndarray, k: int) -> float:
    hits = 0
    for g, t in zip(got, truth):
        hits += len(set(g[:k].tolist()) & set(t[:k].tolist()))
    return hits / (len(got) * k)


def timed_search(index, query, k, ef, threads, repeat):
    best = None
    comps = 0
    labels = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        labels, _, comps = index.search(query, k=k, ef=ef, threads=threads)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return labels, best, comps


def run_annlite(base, query, gt, k, M, ef_construction, ef_values, threads, repeat):
    idx = annlite.Index(dim=base.shape[1], metric="l2", M=M, ef_construction=ef_construction)
    t0 = time.perf_counter()
    idx.add(base, list(range(len(base))))
    build = time.perf_counter() - t0
    st = idx.stats()

    rows = []
    for ef in ef_values:
        labels, dt, comps = timed_search(idx, query, k, ef, threads, repeat)
        rows.append({
            "engine": "annlite",
            "ef": ef,
            "recall": round(recall_at_k(labels, gt, k), 4),
            "qps": round(len(query) / dt, 1),
            "latency_ms": round(dt / len(query) * 1000, 4),
            "dist_comps_per_query": round(comps / len(query), 1),
            "build_seconds": round(build, 2),
            "memory_mb": round(st["memory_bytes"] / 1e6, 1),
        })
    return rows, st, idx


def run_bruteforce(base, query, k, repeat):
    """Exact scan in numpy, which is the honest floor: BLAS matmul is very fast
    and plenty of "vector databases" never actually beat it at small scale."""
    bsq = (base * base).sum(axis=1)
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        d = bsq[None, :] - 2.0 * (query @ base.T)
        np.argpartition(d, k, axis=1)[:, :k]
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return [{
        "engine": "brute force (numpy BLAS)",
        "ef": None,
        "recall": 1.0,
        "qps": round(len(query) / best, 1),
        "latency_ms": round(best / len(query) * 1000, 4),
        "dist_comps_per_query": len(base),
        "build_seconds": 0.0,
        "memory_mb": round(base.nbytes / 1e6, 1),
    }]


def run_faiss(base, query, gt, k, M, ef_construction, ef_values, repeat):
    try:
        import faiss
    except ImportError:
        return []

    index = faiss.IndexHNSWFlat(base.shape[1], M)
    index.hnsw.efConstruction = ef_construction
    faiss.omp_set_num_threads(1)
    t0 = time.perf_counter()
    index.add(base)
    build = time.perf_counter() - t0

    rows = []
    for ef in ef_values:
        index.hnsw.efSearch = ef
        best = None
        labels = None
        for _ in range(repeat):
            t1 = time.perf_counter()
            _, labels = index.search(query, k)
            dt = time.perf_counter() - t1
            best = dt if best is None else min(best, dt)
        rows.append({
            "engine": "faiss HNSWFlat",
            "ef": ef,
            "recall": round(recall_at_k(labels, gt, k), 4),
            "qps": round(len(query) / best, 1),
            "latency_ms": round(best / len(query) * 1000, 4),
            "dist_comps_per_query": None,
            "build_seconds": round(build, 2),
            "memory_mb": None,
        })
    return rows


def _fastest_at(rows, target_recall):
    """Highest qps among the settings that actually reach the target recall.

    This is the operating point you would pick in production: turn ef down as
    far as it goes without dropping below the accuracy you need.
    """
    ok = [r for r in rows if r["recall"] >= target_recall]
    return max(ok, key=lambda r: r["qps"]) if ok else None


def print_table(rows):
    cols = [("engine", 24, "s"), ("ef", 6, "s"), ("recall", 9, "s"),
            ("qps", 11, "s"), ("latency_ms", 12, "s"),
            ("dist_comps_per_query", 22, "s"), ("build_seconds", 15, "s")]
    head = "".join(n[:w].ljust(w + 1) if i == 0 else n[:w].rjust(w + 1)
                   for i, (n, w, _) in enumerate(cols))
    print(head)
    print("-" * len(head))
    for r in rows:
        line = ""
        for i, (n, w, _) in enumerate(cols):
            v = r.get(n)
            v = "-" if v is None else v
            line += str(v).ljust(w + 1) if i == 0 else str(v).rjust(w + 1)
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift", choices=["sift", "random", "clustered"])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef", default="10,20,40,80,160,320")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    base, query, gt, name = load_dataset(args.dataset, args.n, args.dim, args.nq, args.seed)
    if gt is None:
        gt = exact_ground_truth(base, query, args.k)
    gt = gt[:, : args.k]

    ef_values = [int(x) for x in args.ef.split(",")]
    print(f"dataset {name}: {len(base)} vectors x {base.shape[1]} dims, "
          f"{len(query)} queries, k={args.k}")
    print(f"annlite simd backend: {annlite.simd_backend()}, "
          f"M={args.M} ef_construction={args.ef_construction}, "
          f"threads={args.threads}, best of {args.repeat}")
    print()

    rows = run_bruteforce(base, query, args.k, args.repeat)
    a_rows, st, _ = run_annlite(base, query, gt, args.k, args.M, args.ef_construction,
                                ef_values, args.threads, args.repeat)
    rows += a_rows
    f_rows = run_faiss(base, query, gt, args.k, args.M, args.ef_construction,
                       ef_values, args.repeat)
    if f_rows:
        rows += f_rows
    else:
        print("(faiss not installed, skipping that comparison)\n")

    print_table(rows)
    print()
    print("index stats:", json.dumps(st))

    bf = rows[0]
    for r in a_rows:
        if r["recall"] >= 0.95:
            print(f"\nat recall {r['recall']}: {r['qps'] / bf['qps']:.1f}x brute force qps, "
                  f"{bf['dist_comps_per_query'] / r['dist_comps_per_query']:.1f}x fewer "
                  f"distance computations")
            break

    if f_rows:
        print("\nannlite vs faiss at matched ef:")
        for a, f in zip(a_rows, f_rows):
            print(f"  ef={a['ef']:<4} recall {a['recall']:.4f} vs {f['recall']:.4f}"
                  f"   qps {a['qps']:>8.1f} vs {f['qps']:>8.1f}"
                  f"   ({a['qps'] / f['qps']:.2f}x)")

        # matched ef is not really a fair comparison, ef means slightly
        # different things to two implementations. what a user actually cares
        # about is throughput at the recall they need, so compare the curves.
        print("\nqps at a target recall (the comparison that matters):")
        print(f"  {'target':>8}{'annlite':>12}{'faiss':>12}{'ratio':>10}")
        for target in (0.90, 0.95, 0.99, 0.999):
            a = _fastest_at(a_rows, target)
            f = _fastest_at(f_rows, target)
            if a is None or f is None:
                print(f"  {target:>8.3f}{'n/a':>12}{'n/a':>12}{'-':>10}")
                continue
            print(f"  {target:>8.3f}{a['qps']:>12.1f}{f['qps']:>12.1f}"
                  f"{a['qps'] / f['qps']:>9.2f}x")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"dataset": name, "args": vars(args), "stats": st, "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
