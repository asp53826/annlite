# annlite

An HNSW vector index written from scratch in C++17, with hand-vectorized
distance kernels, a Python extension, an HTTP service, and a recall-vs-latency
benchmark against FAISS.

Approximate nearest neighbour search is the thing underneath every embedding
product, and "it's fast" and "it's accurate" are the same knob turned two
different ways. So the deliverable isn't a number, it's a curve.

```bash
pip install -r requirements-dev.txt
make                          # one clang++ invocation, no cmake
./scripts/fetch_data.sh       # SIFT, 5MB, ships exact ground truth
make test                     # 51 tests
make bench                    # the table below
```

## Results

SIFT-128 (10,000 vectors, 128 dims, 100 queries with exact ground truth from
the dataset), k=10, single threaded, M=16, ef_construction=200, best of 3, on
an M-series CPU.

| engine | ef | recall@10 | QPS | latency | distance comps/query |
|---|---|---|---|---|---|
| brute force (numpy BLAS) | - | 1.000 | 10,360 | 0.097 ms | 10,000 |
| annlite | 10 | 0.949 | 105,741 | 0.0095 ms | 265 |
| annlite | 20 | 0.985 | 65,228 | 0.0153 ms | 379 |
| annlite | 32 | 0.997 | 45,149 | 0.0221 ms | 501 |
| annlite | 96 | 1.000 | 18,319 | 0.0546 ms | 981 |

At 98.5% recall the index does **26x fewer distance computations** than an
exact scan and runs **6.3x faster than BLAS**, which is a higher bar than it
sounds — a `numpy` matmul against 10k vectors is genuinely quick, and plenty of
"vector databases" never actually beat it at this scale.

### Against FAISS

Matched `ef` isn't a fair comparison, since `ef` means slightly different
things to two implementations. What you actually care about is throughput at
the recall you need:

| target recall | annlite QPS | FAISS QPS | ratio |
|---|---|---|---|
| 0.90 | 110,426 | 135,310 | 0.82x |
| 0.95 | 96,339 | 109,354 | 0.88x |
| 0.99 | 55,379 | 54,031 | **1.02x** |
| 0.999 | 32,723 | 17,924 | **1.83x** |

**FAISS is 10-20% faster in the low-recall regime; annlite is at parity at 0.99
and 1.83x faster at 0.999.** The crossover is consistent and has a cause:
annlite gets more recall out of the same `ef` (0.949 vs 0.925 at ef=10, 0.997
vs 0.993 at ef=32), which means the neighbour selection heuristic is building a
better connected graph. That costs a little in the cheap regime and pays off
when you actually need high recall — which, for anything user-facing, you do.

I'm not going to claim to have beaten a library with a decade of tuning behind
it. Landing on the same Pareto frontier is the result.

### Threading

The GIL is released for the whole search, so a query batch really does run in
parallel (SIFT, ef=64, 2000 queries, 10-core machine):

| threads | QPS | speedup | recall |
|---|---|---|---|
| 1 | 25,470 | 1.00x | 0.9990 |
| 2 | 49,989 | 1.96x | 0.9990 |
| 4 | 95,285 | 3.74x | 0.9990 |
| 8 | 144,329 | 5.67x | 0.9990 |

Recall is identical at every thread count, which is the part that matters — see
the concurrency bug below.

## How HNSW works

Build a proximity graph over the vectors and search it greedily: from wherever
you are, hop to the neighbour closest to the query, repeat. Greedy descent on a
flat graph gets stuck in local minima, so stack the graph into layers where the
top holds a handful of nodes with very long edges and each layer down is
denser. Search enters at the top, crosses the space in a few hops, and drops a
layer each time it stops improving. Log-ish hops instead of a linear scan.

Two details decide whether it actually works.

**The neighbour selection heuristic.** Keeping each node's M nearest
neighbours is the obvious choice and it builds a badly connected graph: inside
a cluster every node picks the same nearby cluster members, and nothing links
one cluster to another. The heuristic (Algorithm 4 in the paper) instead keeps
a candidate only if it's closer to the node than to any neighbour already kept,
which preserves the long-range edges that make the graph navigable:

```cpp
for (const auto& [cd, cid] : work) {
  bool dominated = false;
  for (id_t kept : out)
    if (distance(vector_at(cid), vector_at(kept), dim_, metric) < cd) { dominated = true; break; }
  if (!dominated) out.push_back(cid);
}
```

This is where the FAISS recall gap comes from. It's also why the code backfills
with nearest rejects when the heuristic returns fewer than M — otherwise degree
collapses on dense clusters and the graph splits.

**Pruning on the back-link.** Edges are bidirectional, so inserting one can
push a neighbour past its degree cap. Re-running the heuristic on *that*
neighbour is what stops popular nodes from accumulating thousands of edges and
turning search into a scan.

### Memory layout

- vectors in one contiguous `float` block, row major, so a node's data is one
  cache line walk rather than a pointer chase
- layer 0 gets `2*M` edges, upper layers `M`. Layer 0 carries all the traffic
- levels are drawn from an exponential decay, so a node reaches level *l* with
  probability `(1/M)^l` and the top layers stay tiny
- deletes are tombstones. Deleted nodes are still traversed as waypoints, they
  just never enter the result set, so removing 40% of the index doesn't
  disconnect the survivors (there's a test for exactly that)

### The visited set

Every query needs "have I seen this node". A `std::unordered_set` allocates, and
clearing a bitset is O(n) per query when the query only touches a few hundred
nodes. Instead it's a generation counter: bump an integer, and a node counts as
visited if its mark equals the current generation. O(1) reset, zero allocation.

## SIMD

Distance is where essentially all the time goes, so the kernels are written by
hand — NEON on arm64, AVX on x86, scalar fallback everywhere else, selected at
compile time. CI builds on both Ubuntu (x86) and macOS (arm) so both real paths
get exercised.

Four independent accumulators rather than one, because the loop is latency
bound on the FMA dependency chain, not throughput bound:

```cpp
s0 = vfmaq_f32(s0, d0, d0);
s1 = vfmaq_f32(s1, d1, d1);
s2 = vfmaq_f32(s2, d2, d2);
s3 = vfmaq_f32(s3, d3, d3);
```

Cosine normalizes on insert so a comparison stays a single dot product instead
of recomputing two norms every time. The tail loop handles dimensions that
aren't a multiple of the vector width, and the tests check dims 7, 16, 33 and
128 against numpy specifically to catch it.

## A real bug the tests caught

The first version kept one `VisitedPool` as a member of the index. Searches
take a *shared* lock so they can run concurrently — and then all raced on that
one mutable pool. Single threaded everything passed; at 8 threads the results
came back full of duplicated labels.

```python
def test_concurrent_searches_agree_with_serial():
    serial,   ds, _ = idx.search(Q, k=10, ef=100, threads=1)
    parallel, dp, _ = idx.search(Q, k=10, ef=100, threads=8)
    assert np.array_equal(serial, parallel)
```

The fix is one `thread_local` pool per thread, reused across queries so it
still costs no allocation per search. Worth noting the failure mode: it never
crashed, it just silently returned wrong neighbours. Exactly the kind of thing
that ships and quietly degrades whatever is built on top of it.

## API

```python
import _annlite as annlite

idx = annlite.Index(dim=128, metric="l2", M=16, ef_construction=200)
idx.add(vectors, labels)                       # numpy (n, dim) float32
labels, dists, comps = idx.search(queries, k=10, ef=64, threads=4)
truth, _ = idx.brute_force(queries, k=10)      # exact, for measuring recall

idx.remove(label); idx.get_vector(label); label in idx; len(idx)
idx.save("index.ann"); idx = annlite.Index.load("index.ann")
```

Metrics: `l2`, `ip`, `cosine`. Result rows shorter than k are padded with
`2**64 - 1` so you always get a rectangular array and can still tell padding
from a real label.

### HTTP service

```bash
uvicorn python.server:app --port 8080
```

`POST /collections`, `POST /collections/{n}/points`, `POST /collections/{n}/query`,
`DELETE /collections/{n}/points`, `POST /collections/{n}/persist` and `/restore`,
`GET /collections/{n}/stats`, `GET /health`.

Reads run concurrently in a thread pool (the GIL is released in the bindings);
writes are serialized per collection since insert takes an exclusive lock
anyway. Query accepts `"exact": true` to run a brute force scan, so you can
measure real recall against production traffic rather than trusting a
benchmark.

Persistence writes to a temp file and renames, so a crash mid-save can't leave
a corrupt index behind. The format is versioned by magic bytes and load rejects
anything else.

## Tests

51 tests, `pytest`. The ones that earned their place:

- recall against exact brute force at dims 8, 32, 128
- brute-force distances match numpy exactly, at dims 7/16/33/128 — this is what
  validates the SIMD kernel including its tail loop
- higher `ef` never lowers recall (it's the accuracy knob, it must behave)
- an indexed vector finds itself first at distance 0
- **concurrent searches return byte-identical results to serial ones**
- deleting 40% of the index leaves >95% of survivors still findable
- save/load returns identical neighbours *and* identical distances
- load rejects a non-index file; save leaves no `.tmp` behind
- cosine ignores magnitude; inner product doesn't
- 500 duplicate vectors don't collapse the graph (every distance is a tie,
  which is where a sloppy heuristic falls over)
- empty index, k > size, dim mismatch, M=1, dim=0, non-contiguous input

## Being straight about what this is

- **This is HNSW only.** No IVF, no product quantization, no on-disk index. The
  vectors live in RAM as float32; there's no compression, so memory is
  `n * dim * 4` plus roughly `n * 2M * 4` for the graph.
- **Benchmarked at 10k vectors**, not 1M. The recall/latency shape holds, but
  don't read the absolute QPS as a claim about behaviour at scale — at 1M the
  graph stops fitting in cache and the ranking could shift.
- **Deletes are tombstones and never reclaim memory.** A workload that churns
  heavily will grow without bound. Real systems compact; this doesn't.
- **Inserts take an exclusive lock**, so writes don't scale across threads even
  though reads do. Concurrent insert with fine-grained per-node locking is a
  substantially harder problem.
- **`-march=native`** by default, so a binary built on one machine may not run
  on an older one.

## Layout

```
include/annlite/distance.hpp   NEON / AVX / scalar kernels, metric dispatch
include/annlite/hnsw.hpp       index, params, visited pool
src/hnsw.cpp                   search_layer, neighbour heuristic, insert, persistence
python/bindings.cpp            pybind11, releases the GIL, numpy in and out
python/server.py               FastAPI service
bench/benchmark.py             recall/latency curves vs brute force and FAISS
tests/                         51 tests
Makefile                       one clang++ invocation
```

The Makefile detects a broken Command Line Tools libc++ header directory (some
macOS installs ship a truncated one that clang prefers over the SDK's complete
copy) and points at the SDK explicitly when it finds it.
