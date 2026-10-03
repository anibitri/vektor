# Benchmarks

All numbers on this page were measured; the raw results are the CSV files in
[`results/`](../results/). Each CSV starts with the exact command, date, commit, CPU, RAM, OS,
compiler and compiler flags it was measured with. To reproduce:

```bash
make datasets   # download and convert SIFT (once)
make bench      # about 7 minutes on an Apple M2
make plots      # redraw the charts in docs/figures/
```

## Method

- **Dataset: SIFT-100k.** The first 100,000 base vectors and the first 1,000 queries of SIFT1M
  ([TEXMEX corpus](http://corpus-texmex.irisa.fr/)): 128 dimensions, L2 distance. The true
  nearest neighbours of each query were found by brute force when the dataset was converted.
- **recall@10:** the share of each query's true 10 nearest neighbours that the search returns,
  averaged over the 1,000 queries.
- **Queries per second:** one thread, one query at a time. Each setting first runs all queries
  once to warm up, then 5 timed passes; the pass with the median time is reported, so one pass
  slowed down by something else on the laptop does not skew the result. On macOS the benchmark
  asks for the highest thread priority, so it runs on a performance core rather than an
  efficiency core.
- **Latency:** mean and 99th percentile per query, from the same pass.
- **Memory:** `Index::memory_bytes()`, the bytes allocated for the vectors and the graph, worked
  out from the containers' capacities. It does not count the memory allocator's own overhead.
- **Runs:** every setting is measured 3 times, with seeds 42, 43 and 44, so each run builds a
  different random graph. Charts show the mean, with error bars from the lowest to the highest
  run.
- **HNSW settings:** `M` = 8, 16 or 32, `M_max0 = 2 × M`, `ef_construction = 200`, `k = 10`,
  `ef_search` = 10, 20, 40, 80, 160 and 320.

**Machine:** Apple M2 (8 cores: 4 performance, 4 efficiency), 8 GB RAM, macOS 26.6.2,
Apple clang 21.0.0, `-O3 -march=native -fopenmp-simd`.

## Recall vs speed

![Recall vs queries per second for M = 8, 16 and 32](figures/recall_vs_qps.png)

HNSW with `M = 16` (mean of 3 runs; speed-up compared with exact search measured in the same
session, 896 queries/s):

| ef_search | recall@10 | queries/s | speed-up vs exact | mean latency | p99 latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.797 | 42,405 | 47.3× | 0.024 ms | 0.046 ms |
| 20 | 0.907 | 26,989 | 30.1× | 0.037 ms | 0.062 ms |
| 40 | 0.967 | 15,972 | 17.8× | 0.063 ms | 0.092 ms |
| 80 | 0.992 | 9,238 | 10.3× | 0.108 ms | 0.154 ms |
| 160 | 0.999 | 5,198 | 5.8× | 0.193 ms | 0.285 ms |
| 320 | 1.000 | 2,969 | 3.3× | 0.337 ms | 0.500 ms |

Exact search: 896 queries/s, 1.12 ms per query.

- The spec's target, recall@10 ≥ 0.95, is reached at `ef_search = 40`, **17.8× faster** than
  exact search.
- Each doubling of `ef_search` cuts the speed by about 40%, and the recall gains shrink as
  recall approaches 1.
- A bigger `M` gives better recall at the same `ef_search`, but each step of the search checks
  more links. For the same recall, M = 16 and M = 32 are almost equally fast (M = 32 slightly
  ahead near 0.99); M = 8 is clearly slower once recall is above about 0.95.

## Choosing links: heuristic vs simple

![Heuristic vs simple neighbour selection](figures/select.png)

With `M = 16`, comparing the paper's heuristic (Algorithm 4) with simply keeping the M closest
candidates (Algorithm 3):

| ef_search | heuristic recall@10 | simple recall@10 (lowest to highest run) |
| ---: | ---: | ---: |
| 40 | 0.967 | 0.909 (0.850 to 0.939) |
| 80 | 0.992 | 0.946 (0.881 to 0.979) |
| 320 | 1.000 | 0.964 (0.895 to 0.998) |

Build time: 15.5 s with the heuristic, 13.3 s without (mean of 3 runs).

- Without the heuristic, recall levels off below 0.97 on average, even with a large
  `ef_search`. More searching does not help when part of the graph cannot be reached.
- The paper's explanation: the simple method links each node to its closest neighbours, which on
  clustered data like SIFT are all in the same cluster. Links between clusters get trimmed away,
  and a search that starts in the wrong cluster can get stuck. The heuristic keeps links that
  point in different directions, so it keeps those bridges. (We have not measured connectivity
  directly; the levelling-off and the seed dependence fit this explanation.)
- The simple method's results also depend heavily on the seed (0.895 to 0.998 recall at
  `ef_search = 320`): whether a bridge survives is down to the random insertion order.
- The heuristic costs about 2 s more to build, for the extra distance computations between
  candidates.

## Visited set: visit tags vs hash set

![Visit-tag array vs unordered_set](figures/visited.png)

Same index (`M = 16`), only the way a search remembers visited nodes changes. Visit tags are
**1.5 to 1.7 times faster** at every `ef_search` (44,218 vs 28,299 queries/s at `ef_search = 10`;
2,869 vs 1,697 at 320).

A `std::unordered_set` allocates memory for every node it stores, computes a hash, and spreads
its entries across memory. The visit-tag array is one array lookup and one write per node, with
no allocation and no clearing between searches (see [hnsw.md](hnsw.md#visit-tags-instead-of-a-hash-set)).
Both versions return exactly the same results; a test checks this.

## Memory vs M

![Bytes per vector and recall for each M](figures/memory_vs_m.png)

| M | bytes per vector | of which graph | index size (100k vectors) | recall@10 at ef_search = 40 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 636 | 124 | 63.6 MB | 0.919 |
| 16 | 697 | 185 | 69.8 MB | 0.967 |
| 32 | 824 | 312 | 82.5 MB | 0.981 |

- The vectors themselves take 512 bytes each (128 × 4 bytes), so the graph adds 24% (M = 8) to
  61% (M = 32) on top.
- Layer 0 holds up to `2 × M` links of 4 bytes each, plus one spare slot, plus a 24-byte
  `std::vector` header per list. The headers and the allocator's bookkeeping are why flattening
  layer 0 into one array (an optional extra) would save memory.
- The largest index here (82.5 MB) is well under the spec's 200 MB limit. The whole benchmark
  process peaked at 129 MB resident memory for M = 16, including the loaded dataset (measured
  once with `/usr/bin/time -l` at commit 8384f58).

## Limitations

- One dataset so far. GloVe, Fashion-MNIST, synthetic data and the intrinsic-dimension
  experiment come later.
- One laptop, one thread. Laptop timings depend on background load and temperature; the median
  of 5 passes and the run-to-run error bars show how much they moved.
- Memory is estimated from container capacities, not measured from the operating system.
- Recall compares row numbers. SIFT values are whole numbers, so two vectors can be exactly the
  same distance from a query; if they tie for 10th place, returning the "other" one counts as a
  miss.
- No comparison with hnswlib yet.
