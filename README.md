# Vektor

[![CI](https://github.com/anibitri/vektor/actions/workflows/ci.yml/badge.svg)](https://github.com/anibitri/vektor/actions/workflows/ci.yml)

Vektor is a small vector search engine written in C++20. Given a query vector, it finds the
closest stored vectors, either exactly (brute force) or approximately with HNSW, a graph-based
algorithm implemented here from scratch. The plan is to benchmark it on standard datasets, serve
it over a REST API, and use it to power a small RAG (retrieval-augmented generation) app.

**Status: early work in progress.** Done so far: exact search, the benchmark dataset format and
the SIFT-100k dataset. Next: HNSW, then the REST server, the RAG app and the UI.

## Things to change

A temporary checklist of things only you can do. Delete this section when it's done.

- [ ] Create the GitHub repo `anibitri/vektor` and push. CI has not run yet, so check that the
      first run passes.

## Build and test

You need CMake 3.20 or newer and a C++20 compiler. GoogleTest is downloaded by CMake, so there is
nothing else to install.

```bash
make build       # Release build, tuned for this CPU (-march=native)
make test        # run the tests
make test-asan   # run the tests under AddressSanitizer + UndefinedBehaviorSanitizer
make format      # format the code with clang-format
make lint        # check the code with clang-tidy
make clean       # delete build files and downloaded datasets
```

CI runs on every push: formatting, clang-tidy, Release builds with GCC 14 and Clang 21, and a
Debug build with AddressSanitizer + UndefinedBehaviorSanitizer.

## Datasets

`make datasets` downloads each dataset, converts it to a `.vkd` file in `data/`, and then deletes
the download. `data/` is git-ignored.

| Dataset   | Base vectors                 | Queries     | Dim | Metric | `.vkd` size |
| --------- | ---------------------------- | ----------- | --- | ------ | ----------- |
| SIFT-100k | first 100,000 of SIFT1M      | first 1,000 | 128 | L2     | 52.1 MB     |

A `.vkd` file holds the base vectors, the query vectors, and the true 100 nearest neighbours of
each query. These are found once by brute force when the file is made, so benchmarks never have
to recompute them. The layout, all little-endian:

```
[magic "VKD1"][u32 dim][u32 n_base][u32 n_query][u32 gt_k][u32 metric: 0 = L2, 1 = cosine]
[float32 base vectors][float32 query vectors][u32 true neighbours, gt_k per query]
```

To measure exact search on a dataset:

```bash
build/vektor-bench run --data data/sift-100k.vkd
```

## How it works so far

- **Storage.** All vectors live in one flat `std::vector<float>`; row `i` is values
  `i * dim` to `(i + 1) * dim`. That is exactly `4 × dim` bytes per vector, with nothing in
  between, which is friendly to the CPU cache.
- **Distances.** L2 is the squared Euclidean distance: the square root is skipped because it does
  not change which vector is closest. For cosine, each vector is scaled to length 1 when it is
  added, so the distance is just `1 − dot product`.
- **SIMD.** The distance functions are plain loops, but a plain loop is not enough for the
  compiler to use SIMD for the sum: floating-point addition is not associative, so by default it
  must add the numbers one at a time, in order. A `#pragma omp simd reduction(+ : sum)` line (with
  the `-fopenmp-simd` flag, which enables only these hints, not threads) allows it to keep several
  partial sums at once. Checked in the assembly on an Apple M2: without the pragma the sum is 16
  scalar `fadd` instructions per 16 values; with it, four vector `fmla.4s` instructions.
- **Exact search** computes the distance to every vector and keeps the best `k` in a max-heap,
  so the worst of the current best is always on top, ready to be replaced.

## Licence

MIT. See [LICENSE](LICENSE).
