# HNSW in Vektor

This page maps the HNSW paper to the code, so you can read them side by side:
Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical
Navigable Small World graphs* (IEEE TPAMI, 2018; [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)).

All of the graph code is in [`src/core/hnsw.cpp`](../src/core/hnsw.cpp). Storage, input checks
and exact search are in [`src/core/index.cpp`](../src/core/index.cpp).

## The idea in one paragraph

Every vector is a node in a graph, linked to some of its nearest neighbours. Searching means
starting somewhere and repeatedly moving to whichever neighbour is closer to the query. On one
big graph that walk takes many small steps, so HNSW stacks several graphs in layers. Every node
is on layer 0; a random few are also on layer 1, fewer still on layer 2, and so on. The upper
layers are sparse, so their links are long: a search crosses most of the space in a few hops up
there, then drops down a layer at a time and finishes with a careful search on layer 0.

## Paper to code

| Paper | Code | Notes |
| --- | --- | --- |
| Random layer: `l = floor(-ln(unif(0..1)) * mL)`, `mL = 1 / ln(M)` | `Index::random_level` | The uniform number is built from raw random bits, so every compiler's standard library gives the same layers for the same seed. |
| Algorithm 1, INSERT | `Index::add` | As in the paper, all `ef_construction` nodes found on one layer become the entry points for the layer below. |
| Algorithm 2, SEARCH-LAYER | `Index::search_layer` | A min-heap of nodes to explore and a max-heap of the best `ef` found. It is a template over the visited set (see below). |
| SEARCH-LAYER with `ef = 1` on the upper layers | `Index::greedy_closest` | With only one result needed, a plain loop does the same job: move to a closer neighbour until none is closer. |
| Algorithm 3, SELECT-NEIGHBORS-SIMPLE | `Index::select_neighbours` with `heuristic = false` | Take the `M` closest candidates. |
| Algorithm 4, SELECT-NEIGHBORS-HEURISTIC | `Index::select_neighbours` with `heuristic = true` | Keep a candidate only if it is closer to the new node than to every link already kept. Like hnswlib, it leaves out the paper's two optional extras (`extendCandidates`, `keepPrunedConnections`). |
| Algorithm 1, lines 12 to 16: shrink a neighbour's links when it has too many | `Index::link_back` | Re-chooses that node's links with the same method as for a new node. |
| Algorithm 5, K-NN-SEARCH | `Index::search` | Greedy descent to layer 0, then SEARCH-LAYER with `ef = max(ef_search, k)`. |

When there are no more candidates than links allowed, both selection methods keep them all.

## The parameters

| Parameter | Default | What it controls |
| --- | --- | --- |
| `M` | 16 | Links per node on the upper layers, and links made for each new node. More links: better recall, more memory, slower build. |
| `M_max0` | 32 (= 2 × M) | Links per node on layer 0, the layer every node is on. The paper recommends 2 × M. |
| `ef_construction` | 200 | How many candidates are considered when linking a new node. Higher: a better graph and a slower build. |
| `ef_search` | per query | How many candidates a search keeps. Higher: better recall, slower search. This is the knob you turn at query time. |

## Visit tags instead of a hash set

A search must remember which nodes it has already seen. The obvious choice is a
`std::unordered_set`, but every insert allocates memory and hashes, and a new set is needed for
every search.

Instead, Vektor keeps one `std::vector<std::uint32_t>` with a slot per node, plus a counter.
Each search increases the counter, and "visited" means "my slot holds the current counter
value". Starting a new search costs nothing, because old values simply stop matching. Only when
the counter wraps around (after about 4 billion searches) is the array cleared. The array lives
in a `thread_local`, so searches on different threads never share one.

[`docs/benchmarks.md`](benchmarks.md) measures the difference.

## Memory layout

- Vectors: one flat `std::vector<float>`, `4 × dim` bytes per vector.
- Graph: `links_[node][layer]` is a `std::vector<std::uint32_t>`. Node numbers are 32-bit, half
  the size of `std::size_t`. Each list reserves one spare slot, because it briefly holds one link
  too many before it is trimmed; without that, it would double its capacity and waste memory.
- The nested vectors are simple but cost a 24-byte header per list plus the allocator's own
  overhead. Flattening layer 0 into one fixed-stride array would remove that (a possible extra).

## Thread safety

`search` and `search_exact` only read the index, so many can run at once. `add` changes it, so
it must not run at the same time as anything else. The REST server enforces this with a
`std::shared_mutex`: searches take a shared lock, adds take a unique lock. Finer-grained locking
(per node, as hnswlib does) would let adds run alongside searches; that is future work.
