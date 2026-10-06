# RAG evaluation

How well does the RAG app find the right text, and where does the time go when it answers? All
numbers on this page were measured; the raw results are [`results/rag-eval.csv`](../results/rag-eval.csv)
(retrieval) and [`results/rag-ask.json`](../results/rag-ask.json) (every answer, with timings).
To reproduce, with Ollama running:

```bash
build/vektor-server --llm-model qwen2.5:1.5b &   # or make up
make rag-ingest                                   # fetch the corpus once and ingest it
make rag-eval                                     # about 2 minutes
```

## Setup

- **Corpus:** the 2,031 arXiv abstracts in information retrieval (cs.IR) submitted from January
  to June 2025, fetched by [`scripts/fetch_corpus.py`](../scripts/fetch_corpus.py) (arXiv's
  titles and abstracts are free to use under CC0). Each document is the title plus the abstract
  (median 185 words); chunks of 300 words with 50 words of overlap give 2,034 chunks, almost one
  per abstract.
- **Index:** `all-minilm` embeddings (384 dimensions) in a cosine HNSW index (`M = 16`,
  `ef_construction = 200`): 6.8 MB in memory, 6.4 MB on disk. Embedding the corpus with Ollama
  took 26.5 s.
- **Questions:** 40, in [`scripts/rag-questions.json`](../scripts/rag-questions.json). Each was
  written from one randomly chosen abstract (seed 7) and is labelled with that abstract's chunk;
  most avoid the paper's name, so retrieval has to work from meaning. They were drafted by Claude
  and checked against each abstract.
- **Measures:** recall@k is the share of questions whose labelled chunk is in the top k; MRR is
  the mean of 1 / rank (0 if not in the top 10); "vs exact" is the share of exact search's top 10
  that HNSW also returns.
- **Machine:** Apple M2, 8 GB RAM, macOS 26.6.2, Ollama 0.35.1, one request at a time; measured
  at commit c7f024a.

## Retrieval

| search | ef_search | recall@1 | recall@5 | recall@10 | MRR | vs exact | search time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact | | 0.925 | 1.000 | 1.000 | 0.963 | | 0.087 ms |
| HNSW | 10 | 0.900 | 0.975 | 0.975 | 0.938 | 0.908 | 0.047 ms |
| HNSW | 20 | 0.925 | 1.000 | 1.000 | 0.963 | 0.945 | 0.066 ms |
| HNSW | 40 | 0.925 | 1.000 | 1.000 | 0.963 | 0.978 | 0.102 ms |
| HNSW | 80 | 0.925 | 1.000 | 1.000 | 0.963 | 0.990 | 0.139 ms |
| HNSW | 160 | 0.925 | 1.000 | 1.000 | 0.963 | 0.995 | 0.161 ms |

- **Retrieval works well on this set.** Exact search puts the labelled abstract first for 37 of
  the 40 questions and in the top 5 for all of them.
- **HNSW loses nothing from `ef_search = 20` up.** At 10 it misses one question's chunk. Its
  full top 10 converges on exact search's more slowly (91% at `ef_search = 10`, 99% at 80), but
  the extra differences are below the answering chunk.
- **At this size, HNSW is not needed.** Exact search over 2,034 vectors takes 0.087 ms; HNSW is
  only faster at `ef_search` 10 to 20. Graph search pays off on larger collections: on SIFT's
  100,000 vectors it is 16.6 times faster than exact search (see [benchmarks.md](benchmarks.md)).
- **Weak matches are where HNSW struggles.** For a query far from every document, such as
  "quantum semantics of language" (best score 0.61, the rest about 0.3), HNSW at
  `ef_search = 10` returns only 2 of exact search's top 5 and needs 320 to match it. The
  [GIF in the README](../README.md) shows this.
- **Limitation:** the questions were written from the abstracts by the same party that built the
  system, and many contain distinctive terms, so they are easier than real users' questions.

## Answers and where the time goes

Every question asked through `/rag/ask` with `qwen2.5:1.5b` and the 5 closest chunks:

| | count (of 40) |
| --- | ---: |
| answer contains the expected fact (keyword check) | 39 |
| the answering chunk was among the 5 sources | 40 |
| answer cites a source by number, like [1] | 3 |
| answer is "I don't know" | 0 |

| stage | median time |
| --- | ---: |
| embedding the question (Ollama, `all-minilm`) | 27.1 ms |
| vector search (HNSW, `ef_search = 100`) | 0.67 ms |
| language model (`qwen2.5:1.5b`) | 2,821 ms |

- **The language model takes 99% of the time.** The vector search is about 0.02% of it, and
  embedding the question about 1%. Making answers faster means a smaller or faster model, or
  streaming the answer, not a faster index. (The search time is higher here than in the retrieval
  table because the language model keeps the CPU busy between requests.)
- **Answers are accurate but rarely cite.** The prompt asks for citations, but the 1.5-billion
  parameter model seldom gives them. The API returns the numbered sources anyway, and the UI shows
  them next to the answer.

### Prompt experiment

Four prompt wordings were compared on the same 40 questions plus 3 questions the corpus cannot
answer (a one-off run, not part of `make rag-eval`):

| prompt | cites a source | expected fact | "I don't know" on the 3 unanswerable |
| --- | ---: | ---: | ---: |
| instructions before the sources (used) | 3 | 38 | 3 |
| instructions after the sources | 40 | 31 | 2 |
| instructions after the sources, with an example answer | 40 | 18 | 2 |
| as used, plus "cite the source numbers" at the end | 37 | 36 | 2 |
| as used, plus all the rules repeated at the end | 31 | 12 | 3 |

Every wording that made the model cite also made it either answer from its own knowledge (it
wrote a sourdough recipe, with a wrong oven temperature, for a question the papers do not cover)
or refuse questions it could answer. Answering only from the sources matters more than the
citation format, so the app keeps the first prompt. A larger model would likely follow both rules;
that was not tested.
