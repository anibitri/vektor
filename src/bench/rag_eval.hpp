#pragma once

#include <string>

#include "core/flags.hpp"

namespace vektor {

// `vektor-bench rag-eval`: measures the RAG app's retrieval through a running
// vektor-server, so the real embedding model and API are used. For each
// labelled question it finds the rank of the chunk that answers it, with exact
// search and with HNSW at each ef_search, and reports recall@1, recall@5,
// recall@10 and MRR, plus how much of exact search's top 10 HNSW returns. With --ask-out it also
// asks every question with /rag/ask and saves the answers and timings. `about` starts the results
// file (command, date, machine).
int rag_eval(Flags& flags, const std::string& about);

}  // namespace vektor
