// RAG (retrieval-augmented generation): split documents into chunks, embed
// them with Ollama, store them in a cosine index, and answer questions from
// the closest chunks.

#include <algorithm>
#include <cctype>
#include <format>
#include <mutex>
#include <unordered_set>
#include <utility>

#include "server/server.hpp"

namespace vektor {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kEmbedBatch = 32;  // texts per request to Ollama
constexpr std::size_t kMaxTextBytes = 10'000;
constexpr std::size_t kDefaultEf = 100;

bool is_space(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

struct Hit {
    std::string id;
    json meta;  // doc_id, title, chunk, text
    float distance;
};

struct Retrieved {
    std::vector<Hit> hits;
    double embed_ms;
    double search_ms;
};

void require_ingested(const Store& rag) {
    std::shared_lock lock(rag.mutex);
    if (!rag.index) {
        throw HttpError(409, "nothing ingested yet: POST documents to /rag/ingest first");
    }
}

// Embeds the query, then finds the k closest chunks.
Retrieved retrieve(const Store& rag, const OllamaClient& ollama, const std::string& model,
                   const std::string& query, std::size_t k, std::size_t ef, bool exact) {
    require_ingested(rag);
    auto start = Clock::now();
    const std::vector<float> vector = ollama.embed(model, {query}).front();
    const double embed_ms = ms_since(start);

    std::shared_lock lock(rag.mutex);
    if (!rag.index) {
        throw HttpError(409, "nothing ingested yet");
    }
    const Index& index = *rag.index;
    start = Clock::now();
    const std::vector<Result> found =
        exact ? index.search_exact(vector, k) : index.search(vector, k, ef);
    const double search_ms = ms_since(start);
    Retrieved out{.hits = {}, .embed_ms = embed_ms, .search_ms = search_ms};
    for (const Result& r : found) {
        out.hits.push_back({.id = index.id(r.row),
                            .meta = json::parse(index.meta(r.row)),
                            .distance = r.distance});
    }
    return out;
}

struct Chunk {
    std::string id;  // "<document id>#<chunk number>"
    json meta;       // doc_id, title, chunk, text
};

struct Split {
    std::vector<Chunk> chunks;
    std::vector<std::string> doc_ids;
};

// Checks the documents of an ingest request and splits them into chunks.
Split split_documents(const json& docs, const ServerConfig& config) {
    if (!docs.is_array() || docs.empty()) {
        throw HttpError(400, "'documents' must be a non-empty array");
    }
    Split out;
    std::unordered_set<std::string> seen;
    for (std::size_t i = 0; i < docs.size(); ++i) {
        try {
            const json& doc = docs[i];
            if (!doc.is_object()) {
                throw HttpError(400, "each document must be a JSON object");
            }
            const std::string id = string_field(doc, "id", 200);
            const std::string title = doc.contains("title") ? string_field(doc, "title", 1000) : "";
            const std::string text = string_field(doc, "text", config.max_body_bytes);
            if (id.empty()) {
                throw HttpError(400, "'id' must not be empty");
            }
            if (!seen.insert(id).second) {
                throw HttpError(409, std::format("document '{}' appears twice", id));
            }
            out.doc_ids.push_back(id);
            const auto pieces = chunk_words(text, config.chunk_words, config.overlap_words);
            for (std::size_t c = 0; c < pieces.size(); ++c) {
                out.chunks.push_back(
                    {.id = std::format("{}#{}", id, c),
                     .meta = {
                         {"doc_id", id}, {"title", title}, {"chunk", c}, {"text", pieces[c]}}});
            }
        } catch (const HttpError& e) {
            throw HttpError(e.status(), std::format("documents[{}]: {}", i, e.what()));
        }
    }
    if (out.chunks.empty()) {
        throw HttpError(400, "the documents contain no text");
    }
    return out;
}

// Embeds the chunks' text with Ollama, kEmbedBatch texts per request.
std::vector<std::vector<float>> embed_chunks(const OllamaClient& ollama, const std::string& model,
                                             const std::vector<Chunk>& chunks) {
    std::vector<std::vector<float>> vectors;
    vectors.reserve(chunks.size());
    for (std::size_t first = 0; first < chunks.size(); first += kEmbedBatch) {
        std::vector<std::string> texts;
        for (std::size_t i = first; i < std::min(first + kEmbedBatch, chunks.size()); ++i) {
            texts.push_back(chunks[i].meta["text"].get<std::string>());
        }
        for (auto& v : ollama.embed(model, texts)) {
            vectors.push_back(std::move(v));
        }
    }
    return vectors;
}

}  // namespace

std::vector<std::string> chunk_words(std::string_view text, std::size_t size, std::size_t overlap) {
    if (size == 0 || overlap >= size) {
        throw std::invalid_argument("chunk size must be larger than the overlap");
    }
    std::vector<std::string_view> words;
    for (std::size_t i = 0; i < text.size();) {
        while (i < text.size() && is_space(text[i])) {
            ++i;
        }
        const std::size_t start = i;
        while (i < text.size() && !is_space(text[i])) {
            ++i;
        }
        if (i > start) {
            words.push_back(text.substr(start, i - start));
        }
    }
    std::vector<std::string> chunks;
    for (std::size_t first = 0; first < words.size(); first += size - overlap) {
        const std::size_t last = std::min(first + size, words.size());
        std::string chunk;
        for (std::size_t w = first; w < last; ++w) {
            if (w > first) {
                chunk += ' ';
            }
            chunk += words[w];
        }
        chunks.push_back(std::move(chunk));
        if (last == words.size()) {
            break;
        }
    }
    return chunks;
}

std::string build_prompt(const std::string& question, const std::vector<RagSource>& sources) {
    std::string prompt =
        "Answer the question using only the numbered sources below. Cite the sources you use "
        "by number, like [1]. If the sources do not contain the answer, reply exactly: "
        "I don't know.\n\nSources:\n\n";
    for (std::size_t i = 0; i < sources.size(); ++i) {
        prompt += std::format("[{}] {}\n{}\n\n", i + 1, sources[i].title, sources[i].text);
    }
    prompt += std::format("Question: {}\nAnswer:", question);
    return prompt;
}

json Api::rag_ingest(const json& body) {
    const Split split = split_documents(field(body, "documents"), config_);
    {
        // Fail before the slow embedding step if a document is already stored.
        std::shared_lock lock(rag_.mutex);
        for (const std::string& id : split.doc_ids) {
            if (rag_.index && rag_.index->contains(id + "#0")) {
                throw HttpError(409, std::format("document '{}' is already ingested", id));
            }
        }
    }
    // Embed without holding the lock, so searches keep running meanwhile.
    const auto start = Clock::now();
    const std::vector<std::vector<float>> vectors =
        embed_chunks(ollama_, config_.embed_model, split.chunks);
    const double embed_ms = ms_since(start);

    std::unique_lock lock(rag_.mutex);
    if (!rag_.index) {
        rag_.index.emplace(vectors.front().size(), Metric::Cosine);
    }
    Index& index = *rag_.index;
    // Check everything before adding anything.
    for (std::size_t i = 0; i < split.chunks.size(); ++i) {
        try {
            (void)prepare_vector(vectors[i], index.dim(), index.metric());
        } catch (const std::invalid_argument& e) {
            throw HttpError(502, std::format("unusable embedding from Ollama ({}); was the "
                                             "embedding model changed?",
                                             e.what()));
        }
        if (index.contains(split.chunks[i].id)) {
            throw HttpError(409, std::format("chunk '{}' is already ingested", split.chunks[i].id));
        }
    }
    for (std::size_t i = 0; i < split.chunks.size(); ++i) {
        index.add(split.chunks[i].id, vectors[i], split.chunks[i].meta.dump());
    }
    if (const auto dir = rag_.path.parent_path(); !dir.empty()) {
        std::filesystem::create_directories(dir);
    }
    index.save(rag_.path);
    return {{"documents", split.doc_ids.size()},
            {"chunks", split.chunks.size()},
            {"size", index.size()},
            {"embed_ms", embed_ms}};
}

json Api::rag_search(const json& body) const {
    const std::string query = string_field(body, "query", kMaxTextBytes);
    const std::size_t k = count_field(body, "k", 5, 1, 100);
    const std::size_t ef = count_field(body, "ef_search", kDefaultEf, 1, 10000);
    const bool exact = bool_field(body, "exact", false);
    const Retrieved r = retrieve(rag_, ollama_, config_.embed_model, query, k, ef, exact);
    json results = json::array();
    for (const Hit& hit : r.hits) {
        json item = hit.meta;
        item["id"] = hit.id;
        item["distance"] = hit.distance;
        item["score"] = 1.0F - hit.distance;  // cosine similarity
        results.push_back(std::move(item));
    }
    return {
        {"results", results}, {"took_ms", r.search_ms}, {"embed_ms", r.embed_ms}, {"exact", exact}};
}

json Api::rag_ask(const json& body) const {
    if (config_.llm_model.empty()) {
        throw HttpError(503,
                        "answers are turned off (low-space mode): start vektor-server with "
                        "--llm-model to use /rag/ask");
    }
    const std::string question = string_field(body, "question", kMaxTextBytes);
    const std::size_t k = count_field(body, "k", 5, 1, 20);
    const Retrieved r =
        retrieve(rag_, ollama_, config_.embed_model, question, k, kDefaultEf, false);

    std::vector<RagSource> sources;
    json sources_json = json::array();
    for (std::size_t i = 0; i < r.hits.size(); ++i) {
        const Hit& hit = r.hits[i];
        sources.push_back({.title = hit.meta["title"].get<std::string>(),
                           .text = hit.meta["text"].get<std::string>()});
        json item = hit.meta;
        item["n"] = i + 1;
        item["id"] = hit.id;
        item["score"] = 1.0F - hit.distance;
        sources_json.push_back(std::move(item));
    }
    const auto start = Clock::now();
    const std::string answer = ollama_.generate(config_.llm_model, build_prompt(question, sources));
    const double llm_ms = ms_since(start);
    return {
        {"answer", answer},
        {"sources", sources_json},
        {"model", config_.llm_model},
        {"timings", {{"embed_ms", r.embed_ms}, {"search_ms", r.search_ms}, {"llm_ms", llm_ms}}}};
}

}  // namespace vektor
