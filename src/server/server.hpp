#pragma once

// Shared declarations for vektor-server: the REST API (routes.cpp), the RAG
// endpoints (rag.cpp) and the Ollama client (ollama_client.cpp).

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/index.hpp"

namespace httplib {
class Server;
}

namespace vektor {

using nlohmann::json;

// An error with the HTTP status code to answer with.
class HttpError : public std::runtime_error {
public:
    HttpError(int status, const std::string& message)
        : std::runtime_error(message), status_(status) {}
    [[nodiscard]] int status() const { return status_; }

private:
    int status_;
};

// An index shared by the server's threads. Searches take a shared lock (many
// at once); changes take a unique lock (one at a time, with no searches).
struct Store {
    mutable std::shared_mutex mutex;
    std::optional<Index> index;     // empty until created, loaded or ingested
    std::filesystem::path path;     // where the index is saved
    mutable std::mutex save_mutex;  // one save at a time (saving only needs the shared lock)
};

// Calls Ollama's HTTP API (https://github.com/ollama/ollama/blob/main/docs/api.md).
class OllamaClient {
public:
    explicit OllamaClient(std::string base_url) : base_url_(std::move(base_url)) {}

    // One embedding per text, in the same order. Throws HttpError(502) if
    // Ollama cannot be reached or returns an error.
    [[nodiscard]] std::vector<std::vector<float>> embed(
        const std::string& model, const std::vector<std::string>& texts) const;

    // The model's answer to the prompt (temperature 0, so answers repeat).
    [[nodiscard]] std::string generate(const std::string& model, const std::string& prompt) const;

private:
    [[nodiscard]] json post(const std::string& path, const json& body) const;

    std::string base_url_;
};

// Splits text into chunks of `size` words. Each chunk starts with the last
// `overlap` words of the previous one, so a sentence cut at a boundary is
// still whole in one of the two chunks.
std::vector<std::string> chunk_words(std::string_view text, std::size_t size, std::size_t overlap);

struct RagSource {
    std::string title;
    std::string text;
};

// The prompt sent to the LLM: numbered sources, then the question.
std::string build_prompt(const std::string& question, const std::vector<RagSource>& sources);

struct ServerConfig {
    std::filesystem::path data_dir = "data";  // index.vkt and rag.vkt are kept here
    std::string ollama_url = "http://localhost:11434";
    std::string embed_model = "all-minilm";
    std::string llm_model;  // empty: low-space mode, /rag/ask is turned off
    std::size_t chunk_words = 300;
    std::size_t overlap_words = 50;
    std::size_t max_body_bytes = std::size_t{64} << 20U;
    std::filesystem::path ui_dir;  // the built UI (ui/dist) to serve at /; empty: no UI
};

class Api {
public:
    // Loads data_dir/index.vkt and data_dir/rag.vkt if they exist.
    explicit Api(ServerConfig config);

    void register_routes(httplib::Server& server);

    [[nodiscard]] const ServerConfig& config() const { return config_; }
    [[nodiscard]] std::size_t index_size() const;
    [[nodiscard]] std::size_t rag_size() const;

private:
    // Vector index endpoints (routes.cpp).
    json create_index(const json& body);
    json add_vectors(const json& body);
    [[nodiscard]] json search(const json& body) const;
    [[nodiscard]] json stats() const;
    [[nodiscard]] json save() const;

    // RAG endpoints (rag.cpp).
    json rag_ingest(const json& body);
    [[nodiscard]] json rag_search(const json& body) const;
    [[nodiscard]] json rag_ask(const json& body) const;

    ServerConfig config_;
    OllamaClient ollama_;
    Store vectors_;
    Store rag_;
};

// Helpers for reading request fields, shared by routes.cpp and rag.cpp. They
// throw HttpError(400) with a message naming the field.
const json& field(const json& body, const char* name);
std::string string_field(const json& body, const char* name, std::size_t max_length);
std::size_t count_field(const json& body, const char* name, std::size_t fallback, std::size_t min,
                        std::size_t max);
bool bool_field(const json& body, const char* name, bool fallback);

double ms_since(std::chrono::steady_clock::time_point start);

}  // namespace vektor
