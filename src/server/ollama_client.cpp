// Calls Ollama's /api/embed and /api/generate endpoints with cpp-httplib.

#include <httplib.h>

#include <format>

#include "server/server.hpp"

namespace vektor {

json OllamaClient::post(const std::string& path, const json& body) const {
    httplib::Client client(base_url_);
    client.set_connection_timeout(std::chrono::seconds(5));
    client.set_read_timeout(std::chrono::minutes(5));  // a small LLM on a laptop can be slow
    const auto res = client.Post(path, body.dump(), "application/json");
    if (!res) {
        throw HttpError(502, std::format("cannot reach Ollama at {} ({}): is it running?",
                                         base_url_, httplib::to_string(res.error())));
    }
    const json reply = json::parse(res->body, nullptr, /*allow_exceptions=*/false);
    if (res->status != 200) {
        const bool has_message =
            reply.is_object() && reply.contains("error") && reply["error"].is_string();
        throw HttpError(502,
                        std::format("Ollama answered {} to {}: {}", res->status, path,
                                    has_message ? reply["error"].get<std::string>() : res->body));
    }
    if (reply.is_discarded() || !reply.is_object()) {
        throw HttpError(502, std::format("Ollama sent invalid JSON for {}", path));
    }
    return reply;
}

std::vector<std::vector<float>> OllamaClient::embed(const std::string& model,
                                                    const std::vector<std::string>& texts) const {
    const json reply = post("/api/embed", {{"model", model}, {"input", texts}});
    const auto it = reply.find("embeddings");
    if (it == reply.end() || !it->is_array() || it->size() != texts.size()) {
        throw HttpError(502, "Ollama sent the wrong number of embeddings");
    }
    try {
        return it->get<std::vector<std::vector<float>>>();
    } catch (const json::exception&) {
        throw HttpError(502, "Ollama sent malformed embeddings");
    }
}

std::string OllamaClient::generate(const std::string& model, const std::string& prompt) const {
    const json reply = post("/api/generate", {{"model", model},
                                              {"prompt", prompt},
                                              {"stream", false},
                                              {"options", {{"temperature", 0}}}});
    const auto it = reply.find("response");
    if (it == reply.end() || !it->is_string()) {
        throw HttpError(502, "Ollama's answer has no 'response' text");
    }
    return it->get<std::string>();
}

}  // namespace vektor
