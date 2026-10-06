// The REST API: routes, request checks, and the vector index endpoints.
// Every response is JSON; errors look like {"error": "..."}.

#include <httplib.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <mutex>
#include <unordered_set>
#include <utility>

#include "server/server.hpp"

namespace vektor {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxDim = 65536;
constexpr std::size_t kMaxK = 1000;
constexpr std::size_t kMaxEf = 10000;
constexpr std::size_t kMaxIdLength = 256;
constexpr std::size_t kDefaultEf = 100;

void reply(httplib::Response& res, int status, const json& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
}

// Runs a handler and turns its result, or the exception it throws, into a response.
template <typename Handler>
void respond(httplib::Response& res, int ok_status, Handler&& handler) {
    try {
        reply(res, ok_status, std::forward<Handler>(handler)());
    } catch (const HttpError& e) {
        reply(res, e.status(), {{"error", e.what()}});
    } catch (const json::exception& e) {
        reply(res, 400, {{"error", std::string("bad request: ") + e.what()}});
    } catch (const std::invalid_argument& e) {
        reply(res, 400, {{"error", e.what()}});
    } catch (const std::exception& e) {
        reply(res, 500, {{"error", e.what()}});
    }
}

json parse_body(const httplib::Request& req) {
    json body = json::parse(req.body);
    if (!body.is_object()) {
        throw HttpError(400, "the request body must be a JSON object");
    }
    return body;
}

std::vector<float> vector_field(const json& body, const char* name) {
    const json& value = field(body, name);
    if (!value.is_array()) {
        throw HttpError(400, std::format("'{}' must be an array of numbers", name));
    }
    std::vector<float> out;
    out.reserve(value.size());
    for (const json& x : value) {
        if (!x.is_number()) {
            throw HttpError(400, std::format("'{}' must be an array of numbers", name));
        }
        out.push_back(x.get<float>());  // values too big for a float become Inf: rejected later
    }
    return out;
}

json parse_meta(std::string_view meta) { return meta.empty() ? json(nullptr) : json::parse(meta); }

json index_stats(const Store& store) {
    std::shared_lock lock(store.mutex);
    if (!store.index) {
        return nullptr;
    }
    const Index& index = *store.index;
    return {{"size", index.size()},
            {"dim", index.dim()},
            {"metric", std::string(to_string(index.metric()))},
            {"params",
             {{"M", index.params().M},
              {"M_max0", index.params().M_max0},
              {"ef_construction", index.params().ef_construction}}},
            {"memory_bytes", index.memory_bytes()}};
}

}  // namespace

const json& field(const json& body, const char* name) {
    const auto it = body.find(name);
    if (it == body.end() || it->is_null()) {
        throw HttpError(400, std::format("missing field '{}'", name));
    }
    return *it;
}

std::string string_field(const json& body, const char* name, std::size_t max_length) {
    const json& value = field(body, name);
    if (!value.is_string()) {
        throw HttpError(400, std::format("'{}' must be a string", name));
    }
    std::string s = value.get<std::string>();
    if (s.size() > max_length) {
        throw HttpError(400, std::format("'{}' is longer than {} bytes", name, max_length));
    }
    return s;
}

std::size_t count_field(const json& body, const char* name, std::size_t fallback, std::size_t min,
                        std::size_t max) {
    const auto it = body.find(name);
    if (it == body.end() || it->is_null()) {
        return fallback;
    }
    if (!it->is_number_unsigned() || it->get<std::uint64_t>() < min ||
        it->get<std::uint64_t>() > max) {
        throw HttpError(400,
                        std::format("'{}' must be a whole number from {} to {}", name, min, max));
    }
    return it->get<std::size_t>();
}

bool bool_field(const json& body, const char* name, bool fallback) {
    const auto it = body.find(name);
    if (it == body.end() || it->is_null()) {
        return fallback;
    }
    if (!it->is_boolean()) {
        throw HttpError(400, std::format("'{}' must be true or false", name));
    }
    return it->get<bool>();
}

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Api::Api(ServerConfig config) : config_(std::move(config)), ollama_(config_.ollama_url) {
    vectors_.path = config_.data_dir / "index.vkt";
    rag_.path = config_.data_dir / "rag.vkt";
    for (Store* store : {&vectors_, &rag_}) {
        if (std::filesystem::exists(store->path)) {
            store->index = Index::load(store->path);
        }
    }
}

std::size_t Api::index_size() const {
    std::shared_lock lock(vectors_.mutex);
    return vectors_.index ? vectors_.index->size() : 0;
}

std::size_t Api::rag_size() const {
    std::shared_lock lock(rag_.mutex);
    return rag_.index ? rag_.index->size() : 0;
}

void Api::register_routes(httplib::Server& server) {
    server.set_payload_max_length(config_.max_body_bytes);
    if (!config_.ui_dir.empty() && !server.set_mount_point("/", config_.ui_dir.string())) {
        throw std::invalid_argument("UI folder not found: " + config_.ui_dir.string());
    }
    server.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        reply(res, 200, {{"status", "ok"}});
    });
    server.Get("/stats", [this](const httplib::Request&, httplib::Response& res) {
        respond(res, 200, [&] { return stats(); });
    });
    server.Post("/index", [this](const httplib::Request& req, httplib::Response& res) {
        respond(res, 201, [&] { return create_index(parse_body(req)); });
    });
    server.Post("/vectors", [this](const httplib::Request& req, httplib::Response& res) {
        respond(res, 200, [&] { return add_vectors(parse_body(req)); });
    });
    server.Post("/search", [this](const httplib::Request& req, httplib::Response& res) {
        respond(res, 200, [&] { return search(parse_body(req)); });
    });
    server.Post("/save", [this](const httplib::Request&, httplib::Response& res) {
        respond(res, 200, [&] { return save(); });
    });
    server.Post("/rag/ingest", [this](const httplib::Request& req, httplib::Response& res) {
        respond(res, 200, [&] { return rag_ingest(parse_body(req)); });
    });
    server.Post("/rag/search", [this](const httplib::Request& req, httplib::Response& res) {
        respond(res, 200, [&] { return rag_search(parse_body(req)); });
    });
    server.Post("/rag/ask", [this](const httplib::Request& req, httplib::Response& res) {
        respond(res, 200, [&] { return rag_ask(parse_body(req)); });
    });
}

json Api::create_index(const json& body) {
    (void)field(body, "dim");
    const std::size_t dim = count_field(body, "dim", 0, 1, kMaxDim);
    const Metric metric = parse_metric(string_field(body, "metric", 16));
    const std::size_t m = count_field(body, "M", 16, 2, 256);
    const std::size_t ef_construction = count_field(body, "ef_construction", 200, 1, 4096);
    Index index(dim, metric, {.M = m, .M_max0 = 2 * m, .ef_construction = ef_construction});
    {
        std::unique_lock lock(vectors_.mutex);
        vectors_.index = std::move(index);  // replaces any existing index
    }
    return index_stats(vectors_);
}

json Api::add_vectors(const json& body) {
    const json& items = field(body, "items");
    if (!items.is_array() || items.empty()) {
        throw HttpError(400, "'items' must be a non-empty array");
    }
    struct Item {
        std::string id;
        std::vector<float> vector;
        std::string meta;
    };
    std::vector<Item> parsed;
    parsed.reserve(items.size());
    for (std::size_t i = 0; i < items.size(); ++i) {
        try {
            const json& item = items[i];
            if (!item.is_object()) {
                throw HttpError(400, "each item must be a JSON object");
            }
            Item it{.id = string_field(item, "id", kMaxIdLength),
                    .vector = vector_field(item, "vector"),
                    .meta = {}};
            if (it.id.empty()) {
                throw HttpError(400, "'id' must not be empty");
            }
            if (const auto meta = item.find("metadata"); meta != item.end() && !meta->is_null()) {
                if (!meta->is_object()) {
                    throw HttpError(400, "'metadata' must be a JSON object");
                }
                it.meta = meta->dump();
            }
            parsed.push_back(std::move(it));
        } catch (const HttpError& e) {
            throw HttpError(e.status(), std::format("items[{}]: {}", i, e.what()));
        }
    }

    std::unique_lock lock(vectors_.mutex);
    if (!vectors_.index) {
        throw HttpError(409, "no index yet: create one with POST /index");
    }
    Index& index = *vectors_.index;
    // Check every item before adding any, so a bad request changes nothing.
    std::unordered_set<std::string> new_ids;
    for (std::size_t i = 0; i < parsed.size(); ++i) {
        try {
            (void)prepare_vector(parsed[i].vector, index.dim(), index.metric());
        } catch (const std::invalid_argument& e) {
            throw HttpError(400, std::format("items[{}]: {}", i, e.what()));
        }
        if (index.contains(parsed[i].id) || !new_ids.insert(parsed[i].id).second) {
            throw HttpError(409, std::format("items[{}]: ID '{}' already exists", i, parsed[i].id));
        }
    }
    for (Item& item : parsed) {
        index.add(item.id, item.vector, std::move(item.meta));
    }
    return {{"added", parsed.size()}, {"size", index.size()}};
}

json Api::search(const json& body) const {
    const std::vector<float> query = vector_field(body, "vector");
    const std::size_t k = count_field(body, "k", 10, 1, kMaxK);
    const std::size_t ef = count_field(body, "ef_search", kDefaultEf, 1, kMaxEf);
    const bool exact = bool_field(body, "exact", false);

    std::shared_lock lock(vectors_.mutex);
    if (!vectors_.index) {
        throw HttpError(409, "no index yet: create one with POST /index");
    }
    const Index& index = *vectors_.index;
    const auto start = Clock::now();
    const std::vector<Result> found =
        exact ? index.search_exact(query, k) : index.search(query, k, ef);
    const double took_ms = ms_since(start);

    json results = json::array();
    for (const Result& r : found) {
        results.push_back({{"id", index.id(r.row)},
                           {"distance", r.distance},
                           {"metadata", parse_meta(index.meta(r.row))}});
    }
    return {{"results", results}, {"took_ms", took_ms}, {"exact", exact}};
}

json Api::stats() const {
    return {{"index", index_stats(vectors_)},
            {"rag", index_stats(rag_)},
            {"embed_model", config_.embed_model},
            // null in low-space mode, when /rag/ask is turned off
            {"llm_model", config_.llm_model.empty() ? json(nullptr) : json(config_.llm_model)}};
}

json Api::save() const {
    std::shared_lock lock(vectors_.mutex);
    if (!vectors_.index) {
        throw HttpError(409, "no index to save: create one with POST /index");
    }
    std::scoped_lock one_save_at_a_time(vectors_.save_mutex);
    if (const auto dir = vectors_.path.parent_path(); !dir.empty()) {
        std::filesystem::create_directories(dir);
    }
    const auto start = Clock::now();
    vectors_.index->save(vectors_.path);
    return {{"path", vectors_.path.string()},
            {"file_bytes", std::filesystem::file_size(vectors_.path)},
            {"memory_bytes", vectors_.index->memory_bytes()},
            {"took_ms", ms_since(start)}};
}

}  // namespace vektor
