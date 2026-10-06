// Tests for vektor-server: a real HTTP server on a free local port, called
// over HTTP. Ollama is replaced by a small fake HTTP server (see FakeOllama).

#include "server/server.hpp"

#include <gtest/gtest.h>
#include <httplib.h>

#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace vektor {
namespace {

// An httplib server on a free port of 127.0.0.1, running in a background thread.
class TestServer {
public:
    TestServer() = default;
    TestServer(const TestServer&) = delete;
    TestServer& operator=(const TestServer&) = delete;
    TestServer(TestServer&&) = delete;
    TestServer& operator=(TestServer&&) = delete;
    ~TestServer() {
        server.stop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void start() {
        port_ = server.bind_to_any_port("127.0.0.1");
        thread_ = std::thread([this] { server.listen_after_bind(); });
        server.wait_until_ready();
    }

    [[nodiscard]] std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }

    httplib::Server server;

private:
    int port_ = 0;
    std::thread thread_;
};

// Stands in for Ollama. Embeddings are a bag of words: each word adds 1 to one
// of 1024 slots chosen by a hash, so texts sharing words point the same way.
// The "LLM" answers with the number of sources it was given.
class FakeOllama {
public:
    FakeOllama() {
        server_.server.Post("/api/embed", [](const httplib::Request& req, httplib::Response& res) {
            const json body = json::parse(req.body);
            json embeddings = json::array();
            for (const json& text : body.at("input")) {
                embeddings.push_back(embed(text.get<std::string>()));
            }
            res.set_content(json{{"model", body.at("model")}, {"embeddings", embeddings}}.dump(),
                            "application/json");
        });
        server_.server.Post(
            "/api/generate", [](const httplib::Request& req, httplib::Response& res) {
                const std::string prompt = json::parse(req.body).at("prompt");
                int sources = 0;
                while (prompt.find(std::format("[{}]", sources + 1)) != std::string::npos) {
                    ++sources;
                }
                res.set_content(
                    json{{"response", std::format("fake answer from {} sources [1]", sources)},
                         {"done", true}}
                        .dump(),
                    "application/json");
            });
        server_.start();
    }

    [[nodiscard]] std::string url() const { return server_.url(); }

    static std::vector<float> embed(const std::string& text) {
        std::vector<float> v(1024, 0.0F);
        std::uint32_t hash = 2166136261U;  // FNV-1a, one word at a time
        for (const char c : text + " ") {
            if (std::isalnum(static_cast<unsigned char>(c)) != 0) {
                hash = (hash ^ static_cast<std::uint8_t>(std::tolower(c))) * 16777619U;
            } else if (hash != 2166136261U) {
                v[hash % v.size()] += 1.0F;
                hash = 2166136261U;
            }
        }
        return v;
    }

private:
    TestServer server_;
};

struct Reply {
    int status = 0;
    json body;
};

class ServerTest : public testing::Test {
protected:
    void SetUp() override {
        data_dir_ = std::filesystem::path(testing::TempDir()) /
                    (std::string("vektor_server_") +
                     testing::UnitTest::GetInstance()->current_test_info()->name());
        std::filesystem::remove_all(data_dir_);
        start(config());
    }

    [[nodiscard]] ServerConfig config() const {
        ServerConfig c;
        c.data_dir = data_dir_;
        c.ollama_url = ollama_.url();
        c.llm_model = "fake-llm";
        c.chunk_words = 10;
        c.overlap_words = 2;
        return c;
    }

    // (Re)starts the API server, e.g. to check that it reloads its files.
    void start(const ServerConfig& c) {
        server_.reset();
        api_ = std::make_unique<Api>(c);
        server_ = std::make_unique<TestServer>();
        api_->register_routes(server_->server);
        server_->start();
    }

    Reply call(const std::string& method, const std::string& path, const std::string& body = "") {
        httplib::Client client(server_->url());
        const auto res =
            method == "GET" ? client.Get(path) : client.Post(path, body, "application/json");
        if (!res) {
            ADD_FAILURE() << "request failed: " << httplib::to_string(res.error());
            return {};
        }
        return {.status = res->status,
                .body = json::parse(res->body, nullptr, /*allow_exceptions=*/false)};
    }

    Reply post(const std::string& path, const json& body) {
        return call("POST", path, body.dump());
    }
    Reply get(const std::string& path) { return call("GET", path); }

    void create_index(std::size_t dim, const std::string& metric = "l2") {
        ASSERT_EQ(post("/index", {{"dim", dim}, {"metric", metric}}).status, 201);
    }

    std::filesystem::path data_dir_;
    FakeOllama ollama_;
    std::unique_ptr<Api> api_;
    std::unique_ptr<TestServer> server_;  // declared after api_, so it stops first
};

TEST_F(ServerTest, Health) {
    const Reply r = get("/health");
    EXPECT_EQ(r.status, 200);
    EXPECT_EQ(r.body["status"], "ok");
}

TEST_F(ServerTest, CreateAddSearch) {
    const Reply created = post("/index", {{"dim", 3}, {"metric", "l2"}, {"M", 8}});
    EXPECT_EQ(created.status, 201);
    EXPECT_EQ(created.body["size"], 0);
    EXPECT_EQ(created.body["params"]["M_max0"], 16);

    const Reply added =
        post("/vectors", {{"items",
                           {{{"id", "a"}, {"vector", {0, 0, 0}}, {"metadata", {{"color", "red"}}}},
                            {{"id", "b"}, {"vector", {1, 0, 0}}},
                            {{"id", "c"}, {"vector", {5, 5, 5}}}}}});
    ASSERT_EQ(added.status, 200) << added.body;
    EXPECT_EQ(added.body["added"], 3);
    EXPECT_EQ(added.body["size"], 3);

    for (const bool exact : {false, true}) {
        const Reply r = post("/search", {{"vector", {0.9, 0, 0}}, {"k", 2}, {"exact", exact}});
        ASSERT_EQ(r.status, 200) << r.body;
        ASSERT_EQ(r.body["results"].size(), 2U);
        EXPECT_EQ(r.body["results"][0]["id"], "b");
        EXPECT_EQ(r.body["results"][1]["id"], "a");
        EXPECT_NEAR(r.body["results"][0]["distance"].get<double>(), 0.01, 1e-6);
        EXPECT_EQ(r.body["results"][1]["metadata"]["color"], "red");
        EXPECT_TRUE(r.body["results"][0]["metadata"].is_null());
        EXPECT_GE(r.body["took_ms"].get<double>(), 0.0);
        EXPECT_EQ(r.body["exact"], exact);
    }
    // k larger than the index: all vectors come back.
    EXPECT_EQ(post("/search", {{"vector", {0, 0, 0}}, {"k", 50}}).body["results"].size(), 3U);
}

TEST_F(ServerTest, RejectsBadRequestsWith400) {
    create_index(3);
    const std::vector<std::pair<std::string, json>> bad = {
        {"/search", json::object()},                        // no vector
        {"/search", {{"vector", {1, 2}}}},                  // wrong dimension
        {"/search", {{"vector", json::array()}}},           // empty vector
        {"/search", {{"vector", {1, "x", 3}}}},             // not a number
        {"/search", {{"vector", "1,2,3"}}},                 // not an array
        {"/search", {{"vector", {1e300, 0, 0}}}},           // too big for a float: Inf
        {"/search", {{"vector", {1, 2, 3}}, {"k", 0}}},     // k out of range
        {"/search", {{"vector", {1, 2, 3}}, {"k", 1001}}},  //
        {"/search", {{"vector", {1, 2, 3}}, {"k", -1}}},    //
        {"/search", {{"vector", {1, 2, 3}}, {"k", 2.5}}},   //
        {"/search", {{"vector", {1, 2, 3}}, {"k", "3"}}},   //
        {"/search", {{"vector", {1, 2, 3}}, {"ef_search", 0}}},
        {"/search", {{"vector", {1, 2, 3}}, {"exact", "yes"}}},
        {"/vectors", json::object()},                          // no items
        {"/vectors", {{"items", json::array()}}},              // empty
        {"/vectors", {{"items", {{{"vector", {1, 2, 3}}}}}}},  // no id
        {"/vectors", {{"items", {{{"id", ""}, {"vector", {1, 2, 3}}}}}}},
        {"/vectors", {{"items", {{{"id", std::string(300, 'x')}, {"vector", {1, 2, 3}}}}}}},
        {"/vectors", {{"items", {{{"id", "a"}, {"vector", {1, 2}}}}}}},
        {"/vectors", {{"items", {{{"id", "a"}, {"vector", {1, 2, 3}}, {"metadata", "text"}}}}}},
        {"/index", {{"metric", "l2"}}},                        // no dim
        {"/index", {{"dim", 0}, {"metric", "l2"}}},            //
        {"/index", {{"dim", 3}, {"metric", "dot"}}},           // unknown metric
        {"/index", {{"dim", 3}, {"metric", "l2"}, {"M", 1}}},  //
    };
    for (const auto& [path, body] : bad) {
        const Reply r = post(path, body);
        EXPECT_EQ(r.status, 400) << path << " " << body;
        EXPECT_TRUE(r.body["error"].is_string()) << path << " " << body;
    }
    EXPECT_EQ(call("POST", "/search", "{not json").status, 400);
    EXPECT_EQ(call("POST", "/search", "[1, 2, 3]").status, 400);
    EXPECT_EQ(get("/stats").body["index"]["size"], 0);  // nothing was added
}

TEST_F(ServerTest, BadItemOrDuplicateAddsNothing) {
    create_index(2);
    ASSERT_EQ(post("/vectors", {{"items", {{{"id", "a"}, {"vector", {1, 2}}}}}}).status, 200);
    EXPECT_EQ(post("/vectors", {{"items", {{{"id", "a"}, {"vector", {3, 4}}}}}}).status, 409);
    EXPECT_EQ(
        post("/vectors",
             {{"items", {{{"id", "b"}, {"vector", {3, 4}}}, {{"id", "b"}, {"vector", {5, 6}}}}}})
            .status,
        409);
    EXPECT_EQ(post("/vectors",
                   {{"items", {{{"id", "c"}, {"vector", {3, 4}}}, {{"id", "d"}, {"vector", {5}}}}}})
                  .status,
              400);
    EXPECT_EQ(get("/stats").body["index"]["size"], 1);
}

TEST_F(ServerTest, NeedsAnIndexFirst) {
    EXPECT_EQ(post("/search", {{"vector", {1, 2, 3}}}).status, 409);
    EXPECT_EQ(post("/vectors", {{"items", {{{"id", "a"}, {"vector", {1}}}}}}).status, 409);
    EXPECT_EQ(post("/save", json::object()).status, 409);
    const Reply stats = get("/stats");
    EXPECT_EQ(stats.status, 200);
    EXPECT_TRUE(stats.body["index"].is_null());
    EXPECT_TRUE(stats.body["rag"].is_null());
    EXPECT_EQ(stats.body["embed_model"], "all-minilm");
    EXPECT_EQ(stats.body["llm_model"], "fake-llm");
}

TEST_F(ServerTest, ServesTheUi) {
    const auto ui = data_dir_ / "ui";
    std::filesystem::create_directories(ui);
    std::ofstream(ui / "index.html") << "<h1>Vektor UI</h1>";
    ServerConfig c = config();
    c.ui_dir = ui;
    start(c);
    httplib::Client client(server_->url());
    const auto page = client.Get("/");
    ASSERT_TRUE(page);
    EXPECT_EQ(page->status, 200);
    EXPECT_EQ(page->body, "<h1>Vektor UI</h1>");
    EXPECT_EQ(get("/health").body["status"], "ok");  // the API still answers next to the UI

    c.ui_dir = data_dir_ / "missing";
    EXPECT_THROW(start(c), std::invalid_argument);
}

TEST_F(ServerTest, RejectsHugeBodies) {
    ServerConfig c = config();
    c.max_body_bytes = 1000;
    start(c);
    EXPECT_EQ(call("POST", "/search", std::string(5000, ' ')).status, 413);
}

TEST_F(ServerTest, SaveAndReload) {
    create_index(8, "cosine");
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    json items = json::array();
    for (int i = 0; i < 50; ++i) {
        std::vector<float> v(8);
        for (float& x : v) {
            x = uniform(rng);
        }
        items.push_back({{"id", std::format("v{}", i)}, {"vector", v}, {"metadata", {{"i", i}}}});
    }
    ASSERT_EQ(post("/vectors", {{"items", items}}).status, 200);
    const json query = {{"vector", items[7]["vector"]}, {"k", 5}};
    const json before = post("/search", query).body["results"];

    const Reply saved = post("/save", json::object());
    ASSERT_EQ(saved.status, 200) << saved.body;
    EXPECT_TRUE(std::filesystem::exists(data_dir_ / "index.vkt"));
    EXPECT_GT(saved.body["file_bytes"].get<std::size_t>(), 50U * 8U * 4U);

    start(config());  // a new server process would do the same: load data_dir/index.vkt
    EXPECT_EQ(get("/stats").body["index"]["size"], 50);
    const json after = post("/search", query).body["results"];
    EXPECT_EQ(after, before);
    EXPECT_EQ(after[0]["id"], "v7");
    EXPECT_EQ(after[0]["metadata"]["i"], 7);
}

TEST_F(ServerTest, ConcurrentSearchesWhileAdding) {
    create_index(8);
    std::mt19937_64 rng(2);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    const auto batch = [&](int first, int count) {
        json items = json::array();
        for (int i = first; i < first + count; ++i) {
            std::vector<float> v(8);
            for (float& x : v) {
                x = uniform(rng);
            }
            items.push_back({{"id", std::to_string(i)}, {"vector", v}});
        }
        return json{{"items", items}};
    };
    ASSERT_EQ(post("/vectors", batch(0, 100)).status, 200);

    std::vector<json> batches;
    batches.reserve(20);
    for (int b = 0; b < 20; ++b) {
        batches.push_back(batch(100 + (b * 10), 10));
    }
    std::atomic<int> failures = 0;
    std::vector<std::thread> threads;
    threads.emplace_back([&] {
        httplib::Client client(server_->url());
        for (const json& b : batches) {
            const auto res = client.Post("/vectors", b.dump(), "application/json");
            failures += (!res || res->status != 200) ? 1 : 0;
        }
    });
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&] {
            httplib::Client client(server_->url());
            const std::string query = json{{"vector", {0, 0, 0, 0, 0, 0, 0, 0}}, {"k", 5}}.dump();
            for (int i = 0; i < 50; ++i) {
                const auto res = client.Post("/search", query, "application/json");
                failures +=
                    (!res || res->status != 200 || json::parse(res->body)["results"].size() != 5)
                        ? 1
                        : 0;
            }
        });
    }
    for (std::thread& t : threads) {
        t.join();
    }
    EXPECT_EQ(failures, 0);
    EXPECT_EQ(get("/stats").body["index"]["size"], 300);
}

json three_documents() {
    return {
        {"documents",
         {{{"id", "fruit"},
           {"title", "Fruit"},
           {"text",
            "apples and pears grow on trees in the orchard while bananas grow on "
            "large plants in warm countries"}},
          {{"id", "space"},
           {"title", "Space"},
           {"text",
            "the rocket left the launch pad and carried the probe into orbit "
            "around the red planet"}},
          {{"id", "music"},
           {"title", "Music"},
           {"text", "the violin and the cello play the melody while the drums keep the beat"}}}}};
}

TEST_F(ServerTest, RagIngestSearchAsk) {
    const Reply ingested = post("/rag/ingest", three_documents());
    ASSERT_EQ(ingested.status, 200) << ingested.body;
    EXPECT_EQ(ingested.body["documents"], 3);
    // 18 + 16 + 14 words in chunks of 10 with 2 overlapping: 2 + 2 + 2 chunks.
    EXPECT_EQ(ingested.body["chunks"], 6);
    EXPECT_TRUE(std::filesystem::exists(data_dir_ / "rag.vkt"));

    const Reply found = post("/rag/search", {{"query", "rocket probe orbit planet"}, {"k", 3}});
    ASSERT_EQ(found.status, 200) << found.body;
    ASSERT_EQ(found.body["results"].size(), 3U);
    EXPECT_EQ(found.body["results"][0]["doc_id"], "space");
    EXPECT_EQ(found.body["results"][0]["title"], "Space");
    EXPECT_TRUE(found.body["results"][0]["text"].is_string());
    EXPECT_GT(found.body["results"][0]["score"].get<double>(), 0.3);
    EXPECT_GE(found.body["embed_ms"].get<double>(), 0.0);

    const Reply asked = post("/rag/ask", {{"question", "what did the rocket carry?"}, {"k", 2}});
    ASSERT_EQ(asked.status, 200) << asked.body;
    EXPECT_EQ(asked.body["answer"], "fake answer from 2 sources [1]");
    ASSERT_EQ(asked.body["sources"].size(), 2U);
    EXPECT_EQ(asked.body["sources"][0]["n"], 1);
    EXPECT_EQ(asked.body["sources"][0]["doc_id"], "space");
    for (const char* key : {"embed_ms", "search_ms", "llm_ms"}) {
        EXPECT_TRUE(asked.body["timings"][key].is_number()) << key;
    }

    start(config());  // reloads data_dir/rag.vkt
    EXPECT_EQ(get("/stats").body["rag"]["size"], 6);
    EXPECT_EQ(post("/rag/search", {{"query", "violin"}}).body["results"][0]["doc_id"], "music");
}

TEST_F(ServerTest, RagRejectsDuplicatesAndBadDocuments) {
    ASSERT_EQ(post("/rag/ingest", three_documents()).status, 200);
    EXPECT_EQ(post("/rag/ingest", three_documents()).status, 409);
    EXPECT_EQ(post("/rag/ingest", {{"documents", {{{"id", "x"}}}}}).status, 400);  // no text
    EXPECT_EQ(post("/rag/ingest", {{"documents", {{{"id", "x"}, {"text", "   "}}}}}).status, 400);
    EXPECT_EQ(post("/rag/ingest", {{"documents", json::array()}}).status, 400);
    EXPECT_EQ(get("/stats").body["rag"]["size"], 6);
}

TEST_F(ServerTest, RagNeedsDocumentsFirst) {
    EXPECT_EQ(post("/rag/search", {{"query", "anything"}}).status, 409);
    EXPECT_EQ(post("/rag/ask", {{"question", "anything"}}).status, 409);
}

TEST_F(ServerTest, AskIsOffInLowSpaceMode) {
    ServerConfig c = config();
    c.llm_model.clear();
    start(c);
    ASSERT_EQ(post("/rag/ingest", three_documents()).status, 200);
    EXPECT_EQ(post("/rag/search", {{"query", "violin"}}).status, 200);
    EXPECT_TRUE(get("/stats").body["llm_model"].is_null());
    const Reply r = post("/rag/ask", {{"question", "anything"}});
    EXPECT_EQ(r.status, 503);
    EXPECT_NE(r.body["error"].get<std::string>().find("low-space"), std::string::npos);
}

TEST_F(ServerTest, OllamaDownGives502) {
    ServerConfig c = config();
    c.ollama_url = "http://127.0.0.1:1";  // nothing listens on port 1
    start(c);
    const Reply r = post("/rag/ingest", three_documents());
    EXPECT_EQ(r.status, 502);
    EXPECT_NE(r.body["error"].get<std::string>().find("cannot reach Ollama"), std::string::npos);
}

std::string words(int from, int to) {
    std::string s;
    for (int i = from; i < to; ++i) {
        s += (s.empty() ? "w" : " w") + std::to_string(i);
    }
    return s;
}

TEST(ChunkWords, OverlapsAndBoundaries) {
    const auto chunks = chunk_words(words(0, 25), 10, 2);
    ASSERT_EQ(chunks.size(), 3U);
    EXPECT_EQ(chunks[0], words(0, 10));
    EXPECT_EQ(chunks[1], words(8, 18));  // starts with the last 2 words of chunk 0
    EXPECT_EQ(chunks[2], words(16, 25));
    EXPECT_EQ(chunk_words(words(0, 10), 10, 2).size(), 1U);  // exactly one chunk's worth
    EXPECT_EQ(chunk_words(words(0, 11), 10, 2).size(), 2U);
    EXPECT_EQ(chunk_words(words(0, 3), 10, 2), std::vector<std::string>{words(0, 3)});
}

TEST(ChunkWords, WhitespaceAndEdgeCases) {
    EXPECT_TRUE(chunk_words("", 10, 2).empty());
    EXPECT_TRUE(chunk_words(" \n\t ", 10, 2).empty());
    EXPECT_EQ(chunk_words("  one\ntwo\t three  ", 10, 2),
              std::vector<std::string>{"one two three"});
    EXPECT_THROW((void)chunk_words("a b", 5, 5), std::invalid_argument);
    EXPECT_THROW((void)chunk_words("a b", 0, 0), std::invalid_argument);
}

TEST(BuildPrompt, NumbersTheSources) {
    const std::string prompt =
        build_prompt("Why?", {{.title = "A", .text = "first"}, {.title = "B", .text = "second"}});
    EXPECT_NE(prompt.find("[1] A\nfirst"), std::string::npos);
    EXPECT_NE(prompt.find("[2] B\nsecond"), std::string::npos);
    EXPECT_NE(prompt.find("I don't know"), std::string::npos);
    EXPECT_TRUE(prompt.ends_with("Question: Why?\nAnswer:"));
}

}  // namespace
}  // namespace vektor
