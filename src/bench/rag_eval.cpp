#include "bench/rag_eval.hpp"

#include <httplib.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace vektor {
namespace {

using nlohmann::json;

struct Question {
    std::string id;
    std::string chunk;  // ID of the chunk that answers it
    std::string question;
    std::vector<std::string> keywords;  // an answer is counted correct if it has any of these
};

std::vector<Question> load_questions(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    // Keep the parsed document alive: looping over a part of a temporary would dangle.
    const json doc = json::parse(in);
    std::vector<Question> out;
    for (const json& q : doc.at("questions")) {
        out.push_back({.id = q.at("id"),
                       .chunk = q.at("chunk"),
                       .question = q.at("question"),
                       .keywords = q.at("keywords").get<std::vector<std::string>>()});
    }
    return out;
}

json call(httplib::Client& client, const std::string& path, const json* body) {
    const auto res =
        body == nullptr ? client.Get(path) : client.Post(path, body->dump(), "application/json");
    if (!res) {
        throw std::runtime_error("cannot reach vektor-server: " + httplib::to_string(res.error()));
    }
    const json reply = json::parse(res->body, nullptr, /*allow_exceptions=*/false);
    if (res->status != 200) {
        throw std::runtime_error(std::format("{} answered {}: {}", path, res->status, res->body));
    }
    return reply;
}

// 1-based rank of the chunk in the results, or 0 if it is not there.
std::size_t rank_of(const json& results, const std::string& chunk) {
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (results[i].at("id") == chunk) {
            return i + 1;
        }
    }
    return 0;
}

std::string lower(std::string s) {
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

struct Scores {
    std::vector<std::vector<std::string>> ids;  // each question's top-10 chunk IDs
    double overlap = 0;  // share of exact search's top 10 also returned (HNSW only)
    double recall1 = 0;
    double recall5 = 0;
    double recall10 = 0;
    double mrr = 0;  // mean of 1 / rank (0 when not in the top 10)
    double search_ms = 0;
    double embed_ms = 0;
};

// Runs every question. With `reference` (the exact-search scores), also measures
// how much of exact search's top 10 this search returns.
Scores evaluate(httplib::Client& client, const std::vector<Question>& questions, bool exact,
                std::size_t ef, const Scores* reference) {
    Scores s;
    for (std::size_t i = 0; i < questions.size(); ++i) {
        const Question& q = questions[i];
        const json body = {{"query", q.question}, {"k", 10}, {"ef_search", ef}, {"exact", exact}};
        const json reply = call(client, "/rag/search", &body);
        std::vector<std::string> ids;
        for (const json& r : reply.at("results")) {
            ids.push_back(r.at("id"));
        }
        if (reference != nullptr) {
            const auto& want = reference->ids[i];
            const auto found = std::ranges::count_if(ids, [&](const std::string& id) {
                return std::ranges::find(want, id) != want.end();
            });
            s.overlap += static_cast<double>(found) / static_cast<double>(want.size());
        }
        s.ids.push_back(std::move(ids));
        const std::size_t rank = rank_of(reply.at("results"), q.chunk);
        s.recall1 += rank == 1 ? 1 : 0;
        s.recall5 += rank >= 1 && rank <= 5 ? 1 : 0;
        s.recall10 += rank >= 1 ? 1 : 0;
        s.mrr += rank >= 1 ? 1.0 / static_cast<double>(rank) : 0.0;
        s.search_ms += reply.at("took_ms").get<double>();
        s.embed_ms += reply.at("embed_ms").get<double>();
    }
    const auto n = static_cast<double>(questions.size());
    return {.ids = std::move(s.ids),
            .overlap = s.overlap / n,
            .recall1 = s.recall1 / n,
            .recall5 = s.recall5 / n,
            .recall10 = s.recall10 / n,
            .mrr = s.mrr / n,
            .search_ms = s.search_ms / n,
            .embed_ms = s.embed_ms / n};
}

double median(std::vector<double> v) {
    std::ranges::sort(v);
    const std::size_t m = v.size() / 2;
    return v.size() % 2 == 1 ? v[m] : (v[m - 1] + v[m]) / 2.0;
}

// Asks every question and saves the answers; prints a short summary.
void ask_all(httplib::Client& client, const std::vector<Question>& questions,
             const std::string& path, const std::string& about) {
    json records = json::array();
    std::vector<double> embed;
    std::vector<double> search;
    std::vector<double> llm;
    int correct = 0;
    int cited = 0;
    int dont_know = 0;
    int label_in_sources = 0;
    for (const Question& q : questions) {
        const json body = {{"question", q.question}, {"k", 5}};
        const json reply = call(client, "/rag/ask", &body);
        const std::string answer = reply.at("answer");
        const std::string text = lower(answer);
        const bool is_correct = std::ranges::any_of(q.keywords, [&](const std::string& k) {
            return text.find(lower(k)) != std::string::npos;
        });
        bool has_citation = false;
        for (std::size_t i = 0; i + 2 < answer.size(); ++i) {
            has_citation =
                has_citation ||
                (answer[i] == '[' && std::isdigit(static_cast<unsigned char>(answer[i + 1])) != 0);
        }
        json source_ids = json::array();
        for (const json& s : reply.at("sources")) {
            source_ids.push_back(s.at("id"));
        }
        const bool found = std::ranges::find(source_ids, json(q.chunk)) != source_ids.end();
        correct += is_correct ? 1 : 0;
        cited += has_citation ? 1 : 0;
        dont_know += text.find("don't know") != std::string::npos ? 1 : 0;
        label_in_sources += found ? 1 : 0;
        const json& t = reply.at("timings");
        embed.push_back(t.at("embed_ms"));
        search.push_back(t.at("search_ms"));
        llm.push_back(t.at("llm_ms"));
        records.push_back({{"id", q.id},
                           {"question", q.question},
                           {"answer", answer},
                           {"keyword_match", is_correct},
                           {"cites_a_source", has_citation},
                           {"answering_chunk_in_sources", found},
                           {"sources", source_ids},
                           {"timings", t}});
        std::cout << "." << std::flush;
    }
    const auto n = static_cast<int>(questions.size());
    const json summary = {
        {"questions", n},
        {"keyword_match", correct},
        {"cites_a_source", cited},
        {"says_dont_know", dont_know},
        {"answering_chunk_in_sources", label_in_sources},
        {"median_ms", {{"embed", median(embed)}, {"search", median(search)}, {"llm", median(llm)}}},
        {"mean_ms",
         {{"embed", std::accumulate(embed.begin(), embed.end(), 0.0) / n},
          {"search", std::accumulate(search.begin(), search.end(), 0.0) / n},
          {"llm", std::accumulate(llm.begin(), llm.end(), 0.0) / n}}}};
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    out << json{{"about", about}, {"summary", summary}, {"answers", records}}.dump(2) << '\n';
    std::cout << std::format(
        "\nasked {} questions: answer contains the expected keyword {}, cites a source {}, says "
        "\"I don't know\" {}, answering chunk among the 5 sources {}\nmedian time: embed {:.1f} "
        "ms, search {:.3f} ms, LLM {:.0f} ms\n",
        n, correct, cited, dont_know, label_in_sources, median(embed), median(search), median(llm));
}

}  // namespace

int rag_eval(Flags& flags, const std::string& about) {
    const std::string questions_path = flags.get("questions");
    const std::string server = flags.get("server", "http://127.0.0.1:8080");
    const std::vector<std::uint64_t> ef_values = flags.get_counts("ef-search", "10,20,40,80,160");
    const std::string out_path = flags.get("out", "");
    const std::string ask_out = flags.get("ask-out", "");
    const std::string commit = flags.get("commit", "unknown");
    flags.check_all_used();

    const std::vector<Question> questions = load_questions(questions_path);
    httplib::Client client(server);
    client.set_read_timeout(std::chrono::minutes(5));
    const json stats = call(client, "/stats", nullptr);
    if (stats.at("rag").is_null()) {
        throw std::runtime_error("the server has no RAG index: run `make rag-ingest` first");
    }
    const std::string header = std::format(
        "{}# commit: {}\n# rag index: {} chunks, embedding model {}\n# questions: {} from {}\n",
        about, commit, stats["rag"]["size"].get<std::size_t>(),
        stats["embed_model"].get<std::string>(), questions.size(), questions_path);

    std::ofstream csv;
    if (!out_path.empty()) {
        csv.open(out_path);
        if (!csv) {
            throw std::runtime_error("cannot write " + out_path);
        }
        csv << header
            << "search,ef_search,recall_at_1,recall_at_5,recall_at_10,mrr,overlap_with_exact,"
               "search_ms,embed_ms\n";
    }
    std::cout << std::format("{:<8} {:>9} {:>9} {:>9} {:>10} {:>7} {:>9} {:>10} {:>9}\n", "search",
                             "ef_search", "recall@1", "recall@5", "recall@10", "MRR", "vs exact",
                             "search ms", "embed ms");
    const auto report = [&](const std::string& search, const std::string& ef, const Scores& s,
                            bool has_overlap) {
        const std::string overlap = has_overlap ? std::format("{:.4f}", s.overlap) : "";
        if (csv.is_open()) {
            csv << std::format("{},{},{:.4f},{:.4f},{:.4f},{:.4f},{},{:.4f},{:.2f}\n", search, ef,
                               s.recall1, s.recall5, s.recall10, s.mrr, overlap, s.search_ms,
                               s.embed_ms)
                << std::flush;
        }
        std::cout << std::format(
            "{:<8} {:>9} {:>9.3f} {:>9.3f} {:>10.3f} {:>7.3f} {:>9} {:>10.4f} {:>9.1f}\n", search,
            ef, s.recall1, s.recall5, s.recall10, s.mrr, overlap, s.search_ms, s.embed_ms);
    };
    const Scores exact = evaluate(client, questions, true, 100, nullptr);  // ef_search unused
    report("exact", "", exact, false);
    for (const std::uint64_t ef : ef_values) {
        report("hnsw", std::to_string(ef), evaluate(client, questions, false, ef, &exact), true);
    }
    if (!ask_out.empty()) {
        ask_all(client, questions, ask_out, header);
    }
    return 0;
}

}  // namespace vektor
