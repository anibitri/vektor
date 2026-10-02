// vektor-bench: converts datasets and measures search speed and recall.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <format>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bench/dataset.hpp"
#include "core/index.hpp"

namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* kUsage = R"(usage:
  vektor-bench convert --base FILE.fvecs --queries FILE.fvecs --metric l2|cosine --out FILE.vkd
                       [--n-base N] [--n-queries N] [--gt-k 100]
      Takes the first N vectors of each file, finds the true top gt-k neighbours
      of every query by brute force, and writes one .vkd file.
  vektor-bench run --data FILE.vkd [--k 10]
      Measures exact (brute-force) search on the dataset.
)";

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Command-line flags given as "--name value" pairs.
class Flags {
public:
    explicit Flags(std::span<char* const> args) {
        for (std::size_t i = 0; i < args.size(); i += 2) {
            const std::string name = args[i];
            if (!name.starts_with("--") || i + 1 >= args.size()) {
                throw std::invalid_argument("expected '--name value', got '" + name + "'");
            }
            values_[name.substr(2)] = args[i + 1];
        }
    }

    std::string get(const std::string& name) {
        const auto it = values_.find(name);
        if (it == values_.end()) {
            throw std::invalid_argument("missing --" + name);
        }
        std::string value = it->second;
        values_.erase(it);
        return value;
    }

    std::string get(const std::string& name, const std::string& fallback) {
        return values_.contains(name) ? get(name) : fallback;
    }

    std::uint64_t get_count(const std::string& name, std::uint64_t fallback) {
        if (!values_.contains(name)) {
            return fallback;
        }
        const std::string text = get(name);
        std::uint64_t value = 0;
        const auto [end, err] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (err != std::errc() || end != text.data() + text.size()) {
            throw std::invalid_argument("--" + name + " must be a whole number, got '" + text +
                                        "'");
        }
        return value;
    }

    // Call after reading every flag, to catch typos.
    void check_all_used() const {
        if (!values_.empty()) {
            throw std::invalid_argument("unknown flag --" + values_.begin()->first);
        }
    }

private:
    std::map<std::string, std::string> values_;
};

int convert(Flags& flags) {
    const std::string base_path = flags.get("base");
    const std::string queries_path = flags.get("queries");
    const std::string out_path = flags.get("out");
    const vektor::Metric metric = vektor::parse_metric(flags.get("metric"));
    const auto all = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t n_base = flags.get_count("n-base", all);
    const std::uint64_t n_queries = flags.get_count("n-queries", all);
    const std::uint64_t gt_k = flags.get_count("gt-k", 100);
    flags.check_all_used();
    if (gt_k == 0 || gt_k > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--gt-k is out of range");
    }

    const auto start = Clock::now();
    vektor::Vectors base = vektor::read_fvecs(base_path, n_base);
    vektor::Vectors queries = vektor::read_fvecs(queries_path, n_queries);
    const vektor::Dataset ds = vektor::make_dataset(std::move(base), std::move(queries), metric,
                                                    static_cast<std::uint32_t>(gt_k));
    vektor::save_vkd(ds, out_path);
    std::cout << std::format(
        "wrote {}: {} base vectors, {} queries, dim {}, {}, true top {} per query ({:.1f} s)\n",
        out_path, ds.base.rows(), ds.queries.rows(), ds.base.dim, vektor::to_string(metric),
        ds.gt_k, seconds_since(start));
    return 0;
}

int run(Flags& flags) {
    const std::string data_path = flags.get("data");
    const std::uint64_t k = flags.get_count("k", 10);
    flags.check_all_used();

    const vektor::Dataset ds = vektor::load_vkd(data_path);
    if (k == 0 || k > ds.gt_k) {
        throw std::invalid_argument(std::format("--k must be between 1 and {}", ds.gt_k));
    }

    vektor::Index index(ds.base.dim, ds.metric);
    index.reserve(ds.base.rows());
    const auto build_start = Clock::now();
    for (std::size_t i = 0; i < ds.base.rows(); ++i) {
        index.add(ds.base.row(i));
    }
    const double build_s = seconds_since(build_start);

    std::vector<double> latency_ms;
    latency_ms.reserve(ds.queries.rows());
    double recall_sum = 0.0;
    for (std::size_t q = 0; q < ds.queries.rows(); ++q) {
        const auto start = Clock::now();
        const std::vector<vektor::Result> found = index.search_exact(ds.queries.row(q), k);
        latency_ms.push_back(seconds_since(start) * 1000.0);
        recall_sum += vektor::recall_at_k(found, ds.truth(q), k);
    }

    const auto n = static_cast<double>(latency_ms.size());
    const double total_ms = std::accumulate(latency_ms.begin(), latency_ms.end(), 0.0);
    std::ranges::sort(latency_ms);
    const auto p99_index = static_cast<std::size_t>(std::ceil(0.99 * n)) - 1;
    std::cout << std::format(
        "{}: exact search, {} queries, recall@{} {:.4f}, {:.1f} queries/s, mean {:.3f} ms, "
        "p99 {:.3f} ms, build {:.3f} s, memory {:.1f} MB ({} bytes/vector)\n",
        data_path, ds.queries.rows(), k, recall_sum / n, n / (total_ms / 1000.0), total_ms / n,
        latency_ms[p99_index], build_s, static_cast<double>(index.memory_bytes()) / 1e6,
        index.memory_bytes() / index.size());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::span<char*> args(argv, static_cast<std::size_t>(argc));
        if (args.size() < 2) {
            std::cerr << kUsage;
            return 2;
        }
        const std::string command = args[1];
        Flags flags(args.subspan(2));
        if (command == "convert") {
            return convert(flags);
        }
        if (command == "run") {
            return run(flags);
        }
        std::cerr << "unknown command '" << command << "'\n" << kUsage;
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
