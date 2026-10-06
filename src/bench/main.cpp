// vektor-bench: converts datasets and measures search speed and recall.

#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "bench/dataset.hpp"
#include "bench/rag_eval.hpp"
#include "core/flags.hpp"
#include "core/index.hpp"

#ifdef __APPLE__
#include <pthread/qos.h>
#include <sys/sysctl.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr const char* kUsage = R"(usage:
  vektor-bench convert --base FILE --metric l2|cosine --out FILE.vkd [--queries FILE]
                       [--format fvecs|word2vec|idx] [--n-base N] [--n-queries N]
                       [--sample-seed S] [--gt-k 100]
      Takes the first N vectors of each file (or, with --sample-seed S > 0, a
      random sample of each file), finds the true top gt-k neighbours of every
      query by brute force, and writes one .vkd file. Without --queries, the
      queries are other rows of the base file (then --n-base and --n-queries
      are required).

  vektor-bench twonn --data NAME[,NAME...] [--sample 2000] [--k 20] [--scope all|sample]
                     [--out FILE.csv] [--commit ID]
      Estimates intrinsic dimension with TwoNN and the Levina-Bickel MLE (k
      neighbours), from a random sample of points. --scope all: neighbours
      come from all the vectors; sample: only from the sample (coarser scale).

  vektor-bench rag-eval --questions FILE.json [--server http://127.0.0.1:8080]
                        [--ef-search 10,20,40,80,160] [--out FILE.csv] [--ask-out FILE.json]
                        [--commit ID]
      Measures RAG retrieval through a running vektor-server: the rank of the
      chunk that answers each question, with exact search and with HNSW at each
      ef_search (recall@1, @5, @10 and MRR). --ask-out: also asks each question
      with /rag/ask and saves the answers and timings.

  NAME is a .vkd file, or synthetic-R: 100,000 + 1,000 points generated in
  memory with intrinsic dimension R in 128 dimensions.

  vektor-bench run --data NAME [--out FILE.csv] [--runs 1] [--k 10]
                   [--M 16] [--ef-construction 200] [--ef-search 10,20,40,80,160,320]
                   [--select heuristic] [--visited tags] [--commit ID] [--save FILE.vkt]
      Measures exact search once per run, then builds an HNSW index for every
      M and --select value (M_max0 = 2 * M) and measures every --visited and
      --ef-search value. Lists are comma-separated. Run r uses seed 42 + r.
      Each measurement: one warm-up pass over all queries, then 5 timed passes;
      the pass with the median time is reported.
      --save: also save each index to FILE.vkt, load it back, check the loaded
      index gives the same results, and print file size and save/load times.
      --select: heuristic or simple. --visited: tags or hash.
)";

#ifdef __clang__
constexpr const char* kCompiler = "clang " __clang_version__;
#elif defined(__GNUC__)
constexpr const char* kCompiler = "gcc " __VERSION__;
#else
constexpr const char* kCompiler = "unknown";
#endif

#ifndef VEKTOR_FLAGS
#define VEKTOR_FLAGS "unknown"
#endif

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

using vektor::Flags;

vektor::Vectors read_vectors(const std::string& path, const std::string& format,
                             std::uint64_t max_rows) {
    if (format == "fvecs") {
        return vektor::read_fvecs(path, max_rows);
    }
    if (format == "word2vec") {
        return vektor::read_word2vec(path, max_rows);
    }
    if (format == "idx") {
        return vektor::read_idx(path, max_rows);
    }
    throw std::invalid_argument("--format must be fvecs, word2vec or idx");
}

int convert(Flags& flags) {
    const std::string base_path = flags.get("base");
    const std::string queries_path = flags.get("queries", "");
    const std::string out_path = flags.get("out");
    const std::string format = flags.get("format", "fvecs");
    const vektor::Metric metric = vektor::parse_metric(flags.get("metric"));
    const auto all = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t n_base = flags.get_count("n-base", all);
    const std::uint64_t n_queries = flags.get_count("n-queries", all);
    const std::uint64_t sample_seed = flags.get_count("sample-seed", 0);  // 0: no sampling
    const std::uint64_t gt_k = flags.get_count("gt-k", 100);
    flags.check_all_used();
    if (gt_k == 0 || gt_k > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("--gt-k is out of range");
    }
    const bool held_out = queries_path.empty();
    if (held_out && (n_base == all || n_queries == all)) {
        throw std::invalid_argument("without --queries, give --n-base and --n-queries");
    }

    const auto start = Clock::now();
    // With sampling, read every row first; otherwise only the rows needed.
    std::uint64_t to_read = held_out ? n_base + n_queries : n_base;
    if (sample_seed != 0) {
        to_read = all;
    }
    vektor::Vectors base = read_vectors(base_path, format, to_read);
    if (sample_seed != 0) {
        vektor::shuffle_rows(base, sample_seed);
    }
    vektor::Vectors queries;
    if (held_out) {
        queries = vektor::slice_rows(base, n_base, n_queries);
    } else if (sample_seed != 0) {  // a random sample of the query file too
        queries = read_vectors(queries_path, format, all);
        vektor::shuffle_rows(queries, sample_seed + 1);
        queries =
            vektor::slice_rows(queries, 0, std::min<std::uint64_t>(n_queries, queries.rows()));
    } else {
        queries = read_vectors(queries_path, format, n_queries);
    }
    base = vektor::slice_rows(base, 0, std::min<std::uint64_t>(n_base, base.rows()));
    const vektor::Dataset ds = vektor::make_dataset(std::move(base), std::move(queries), metric,
                                                    static_cast<std::uint32_t>(gt_k));
    vektor::save_vkd(ds, out_path);
    std::cout << std::format(
        "wrote {}: {} base vectors, {} queries, dim {}, {}, true top {} per query ({:.1f} s)\n",
        out_path, ds.base.rows(), ds.queries.rows(), ds.base.dim, vektor::to_string(metric),
        ds.gt_k, seconds_since(start));
    return 0;
}

// The machine the benchmark runs on, written at the top of every results file.
std::string machine_info() {
    std::string cpu = "unknown";
    std::string cores = std::to_string(std::thread::hardware_concurrency());
    std::string os;
    utsname u{};
    if (uname(&u) == 0) {
        os = std::format("{} {} {}", u.sysname, u.release, u.machine);
    }
#ifdef __APPLE__
    std::array<char, 256> buf{};
    std::size_t len = buf.size();
    if (sysctlbyname("machdep.cpu.brand_string", buf.data(), &len, nullptr, 0) == 0) {
        cpu = buf.data();
    }
    int performance_cores = 0;
    len = sizeof performance_cores;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &performance_cores, &len, nullptr, 0) == 0) {
        cores += std::format(" ({} performance)", performance_cores);
    }
    len = buf.size();
    if (sysctlbyname("kern.osproductversion", buf.data(), &len, nullptr, 0) == 0) {
        os += std::format(" (macOS {})", buf.data());
    }
#elif defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    for (std::string line; std::getline(cpuinfo, line);) {
        if (line.starts_with("model name")) {
            cpu = line.substr(line.find(':') + 2);
            break;
        }
    }
#endif
    const double ram_gb = static_cast<double>(sysconf(_SC_PHYS_PAGES)) *
                          static_cast<double>(sysconf(_SC_PAGESIZE)) / (1024.0 * 1024 * 1024);
    return std::format(
        "# cpu: {}\n# cores: {}\n# ram: {:.1f} GB\n# os: {}\n# compiler: {}\n# flags: {}\n", cpu,
        cores, ram_gb, os, kCompiler, VEKTOR_FLAGS);
}

std::string utc_now() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&now, &tm);
    std::array<char, 32> buf{};
    const std::size_t n = std::strftime(buf.data(), buf.size(), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return {buf.data(), n};
}

struct Measurement {
    double recall = 0;
    double qps = 0;
    double mean_ms = 0;
    double p99_ms = 0;
};

constexpr int kTimedPasses = 5;

// Runs every query once untimed (to warm up caches and check recall), then
// kTimedPasses timed passes, and reports the pass with the median total time,
// so one pass slowed down by other work on the machine does not skew the result.
// Single thread.
Measurement measure(
    const vektor::Dataset& ds, std::size_t k,
    const std::function<std::vector<vektor::Result>(std::span<const float>)>& search) {
    const std::size_t n_queries = ds.queries.rows();
    double recall_sum = 0.0;
    for (std::size_t q = 0; q < n_queries; ++q) {
        recall_sum += vektor::recall_at_k(search(ds.queries.row(q)), ds.truth(q), k);
    }
    std::vector<std::vector<double>> passes(kTimedPasses);  // latency of each query, in ms
    for (std::vector<double>& latency_ms : passes) {
        latency_ms.reserve(n_queries);
        for (std::size_t q = 0; q < n_queries; ++q) {
            const auto start = Clock::now();
            (void)search(ds.queries.row(q));
            latency_ms.push_back(seconds_since(start) * 1000.0);
        }
    }
    const auto total = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0);
    };
    std::ranges::sort(passes, {}, total);
    std::vector<double>& latency_ms = passes[kTimedPasses / 2];

    const auto n = static_cast<double>(n_queries);
    const double total_ms = total(latency_ms);
    std::ranges::sort(latency_ms);
    const auto p99_index = static_cast<std::size_t>(std::ceil(0.99 * n)) - 1;
    return {.recall = recall_sum / n,
            .qps = n / (total_ms / 1000.0),
            .mean_ms = total_ms / n,
            .p99_ms = latency_ms[p99_index]};
}

// Settings for the `run` command.
struct RunOptions {
    std::string data_path;
    std::string out_path;
    std::uint64_t runs = 1;
    std::uint64_t k = 10;
    std::vector<std::uint64_t> m_values;
    std::uint64_t ef_construction = 200;
    std::vector<std::uint64_t> ef_values;
    std::vector<std::string> selects;
    std::vector<std::string> visiteds;
    std::string commit;
    std::string save_path;
};

RunOptions parse_run_options(Flags& flags) {
    RunOptions opt;
    opt.data_path = flags.get("data");
    opt.out_path = flags.get("out", "");
    opt.runs = flags.get_count("runs", 1);
    opt.k = flags.get_count("k", 10);
    opt.m_values = flags.get_counts("M", "16");
    opt.ef_construction = flags.get_count("ef-construction", 200);
    opt.ef_values = flags.get_counts("ef-search", "10,20,40,80,160,320");
    opt.selects = flags.get_list("select", "heuristic");
    opt.visiteds = flags.get_list("visited", "tags");
    opt.commit = flags.get("commit", "unknown");
    opt.save_path = flags.get("save", "");
    flags.check_all_used();
    for (const std::string& s : opt.selects) {
        if (s != "heuristic" && s != "simple") {
            throw std::invalid_argument("--select must be heuristic or simple, got '" + s + "'");
        }
    }
    for (const std::string& v : opt.visiteds) {
        if (v != "tags" && v != "hash") {
            throw std::invalid_argument("--visited must be tags or hash, got '" + v + "'");
        }
    }
    return opt;
}

// Writes each result as a CSV row (if a file was given) and a short line on screen.
class Report {
public:
    Report(const std::string& path, const std::string& header, std::size_t k, std::size_t n_base)
        : k_(k), n_base_(n_base) {
        if (path.empty()) {
            return;
        }
        csv_.open(path);
        if (!csv_) {
            throw std::runtime_error("cannot write " + path);
        }
        csv_ << header
             << "algo,select,visited,M,M_max0,ef_construction,ef_search,run,seed,k,recall,qps,"
                "mean_ms,p99_ms,build_s,memory_bytes,bytes_per_vector\n";
    }

    void add(const std::string& config, const std::string& build_s, std::size_t memory,
             const Measurement& m) {
        if (csv_.is_open()) {
            csv_ << std::format("{},{},{:.4f},{:.1f},{:.4f},{:.4f},{},{},{}\n", config, k_,
                                m.recall, m.qps, m.mean_ms, m.p99_ms, build_s, memory,
                                memory / n_base_)
                 << std::flush;
        }
        std::cout << std::format("{:<48} recall@{} {:.4f}  {:>9.1f} q/s  mean {:.3f} ms\n", config,
                                 k_, m.recall, m.qps, m.mean_ms);
    }

private:
    std::ofstream csv_;
    std::size_t k_;
    std::size_t n_base_;
};

// Saves the index, loads it back, and checks the copy finds the same results.
void check_save_load(const vektor::Dataset& ds, const vektor::Index& index, const std::string& path,
                     std::size_t k) {
    auto start = Clock::now();
    index.save(path);
    const double save_s = seconds_since(start);
    start = Clock::now();
    const vektor::Index loaded = vektor::Index::load(path);
    const double load_s = seconds_since(start);
    for (std::size_t q = 0; q < ds.queries.rows(); ++q) {
        const auto a = index.search(ds.queries.row(q), k, 100);
        const auto b = loaded.search(ds.queries.row(q), k, 100);
        if (!std::ranges::equal(a, b,
                                [](const auto& x, const auto& y) { return x.row == y.row; })) {
            throw std::runtime_error("the loaded index gives different results");
        }
    }
    std::cout << std::format(
        "saved {}: file {:.1f} MB, memory {:.1f} MB, save {:.2f} s, load {:.2f} s, same results "
        "for all {} queries\n",
        path, static_cast<double>(std::filesystem::file_size(path)) / 1e6,
        static_cast<double>(index.memory_bytes()) / 1e6, save_s, load_s, ds.queries.rows());
}

// Builds one HNSW index and measures it for every --visited and --ef-search
// value. Also measures exact search if with_exact is set.
void bench_index(const vektor::Dataset& ds, const RunOptions& opt, std::uint64_t run,
                 std::uint64_t m, const std::string& select, bool with_exact, Report& report) {
    const std::uint64_t seed = 42 + run;
    const vektor::HnswParams params{.M = m,
                                    .M_max0 = 2 * m,
                                    .ef_construction = opt.ef_construction,
                                    .seed = seed,
                                    .heuristic = select == "heuristic"};
    const std::size_t n_base = ds.base.rows();
    vektor::Index index(ds.base.dim, ds.metric, params);
    index.reserve(n_base);
    const auto build_start = Clock::now();
    for (std::size_t i = 0; i < n_base; ++i) {
        index.add(ds.base.row(i));
    }
    const double build_s = seconds_since(build_start);
    std::cout << std::format("run {}: built M={} {} in {:.1f} s\n", run, m, select, build_s);

    if (!opt.save_path.empty()) {
        check_save_load(ds, index, opt.save_path, opt.k);
    }
    const std::size_t k = opt.k;
    if (with_exact) {
        const Measurement exact =
            measure(ds, k, [&](std::span<const float> q) { return index.search_exact(q, k); });
        report.add(std::format("exact,,,,,,,{},{}", run, seed), "",
                   n_base * ds.base.dim * sizeof(float), exact);
    }
    for (const std::string& visited_name : opt.visiteds) {
        const auto visited =
            visited_name == "tags" ? vektor::Visited::Tags : vektor::Visited::HashSet;
        for (const std::uint64_t ef : opt.ef_values) {
            const Measurement hnsw = measure(
                ds, k, [&](std::span<const float> q) { return index.search(q, k, ef, visited); });
            report.add(std::format("hnsw,{},{},{},{},{},{},{},{}", select, visited_name, m,
                                   params.M_max0, opt.ef_construction, ef, run, seed),
                       std::format("{:.3f}", build_s), index.memory_bytes(), hnsw);
        }
    }
}

int run(Flags& flags, const std::string& command_line) {
    const RunOptions opt = parse_run_options(flags);
    const vektor::Dataset ds = vektor::load_dataset(opt.data_path);
    if (opt.k == 0 || opt.k > ds.gt_k) {
        throw std::invalid_argument(std::format("--k must be between 1 and {}", ds.gt_k));
    }
#ifdef __APPLE__
    // Apple silicon has fast and slow (efficiency) cores. The highest priority
    // asks macOS to keep this thread on a fast one.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    const std::string header =
        std::format("# {}\n# date: {}\n# commit: {}\n", command_line, utc_now(), opt.commit) +
        std::format("# dataset: {} base vectors, {} queries, dim {}, {}\n", ds.base.rows(),
                    ds.queries.rows(), ds.base.dim, vektor::to_string(ds.metric)) +
        machine_info();
    Report report(opt.out_path, header, opt.k, ds.base.rows());
    for (std::uint64_t r = 0; r < opt.runs; ++r) {
        bool with_exact = true;  // exact search does not depend on M, so once per run
        for (const std::uint64_t m : opt.m_values) {
            for (const std::string& select : opt.selects) {
                bench_index(ds, opt, r, m, select, with_exact, report);
                with_exact = false;
            }
        }
    }
    return 0;
}

// Estimates the intrinsic dimension of each dataset (TwoNN and Levina-Bickel MLE).
int twonn(Flags& flags, const std::string& command_line) {
    const std::vector<std::string> names = flags.get_list("data", "");
    const std::uint64_t sample = flags.get_count("sample", 2000);
    const std::uint64_t k = flags.get_count("k", 20);
    const std::string scope = flags.get("scope", "all");
    const std::string out_path = flags.get("out", "");
    const std::string commit = flags.get("commit", "unknown");
    flags.check_all_used();
    if (names.empty() || names.front().empty()) {
        throw std::invalid_argument("missing --data");
    }
    if (scope != "all" && scope != "sample") {
        throw std::invalid_argument("--scope must be all or sample");
    }
    std::ofstream csv;
    if (!out_path.empty()) {
        csv.open(out_path);
        if (!csv) {
            throw std::runtime_error("cannot write " + out_path);
        }
        csv << std::format("# {}\n# date: {}\n# commit: {}\n", command_line, utc_now(), commit)
            << machine_info()
            << "dataset,dim,n_base,metric,scope,sample,points_used,twonn,mle_k,mle\n";
    }
    for (const std::string& name : names) {
        const auto start = Clock::now();
        const vektor::Dataset ds = vektor::load_dataset(name);
        const vektor::IntrinsicDimension est = vektor::estimate_intrinsic_dimension(
            ds.base, ds.metric, sample, k, 42, scope == "sample");
        const std::string label = std::filesystem::path(name).stem().string();
        if (csv.is_open()) {
            csv << std::format("{},{},{},{},{},{},{},{:.3f},{},{:.3f}\n", label, ds.base.dim,
                               ds.base.rows(), vektor::to_string(ds.metric), scope, sample,
                               est.points, est.twonn, k, est.mle)
                << std::flush;
        }
        std::cout << std::format(
            "{}: dim {}, intrinsic dimension: TwoNN {:.2f}, MLE (k={}) {:.2f} ({} points, "
            "{:.1f} s)\n",
            label, ds.base.dim, est.twonn, k, est.mle, est.points, seconds_since(start));
    }
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
        std::string command_line = "vektor-bench";
        for (const char* arg : args.subspan(1)) {
            command_line += std::string(" ") + arg;
        }
        if (command == "run") {
            return run(flags, command_line);
        }
        if (command == "twonn") {
            return twonn(flags, command_line);
        }
        if (command == "rag-eval") {
            return vektor::rag_eval(
                flags, std::format("# {}\n# date: {}\n", command_line, utc_now()) + machine_info());
        }
        std::cerr << "unknown command '" << command << "'\n" << kUsage;
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
