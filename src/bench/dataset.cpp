#include "bench/dataset.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <numbers>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/binary.hpp"

namespace vektor {

namespace {

constexpr std::array<char, 4> kMagic = {'V', 'K', 'D', '1'};
constexpr std::size_t kHeaderBytes = kMagic.size() + (5 * sizeof(std::uint32_t));
constexpr std::uint32_t kMaxDim = 1U << 16U;  // sanity limits, far above real datasets
constexpr std::uint32_t kMaxGtK = 1U << 16U;

std::runtime_error file_error(const std::filesystem::path& path, const std::string& what) {
    return std::runtime_error(path.string() + ": " + what);
}

void write_u32(std::ofstream& out, std::uint32_t value) {
    write_raw(out, std::span<const std::uint32_t>(&value, 1));
}

std::uint32_t read_u32(std::ifstream& in) {
    std::uint32_t value = 0;
    read_raw(in, std::span<std::uint32_t>(&value, 1));
    return value;
}

// Random numbers made from the raw generator output only, so every standard
// library gives the same values (the std:: distributions do not).
double uniform_open01(std::mt19937_64& rng) {
    return (static_cast<double>(rng() >> 11U) + 0.5) * 0x1.0p-53;  // (0, 1)
}

double gaussian(std::mt19937_64& rng) {  // Box-Muller
    const double u = uniform_open01(rng);
    const double v = uniform_open01(rng);
    return std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * std::numbers::pi * v);
}

// Fisher-Yates shuffle of 0..n-1.
std::vector<std::uint32_t> shuffled_indices(std::size_t n, std::uint64_t seed) {
    std::vector<std::uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0U);
    std::mt19937_64 rng(seed);
    for (std::size_t i = n; i > 1; --i) {
        std::swap(order[i - 1], order[rng() % i]);
    }
    return order;
}

// A checked copy of all rows, scaled to length 1 for cosine, ready for brute_force().
std::vector<float> prepared_rows(const Vectors& v, Metric metric) {
    std::vector<float> rows;
    rows.reserve(v.data.size());
    for (std::size_t i = 0; i < v.rows(); ++i) {
        const std::vector<float> row = prepare_vector(v.row(i), v.dim, metric);
        rows.insert(rows.end(), row.begin(), row.end());
    }
    return rows;
}

std::uint32_t big_endian_u32(std::span<const unsigned char> b) {
    return (std::uint32_t{b[0]} << 24U) | (std::uint32_t{b[1]} << 16U) |
           (std::uint32_t{b[2]} << 8U) | std::uint32_t{b[3]};
}

}  // namespace

Vectors read_fvecs(const std::filesystem::path& path, std::size_t max_rows) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw file_error(path, "cannot open");
    }
    Vectors out;
    for (std::size_t i = 0; i < max_rows; ++i) {
        std::int32_t len = 0;
        read_raw(in, std::span<std::int32_t>(&len, 1));
        if (in.gcount() == 0) {
            break;  // end of file
        }
        if (!in || len <= 0 || std::cmp_greater(len, kMaxDim)) {
            throw file_error(path, "bad vector length at row " + std::to_string(i));
        }
        const auto dim = static_cast<std::uint32_t>(len);
        if (out.dim == 0) {
            out.dim = dim;
            const std::size_t file_rows = std::filesystem::file_size(path) / (4 + (4 * dim));
            out.data.reserve(std::min(max_rows, file_rows) * dim);
        } else if (dim != out.dim) {
            throw file_error(path, "row " + std::to_string(i) + " has a different length");
        }
        const std::size_t old_size = out.data.size();
        out.data.resize(old_size + dim);
        read_raw(in, std::span<float>(out.data).subspan(old_size));
        if (!in) {
            throw file_error(path, "file ends in the middle of a vector");
        }
    }
    if (out.data.empty()) {
        throw file_error(path, "no vectors found");
    }
    return out;
}

Vectors read_word2vec(const std::filesystem::path& path, std::size_t max_rows) {
    std::ifstream in(path);
    if (!in) {
        throw file_error(path, "cannot open");
    }
    std::string line;
    std::size_t total = 0;
    std::uint32_t dim = 0;
    if (!std::getline(in, line) || !(std::istringstream(line) >> total >> dim) || dim == 0 ||
        dim > kMaxDim) {
        throw file_error(path, "expected a header line '<rows> <dim>'");
    }
    Vectors out{.dim = dim, .data = {}};
    out.data.reserve(std::min(total, max_rows) * dim);
    for (std::size_t row = 0; row < max_rows && std::getline(in, line); ++row) {
        const char* p = line.data();
        const char* const end = p + line.size();
        p = std::find(p, end, ' ');  // skip the word
        for (std::uint32_t i = 0; i < dim; ++i) {
            while (p < end && *p == ' ') {
                ++p;
            }
            float value = 0;
            const auto [next, err] = std::from_chars(p, end, value);
            if (err != std::errc()) {
                throw file_error(path, "bad number on line " + std::to_string(row + 2));
            }
            out.data.push_back(value);
            p = next;
        }
    }
    if (out.data.empty()) {
        throw file_error(path, "no vectors found");
    }
    return out;
}

Vectors read_idx(const std::filesystem::path& path, std::size_t max_rows) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw file_error(path, "cannot open");
    }
    std::array<unsigned char, 16> header{};
    read_raw(in, std::span<unsigned char>(header));
    const std::span<const unsigned char> h(header);
    if (!in || big_endian_u32(h.first(4)) != 0x00000803U) {
        throw file_error(path, "not an IDX file of 8-bit images");
    }
    const std::uint32_t n = big_endian_u32(h.subspan(4, 4));
    const std::uint64_t dim =
        std::uint64_t{big_endian_u32(h.subspan(8, 4))} * big_endian_u32(h.subspan(12, 4));
    if (dim == 0 || dim > kMaxDim) {
        throw file_error(path, "bad image size");
    }
    const std::size_t count = std::min<std::size_t>(n, max_rows);
    std::vector<unsigned char> pixels(count * dim);
    read_raw(in, std::span<unsigned char>(pixels));
    if (!in || count == 0) {
        throw file_error(path, "file is cut short");
    }
    return {.dim = static_cast<std::uint32_t>(dim), .data = {pixels.begin(), pixels.end()}};
}

void shuffle_rows(Vectors& v, std::uint64_t seed) {
    const std::vector<std::uint32_t> order = shuffled_indices(v.rows(), seed);
    std::vector<float> data;
    data.reserve(v.data.size());
    for (const std::uint32_t row : order) {
        const auto r = v.row(row);
        data.insert(data.end(), r.begin(), r.end());
    }
    v.data = std::move(data);
}

Vectors slice_rows(const Vectors& v, std::size_t first, std::size_t n) {
    if (first + n > v.rows()) {
        throw std::invalid_argument("not enough vectors: asked for " + std::to_string(first + n) +
                                    ", have " + std::to_string(v.rows()));
    }
    const auto begin = v.data.begin() + static_cast<std::ptrdiff_t>(first * v.dim);
    return {.dim = v.dim, .data = {begin, begin + static_cast<std::ptrdiff_t>(n * v.dim)}};
}

Dataset make_dataset(Vectors base, Vectors queries, Metric metric, std::uint32_t gt_k) {
    if (base.rows() == 0 || queries.rows() == 0) {
        throw std::invalid_argument("need at least one base vector and one query");
    }
    if (base.dim != queries.dim) {
        throw std::invalid_argument("base and query vectors have different lengths");
    }
    if (gt_k == 0 || gt_k > kMaxGtK) {
        throw std::invalid_argument("gt_k must be between 1 and " + std::to_string(kMaxGtK));
    }
    const std::vector<float> rows = prepared_rows(base, metric);

    Dataset ds;
    ds.metric = metric;
    ds.gt_k = static_cast<std::uint32_t>(std::min<std::size_t>(gt_k, base.rows()));
    ds.base = std::move(base);
    ds.queries = std::move(queries);
    ds.ground_truth.reserve(ds.queries.rows() * ds.gt_k);
    for (std::size_t q = 0; q < ds.queries.rows(); ++q) {
        const std::vector<float> query = prepare_vector(ds.queries.row(q), ds.base.dim, metric);
        for (const Result& r : brute_force(rows, ds.base.dim, metric, query, ds.gt_k)) {
            ds.ground_truth.push_back(r.row);
        }
    }
    return ds;
}

void save_vkd(const Dataset& ds, const std::filesystem::path& path) {
    if (ds.base.dim == 0 || ds.queries.dim != ds.base.dim ||
        ds.ground_truth.size() != ds.queries.rows() * ds.gt_k) {
        throw std::invalid_argument("dataset parts do not fit together");
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw file_error(path, "cannot open for writing");
    }
    out.write(kMagic.data(), kMagic.size());
    write_u32(out, ds.base.dim);
    write_u32(out, static_cast<std::uint32_t>(ds.base.rows()));
    write_u32(out, static_cast<std::uint32_t>(ds.queries.rows()));
    write_u32(out, ds.gt_k);
    write_u32(out, static_cast<std::uint32_t>(ds.metric));
    write_raw(out, std::span<const float>(ds.base.data));
    write_raw(out, std::span<const float>(ds.queries.data));
    write_raw(out, std::span<const std::uint32_t>(ds.ground_truth));
    out.close();
    if (!out) {
        throw file_error(path, "write failed");
    }
}

Dataset load_vkd(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw file_error(path, "cannot open");
    }
    std::array<char, 4> magic{};
    in.read(magic.data(), magic.size());
    if (!in || magic != kMagic) {
        throw file_error(path, "not a .vkd file (wrong magic number)");
    }
    const std::uint32_t dim = read_u32(in);
    const std::uint32_t n_base = read_u32(in);
    const std::uint32_t n_query = read_u32(in);
    const std::uint32_t gt_k = read_u32(in);
    const std::uint32_t metric = read_u32(in);
    if (!in) {
        throw file_error(path, "header is cut short");
    }
    if (dim == 0 || dim > kMaxDim || n_query == 0 || gt_k == 0 || gt_k > kMaxGtK || gt_k > n_base ||
        metric > static_cast<std::uint32_t>(Metric::Cosine)) {
        throw file_error(path, "header has invalid values");
    }
    // Check the size before allocating, so a corrupted header cannot ask for huge buffers.
    const std::uint64_t expected = kHeaderBytes +
                                   ((std::uint64_t{n_base} + n_query) * dim * sizeof(float)) +
                                   (std::uint64_t{n_query} * gt_k * sizeof(std::uint32_t));
    if (std::filesystem::file_size(path) != expected) {
        throw file_error(path, "file size does not match its header (truncated or corrupted?)");
    }

    Dataset ds;
    ds.metric = static_cast<Metric>(metric);
    ds.gt_k = gt_k;
    ds.base = {.dim = dim, .data = std::vector<float>(std::size_t{n_base} * dim)};
    ds.queries = {.dim = dim, .data = std::vector<float>(std::size_t{n_query} * dim)};
    ds.ground_truth.resize(std::size_t{n_query} * gt_k);
    read_raw(in, std::span<float>(ds.base.data));
    read_raw(in, std::span<float>(ds.queries.data));
    read_raw(in, std::span<std::uint32_t>(ds.ground_truth));
    if (!in) {
        throw file_error(path, "read failed");
    }
    if (std::ranges::any_of(ds.ground_truth, [&](std::uint32_t row) { return row >= n_base; })) {
        throw file_error(path, "ground truth points past the last base vector");
    }
    return ds;
}

Dataset make_synthetic(std::uint32_t intrinsic_dim, std::uint32_t dim, std::size_t n_base,
                       std::size_t n_queries, std::uint64_t seed) {
    if (intrinsic_dim == 0 || intrinsic_dim > dim) {
        throw std::invalid_argument("intrinsic dimension must be from 1 to dim");
    }
    constexpr double kNoise = 0.001;  // per coordinate; each coordinate's spread is about 1
    std::mt19937_64 rng(seed);
    // The fixed random matrix: entries N(0, 1/r), so a coordinate's variance is
    // about 1 whatever r is.
    std::vector<double> matrix(std::size_t{dim} * intrinsic_dim);
    for (double& x : matrix) {
        x = gaussian(rng) / std::sqrt(static_cast<double>(intrinsic_dim));
    }
    const auto generate = [&](std::size_t n) {
        Vectors v{.dim = dim, .data = std::vector<float>(n * dim)};
        std::vector<double> z(intrinsic_dim);
        for (std::size_t row = 0; row < n; ++row) {
            for (double& x : z) {
                x = gaussian(rng);
            }
            for (std::size_t j = 0; j < dim; ++j) {
                double sum = 0.0;
                for (std::size_t k = 0; k < intrinsic_dim; ++k) {
                    sum += matrix[(j * intrinsic_dim) + k] * z[k];
                }
                v.data[(row * dim) + j] = static_cast<float>(sum + (kNoise * gaussian(rng)));
            }
        }
        return v;
    };
    Vectors base = generate(n_base);
    Vectors queries = generate(n_queries);
    return make_dataset(std::move(base), std::move(queries), Metric::L2, 100);
}

Dataset load_dataset(const std::string& name) {
    const std::string prefix = "synthetic-";
    if (name.starts_with(prefix)) {
        std::uint32_t r = 0;
        const char* const end = name.data() + name.size();
        const auto [p, err] = std::from_chars(name.data() + prefix.size(), end, r);
        if (err != std::errc() || p != end) {
            throw std::invalid_argument("expected synthetic-<intrinsic dimension>, got " + name);
        }
        return make_synthetic(r, 128, 100'000, 1'000, 1);
    }
    return load_vkd(name);
}

IntrinsicDimension estimate_intrinsic_dimension(const Vectors& vectors, Metric metric,
                                                std::size_t sample, std::size_t k,
                                                std::uint64_t seed, bool within_sample) {
    if (k < 2 || vectors.rows() < k + 1) {
        throw std::invalid_argument("need k >= 2 and more than k points");
    }
    std::vector<std::uint32_t> points = shuffled_indices(vectors.rows(), seed);
    points.resize(std::min(sample, points.size()));
    if (within_sample) {
        // Keep only the sampled points; every one of them is then a sample point.
        Vectors subset{.dim = vectors.dim, .data = {}};
        subset.data.reserve(points.size() * vectors.dim);
        for (const std::uint32_t i : points) {
            const auto row = vectors.row(i);
            subset.data.insert(subset.data.end(), row.begin(), row.end());
        }
        return estimate_intrinsic_dimension(subset, metric, subset.rows(), k, seed, false);
    }
    const std::vector<float> rows = prepared_rows(vectors, metric);

    std::vector<double> mu;
    double mle_sum = 0.0;
    std::vector<double> t;  // distances to the k nearest neighbours, closest first
    for (const std::uint32_t i : points) {
        const std::span<const float> point(rows.data() + (std::size_t{i} * vectors.dim),
                                           vectors.dim);
        t.clear();
        // The k + 1 closest include the point itself, at distance 0.
        for (const Result& hit : brute_force(rows, vectors.dim, metric, point, k + 1)) {
            if (hit.row != i && t.size() < k) {
                // Plain Euclidean distance: L2 distances are stored squared, and for
                // unit vectors |a - b|^2 = 2 * (1 - cos).
                const double d = metric == Metric::L2 ? hit.distance : 2.0 * hit.distance;
                t.push_back(std::sqrt(std::max(d, 0.0)));
            }
        }
        if (t.size() < k || t[0] <= 0.0) {
            continue;  // an exact duplicate: the ratios are undefined
        }
        mu.push_back(t[1] / t[0]);
        double log_sum = 0.0;
        for (std::size_t j = 0; j + 1 < k; ++j) {
            log_sum += std::log(t[k - 1] / t[j]);
        }
        mle_sum += static_cast<double>(k - 1) / log_sum;
    }
    if (mu.size() < 10) {
        throw std::runtime_error("too few points without duplicates to estimate");
    }

    std::ranges::sort(mu);
    const auto n = static_cast<double>(mu.size());
    const auto used = static_cast<std::size_t>(0.9 * n);
    double xy = 0.0;
    double xx = 0.0;
    for (std::size_t i = 0; i < used; ++i) {
        const double x = std::log(mu[i]);
        const double y = -std::log(1.0 - (static_cast<double>(i + 1) / n));
        xy += x * y;
        xx += x * x;
    }
    return {.twonn = xy / xx, .mle = mle_sum / n, .points = mu.size()};
}

double recall_at_k(std::span<const Result> found, std::span<const std::uint32_t> truth,
                   std::size_t k) {
    if (k == 0 || k > truth.size()) {
        throw std::invalid_argument("k must be between 1 and the number of true neighbours");
    }
    const auto true_top = truth.first(k);
    std::size_t hits = 0;
    for (const Result& r : found.first(std::min(k, found.size()))) {
        if (std::ranges::find(true_top, r.row) != true_top.end()) {
            ++hits;
        }
    }
    return static_cast<double>(hits) / static_cast<double>(k);
}

}  // namespace vektor
