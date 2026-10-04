#include "bench/dataset.hpp"

#include <algorithm>
#include <array>
#include <fstream>
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
    // Checked (and for cosine, normalised) copy of the base vectors to search.
    std::vector<float> rows;
    rows.reserve(base.data.size());
    for (std::size_t i = 0; i < base.rows(); ++i) {
        const std::vector<float> v = prepare_vector(base.row(i), base.dim, metric);
        rows.insert(rows.end(), v.begin(), v.end());
    }

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
