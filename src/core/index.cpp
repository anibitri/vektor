// Storage, input checks and exact search. The HNSW graph code is in hnsw.cpp.

#include "core/index.hpp"

#include <cmath>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

namespace vektor {

std::vector<float> prepare_vector(std::span<const float> vec, std::size_t dim, Metric metric) {
    if (vec.size() != dim) {
        throw std::invalid_argument("vector has " + std::to_string(vec.size()) +
                                    " values, expected " + std::to_string(dim));
    }
    std::vector<float> out(vec.begin(), vec.end());
    for (const float x : out) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument("vector contains NaN or Inf");
        }
    }
    if (metric == Metric::Cosine && !normalize(out)) {
        throw std::invalid_argument("cosine metric needs a non-zero vector");
    }
    return out;
}

std::vector<Result> brute_force(std::span<const float> rows, std::size_t dim, Metric metric,
                                std::span<const float> query, std::size_t k) {
    if (k == 0) {
        return {};
    }
    // Max-heap of the best k seen so far: the worst of them is on top, ready to be replaced.
    std::priority_queue<std::pair<float, std::uint32_t>> best;
    const std::size_t n = rows.size() / dim;
    for (std::uint32_t row = 0; row < n; ++row) {
        const float d = distance(metric, query, rows.subspan(row * dim, dim));
        if (best.size() < k) {
            best.emplace(d, row);
        } else if (d < best.top().first) {
            best.pop();
            best.emplace(d, row);
        }
    }
    // Pop worst first, filling the output from the back.
    std::vector<Result> out(best.size());
    for (auto i = out.size(); i-- > 0;) {
        out[i] = {.row = best.top().second, .distance = best.top().first};
        best.pop();
    }
    return out;
}

Index::Index(std::size_t dim, Metric metric, HnswParams params)
    : dim_(dim), metric_(metric), p_(params), rng_(params.seed) {
    if (dim == 0) {
        throw std::invalid_argument("dimension must be at least 1");
    }
    // M_max0 is stored as a 16-bit count in .vkt files.
    if (p_.M < 2 || p_.M_max0 < p_.M || p_.M_max0 > 65535 || p_.ef_construction == 0) {
        throw std::invalid_argument(
            "HNSW needs 2 <= M <= M_max0 <= 65535 and ef_construction >= 1");
    }
}

std::uint32_t Index::add(std::span<const float> vec) {
    if (has_ids()) {
        throw std::invalid_argument("this index has IDs: add the vector with an ID");
    }
    return insert(vec);
}

std::uint32_t Index::add(const std::string& id, std::span<const float> vec, std::string meta) {
    if (size() > 0 && !has_ids()) {
        throw std::invalid_argument("this index has no IDs: add the vector without one");
    }
    if (contains(id)) {
        throw std::invalid_argument("ID '" + id + "' is already in the index");
    }
    const std::uint32_t row = insert(vec);  // checks the vector before changing anything
    ids_.push_back(id);
    meta_.push_back(std::move(meta));
    rows_by_id_.emplace(id, row);
    return row;
}

void Index::reserve(std::size_t n) {
    vectors_.reserve(n * dim_);
    links_.reserve(n);
    if (has_ids()) {
        ids_.reserve(n);
        meta_.reserve(n);
    }
}

std::vector<Result> Index::search_exact(std::span<const float> query, std::size_t k) const {
    const std::vector<float> q = prepare_vector(query, dim_, metric_);
    return brute_force(vectors_, dim_, metric_, q, k);
}

std::size_t Index::memory_bytes() const {
    std::size_t bytes = vectors_.capacity() * sizeof(float);
    bytes += links_.capacity() * sizeof(std::vector<std::vector<std::uint32_t>>);
    for (const auto& layers : links_) {
        bytes += layers.capacity() * sizeof(std::vector<std::uint32_t>);
        for (const auto& list : layers) {
            bytes += list.capacity() * sizeof(std::uint32_t);
        }
    }
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        bytes += (2 * sizeof(std::string)) + ids_[i].size() + meta_[i].size();
    }
    return bytes;
}

}  // namespace vektor
