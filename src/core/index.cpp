#include "core/index.hpp"

#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

namespace vektor {

Index::Index(std::size_t dim, Metric metric) : dim_(dim), metric_(metric) {
    if (dim == 0) {
        throw std::invalid_argument("dimension must be at least 1");
    }
}

std::vector<float> Index::prepare(std::span<const float> vec) const {
    if (vec.size() != dim_) {
        throw std::invalid_argument("vector has " + std::to_string(vec.size()) +
                                    " values, expected " + std::to_string(dim_));
    }
    std::vector<float> out(vec.begin(), vec.end());
    for (const float x : out) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument("vector contains NaN or Inf");
        }
    }
    if (metric_ == Metric::Cosine && !normalize(out)) {
        throw std::invalid_argument("cosine metric needs a non-zero vector");
    }
    return out;
}

std::uint32_t Index::add(std::span<const float> vec) {
    if (size() >= std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("index is full");
    }
    const std::vector<float> v = prepare(vec);
    const auto row = static_cast<std::uint32_t>(size());
    vectors_.insert(vectors_.end(), v.begin(), v.end());
    return row;
}

void Index::reserve(std::size_t n) { vectors_.reserve(n * dim_); }

std::vector<Result> Index::search_exact(std::span<const float> query, std::size_t k) const {
    const std::vector<float> q = prepare(query);
    if (k == 0) {
        return {};
    }
    // Max-heap of the best k seen so far: the worst of them is on top, ready to be replaced.
    std::priority_queue<std::pair<float, std::uint32_t>> best;
    const std::size_t n = size();
    for (std::uint32_t row = 0; row < n; ++row) {
        const float d = distance(metric_, q, vector(row));
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

std::size_t Index::memory_bytes() const { return vectors_.capacity() * sizeof(float); }

}  // namespace vektor
