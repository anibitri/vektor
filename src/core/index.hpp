#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/distance.hpp"

namespace vektor {

// One search hit: the vector's row number (its position in insertion order)
// and its distance to the query.
struct Result {
    std::uint32_t row;
    float distance;
};

class Index {
public:
    Index(std::size_t dim, Metric metric);

    // Adds a copy of vec and returns its row number. Throws std::invalid_argument
    // if vec has the wrong length, contains NaN or Inf, or is all zeros under cosine.
    std::uint32_t add(std::span<const float> vec);

    // Reserves memory for n vectors in total, so adding them does not reallocate.
    void reserve(std::size_t n);

    // Exact search: compares the query with every vector.
    // Returns up to k results, closest first.
    [[nodiscard]] std::vector<Result> search_exact(std::span<const float> query,
                                                   std::size_t k) const;

    [[nodiscard]] std::size_t size() const { return vectors_.size() / dim_; }
    [[nodiscard]] std::size_t dim() const { return dim_; }
    [[nodiscard]] Metric metric() const { return metric_; }

    // The stored vector (normalised if the metric is cosine).
    [[nodiscard]] std::span<const float> vector(std::uint32_t row) const {
        return {vectors_.data() + (static_cast<std::size_t>(row) * dim_), dim_};
    }

    // Bytes allocated for the vectors.
    [[nodiscard]] std::size_t memory_bytes() const;

private:
    // Checks vec and returns a copy, normalised if the metric is cosine.
    [[nodiscard]] std::vector<float> prepare(std::span<const float> vec) const;

    std::size_t dim_;
    Metric metric_;
    std::vector<float> vectors_;  // row i is vectors_[i * dim_ .. (i + 1) * dim_)
};

}  // namespace vektor
