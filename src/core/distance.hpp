#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace vektor {

enum class Metric : std::uint8_t { L2 = 0, Cosine = 1 };

// The distance functions are plain loops. The `omp simd reduction` line lets
// the compiler add up the sum in several parts at once (SIMD). Without it,
// strict floating-point rules force one addition at a time, even at -O3.
// It needs the -fopenmp-simd flag, which turns on these hints only (no threads).

// Squared Euclidean distance. We skip the square root because it does not
// change which vector is closest.
inline float l2_squared(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0F;
#pragma omp simd reduction(+ : sum)
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline float dot(std::span<const float> a, std::span<const float> b) {
    float sum = 0.0F;
#pragma omp simd reduction(+ : sum)
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Cosine distance (1 - cos(angle)) for vectors already scaled to length 1.
inline float cosine_distance(std::span<const float> a, std::span<const float> b) {
    return 1.0F - dot(a, b);
}

inline float distance(Metric metric, std::span<const float> a, std::span<const float> b) {
    return metric == Metric::L2 ? l2_squared(a, b) : cosine_distance(a, b);
}

// Scales v to length 1. Returns false and leaves v alone if v is all zeros.
// Sums in double so very large values do not overflow.
inline bool normalize(std::span<float> v) {
    double sum = 0.0;
    for (const float x : v) {
        sum += static_cast<double>(x) * x;
    }
    if (sum == 0.0) {
        return false;
    }
    const double norm = std::sqrt(sum);
    for (float& x : v) {
        x = static_cast<float>(x / norm);
    }
    return true;
}

inline Metric parse_metric(std::string_view name) {
    if (name == "l2") {
        return Metric::L2;
    }
    if (name == "cosine") {
        return Metric::Cosine;
    }
    throw std::invalid_argument("unknown metric '" + std::string(name) +
                                "' (expected 'l2' or 'cosine')");
}

inline std::string_view to_string(Metric metric) { return metric == Metric::L2 ? "l2" : "cosine"; }

}  // namespace vektor
