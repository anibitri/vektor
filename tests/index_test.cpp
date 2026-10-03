#include "core/index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace vektor {
namespace {

std::vector<float> random_vector(std::mt19937_64& rng, std::size_t dim) {
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    std::vector<float> v(dim);
    for (float& x : v) {
        x = uniform(rng);
    }
    return v;
}

// Reference answer: compute every distance, sort, keep the first k.
std::vector<Result> sort_everything(const Index& index, std::vector<float> query, std::size_t k) {
    if (index.metric() == Metric::Cosine) {
        normalize(query);
    }
    std::vector<Result> all;
    all.reserve(index.size());
    for (std::uint32_t row = 0; row < index.size(); ++row) {
        all.push_back({.row = row, .distance = distance(index.metric(), query, index.vector(row))});
    }
    std::ranges::sort(all, [](const Result& a, const Result& b) {
        return a.distance != b.distance ? a.distance < b.distance : a.row < b.row;
    });
    all.resize(std::min(k, all.size()));
    return all;
}

TEST(Index, RejectsZeroDimension) { EXPECT_THROW(Index(0, Metric::L2), std::invalid_argument); }

TEST(Index, AddReturnsRowNumbers) {
    Index index(2, Metric::L2);
    EXPECT_EQ(index.add(std::vector<float>{1, 2}), 0U);
    EXPECT_EQ(index.add(std::vector<float>{3, 4}), 1U);
    EXPECT_EQ(index.size(), 2U);
    EXPECT_EQ(index.vector(1)[0], 3.0F);
}

TEST(Index, AddRejectsBadVectors) {
    Index index(2, Metric::L2);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_THROW(index.add(std::vector<float>{1}), std::invalid_argument);
    EXPECT_THROW(index.add(std::vector<float>{1, 2, 3}), std::invalid_argument);
    EXPECT_THROW(index.add(std::vector<float>{}), std::invalid_argument);
    EXPECT_THROW(index.add(std::vector<float>{1, nan}), std::invalid_argument);
    EXPECT_THROW(index.add(std::vector<float>{-inf, 1}), std::invalid_argument);
    EXPECT_EQ(index.size(), 0U);
    EXPECT_NO_THROW(index.add(std::vector<float>{0, 0}));  // fine for L2
}

TEST(Index, CosineNormalisesAndRejectsZeroVector) {
    Index index(2, Metric::Cosine);
    EXPECT_THROW(index.add(std::vector<float>{0, 0}), std::invalid_argument);
    index.add(std::vector<float>{3, 4});
    EXPECT_FLOAT_EQ(index.vector(0)[0], 0.6F);
    EXPECT_FLOAT_EQ(index.vector(0)[1], 0.8F);
}

TEST(Index, SearchRejectsBadQuery) {
    Index index(3, Metric::L2);
    index.add(std::vector<float>{1, 2, 3});
    EXPECT_THROW((void)index.search_exact(std::vector<float>{1, 2}, 1), std::invalid_argument);
}

TEST(Index, ExactSearchEdgeCases) {
    Index index(2, Metric::L2);
    EXPECT_TRUE(index.search_exact(std::vector<float>{1, 1}, 5).empty());  // empty index
    index.add(std::vector<float>{1, 1});
    EXPECT_TRUE(index.search_exact(std::vector<float>{1, 1}, 0).empty());   // k = 0
    EXPECT_EQ(index.search_exact(std::vector<float>{1, 1}, 5).size(), 1U);  // k > size
}

TEST(Index, ExactSearchMatchesSortEverything) {
    for (std::uint64_t seed = 0; seed < 20; ++seed) {
        for (const Metric metric : {Metric::L2, Metric::Cosine}) {
            std::mt19937_64 rng(seed);
            const std::size_t dim = 1 + (rng() % 40);
            const std::size_t n = 1 + (rng() % 200);
            Index index(dim, metric);
            for (std::size_t i = 0; i < n; ++i) {
                index.add(random_vector(rng, dim));
            }
            const std::vector<float> query = random_vector(rng, dim);
            for (const std::size_t k : {std::size_t{1}, std::size_t{5}, n, n + 3}) {
                const std::vector<Result> got = index.search_exact(query, k);
                const std::vector<Result> want = sort_everything(index, query, k);
                ASSERT_EQ(got.size(), want.size()) << "seed " << seed << ", k " << k;
                for (std::size_t i = 0; i < got.size(); ++i) {
                    EXPECT_EQ(got[i].row, want[i].row) << "seed " << seed << ", k " << k;
                    EXPECT_NEAR(got[i].distance, want[i].distance,
                                1e-5);  // SIMD sum order may differ
                }
            }
        }
    }
}

TEST(Index, ExactSearchFindsEachStoredVector) {
    for (std::uint64_t seed = 0; seed < 50; ++seed) {
        std::mt19937_64 rng(seed);
        const std::size_t dim = 1 + (rng() % 16);
        const std::size_t n = 1 + (rng() % 50);
        Index index(dim, Metric::L2);
        std::vector<std::vector<float>> stored;
        for (std::size_t i = 0; i < n; ++i) {
            stored.push_back(random_vector(rng, dim));
            index.add(stored.back());
        }
        for (std::uint32_t row = 0; row < n; ++row) {
            const std::vector<Result> got = index.search_exact(stored[row], 1);
            ASSERT_EQ(got.size(), 1U);
            EXPECT_EQ(got[0].row, row) << "seed " << seed;
            EXPECT_EQ(got[0].distance, 0.0F);
        }
    }
}

TEST(Index, ScalingTheQueryDoesNotChangeCosineResults) {
    std::mt19937_64 rng(3);
    Index index(8, Metric::Cosine);
    for (int i = 0; i < 100; ++i) {
        index.add(random_vector(rng, 8));
    }
    std::vector<float> query = random_vector(rng, 8);
    const std::vector<Result> before = index.search_exact(query, 10);
    for (float& x : query) {
        x *= 7.5F;
    }
    const std::vector<Result> after = index.search_exact(query, 10);
    ASSERT_EQ(before.size(), after.size());
    for (std::size_t i = 0; i < before.size(); ++i) {
        EXPECT_EQ(before[i].row, after[i].row);
    }
}

TEST(Index, MemoryBytesCountsReservedVectors) {
    Index index(4, Metric::L2);
    index.reserve(10);
    EXPECT_GE(index.memory_bytes(), 40 * sizeof(float));  // 10 vectors x 4 floats
}

}  // namespace
}  // namespace vektor
