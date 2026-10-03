#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "core/index.hpp"

namespace vektor {
namespace {

// Uniform values in [-1, 1) made from raw random bits, so the data is the same
// with every standard library (std::uniform_real_distribution is not).
std::vector<float> random_vector(std::mt19937_64& rng, std::size_t dim) {
    std::vector<float> v(dim);
    for (float& x : v) {
        x = (static_cast<float>(rng() >> 40U) * 0x1.0p-24F * 2.0F) - 1.0F;
    }
    return v;
}

Index build(std::size_t n, std::size_t dim, Metric metric, HnswParams params,
            std::uint64_t data_seed) {
    std::mt19937_64 rng(data_seed);
    Index index(dim, metric, params);
    index.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        index.add(random_vector(rng, dim));
    }
    return index;
}

TEST(Hnsw, RejectsBadParams) {
    EXPECT_THROW(Index(4, Metric::L2, {.M = 1}), std::invalid_argument);
    EXPECT_THROW(Index(4, Metric::L2, {.M = 16, .M_max0 = 8}), std::invalid_argument);
    EXPECT_THROW(Index(4, Metric::L2, {.ef_construction = 0}), std::invalid_argument);
}

TEST(Hnsw, EmptyIndexAndZeroK) {
    Index index(3, Metric::L2);
    EXPECT_TRUE(index.search(std::vector<float>{1, 2, 3}, 5, 10).empty());
    index.add(std::vector<float>{1, 2, 3});
    EXPECT_TRUE(index.search(std::vector<float>{1, 2, 3}, 0, 10).empty());
    EXPECT_THROW((void)index.search(std::vector<float>{1, 2}, 1, 10), std::invalid_argument);
    const std::vector<Result> one = index.search(std::vector<float>{1, 2, 3}, 5, 10);
    ASSERT_EQ(one.size(), 1U);
    EXPECT_EQ(one[0].row, 0U);
}

// Small M forces many neighbour lists to be trimmed, which is where bugs hide.
TEST(Hnsw, GraphInvariants) {
    for (const bool heuristic : {true, false}) {
        const Index index =
            build(2000, 8, Metric::L2,
                  {.M = 4, .M_max0 = 8, .ef_construction = 32, .heuristic = heuristic}, 1);
        int top = -1;
        for (std::uint32_t node = 0; node < index.size(); ++node) {
            const int level = index.level(node);
            ASSERT_GE(level, 0);
            top = std::max(top, level);
            for (int layer = 0; layer <= level; ++layer) {
                const auto links = index.neighbours(node, layer);
                EXPECT_LE(links.size(), index.max_links(layer));
                EXPECT_EQ(std::set<std::uint32_t>(links.begin(), links.end()).size(), links.size())
                    << "duplicate link at node " << node;
                for (const std::uint32_t next : links) {
                    ASSERT_LT(next, index.size());
                    EXPECT_NE(next, node) << "self link";
                    EXPECT_GE(index.level(next), layer) << "link to a node missing from the layer";
                }
            }
            if (node > 0) {
                EXPECT_FALSE(index.neighbours(node, 0).empty())
                    << "node " << node << " has no links";
            }
        }
        EXPECT_EQ(index.max_level(), top);
        EXPECT_EQ(index.level(index.entry_point()), top);
    }
}

TEST(Hnsw, ResultsAreSortedAndValid) {
    const Index index = build(1000, 16, Metric::Cosine, {.M = 8, .M_max0 = 16}, 2);
    std::mt19937_64 rng(99);
    for (int i = 0; i < 20; ++i) {
        std::vector<float> query = random_vector(rng, 16);
        const std::vector<Result> got = index.search(query, 10, 40);
        ASSERT_EQ(got.size(), 10U);
        normalize(query);
        for (std::size_t j = 0; j < got.size(); ++j) {
            ASSERT_LT(got[j].row, index.size());
            // Tiny differences are expected: the SIMD sum may add in a different order
            // where the distance function is inlined into different code.
            EXPECT_NEAR(got[j].distance, cosine_distance(query, index.vector(got[j].row)), 1e-5);
            if (j > 0) {
                EXPECT_LE(got[j - 1].distance, got[j].distance);
            }
        }
    }
}

// With fewer nodes than M_max0 no list is ever trimmed on layer 0, so every node
// is reachable, and with ef >= n the search must explore everything: the answer
// has to match exact search.
TEST(Hnsw, SmallInputsGiveExactResults) {
    for (std::uint64_t seed = 0; seed < 50; ++seed) {
        for (const Metric metric : {Metric::L2, Metric::Cosine}) {
            for (const bool heuristic : {true, false}) {
                std::mt19937_64 rng(seed);
                const std::size_t n = 1 + (rng() % 30);
                const std::size_t dim = 1 + (rng() % 12);
                const Index index =
                    build(n, dim, metric, {.seed = seed, .heuristic = heuristic}, seed + 1000);
                const std::vector<float> query = random_vector(rng, dim);
                const std::vector<Result> got = index.search(query, 5, n);
                const std::vector<Result> want = index.search_exact(query, 5);
                ASSERT_EQ(got.size(), want.size()) << "seed " << seed;
                for (std::size_t i = 0; i < got.size(); ++i) {
                    EXPECT_EQ(got[i].row, want[i].row) << "seed " << seed;
                }
            }
        }
    }
}

// Recall regression test: if a change makes HNSW worse, this fails in CI.
TEST(Hnsw, RecallAtTenOnRandomData) {
    constexpr std::size_t kDim = 32;
    constexpr std::size_t kQueries = 200;
    const Index index = build(5000, kDim, Metric::L2, {.M = 16, .ef_construction = 100}, 7);
    std::mt19937_64 rng(8);
    std::size_t hits = 0;
    for (std::size_t q = 0; q < kQueries; ++q) {
        const std::vector<float> query = random_vector(rng, kDim);
        const std::vector<Result> want = index.search_exact(query, 10);
        for (const Result& r : index.search(query, 10, 50)) {
            hits += static_cast<std::size_t>(
                std::ranges::any_of(want, [&](const Result& w) { return w.row == r.row; }));
        }
    }
    const double recall = static_cast<double>(hits) / (kQueries * 10.0);
    RecordProperty("recall_at_10", std::to_string(recall));
    EXPECT_GE(recall, 0.9);
}

TEST(Hnsw, HashSetAndVisitTagsGiveSameResults) {
    const Index index = build(2000, 12, Metric::L2, {.M = 8, .M_max0 = 16}, 3);
    std::mt19937_64 rng(4);
    for (int i = 0; i < 20; ++i) {
        const std::vector<float> query = random_vector(rng, 12);
        const std::vector<Result> tags = index.search(query, 10, 30, Visited::Tags);
        const std::vector<Result> hash = index.search(query, 10, 30, Visited::HashSet);
        ASSERT_EQ(tags.size(), hash.size());
        for (std::size_t j = 0; j < tags.size(); ++j) {
            EXPECT_EQ(tags[j].row, hash[j].row);
        }
    }
}

TEST(Hnsw, SameSeedGivesSameGraph) {
    const Index a = build(500, 8, Metric::L2, {.M = 6, .M_max0 = 12, .seed = 5}, 9);
    const Index b = build(500, 8, Metric::L2, {.M = 6, .M_max0 = 12, .seed = 5}, 9);
    ASSERT_EQ(a.max_level(), b.max_level());
    ASSERT_EQ(a.entry_point(), b.entry_point());
    for (std::uint32_t node = 0; node < a.size(); ++node) {
        ASSERT_EQ(a.level(node), b.level(node));
        for (int layer = 0; layer <= a.level(node); ++layer) {
            const auto la = a.neighbours(node, layer);
            const auto lb = b.neighbours(node, layer);
            EXPECT_TRUE(std::ranges::equal(la, lb)) << "node " << node << " layer " << layer;
        }
    }
}

TEST(Hnsw, MemoryIncludesGraph) {
    const Index index = build(100, 4, Metric::L2, {}, 1);
    EXPECT_GT(index.memory_bytes(), 400 * sizeof(float));
}

}  // namespace
}  // namespace vektor
