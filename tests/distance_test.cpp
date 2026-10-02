#include "core/distance.hpp"

#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <vector>

namespace vektor {
namespace {

TEST(Distance, L2SquaredHandComputed) {
    const std::vector<float> a = {1, 2, 3};
    const std::vector<float> b = {4, 6, 3};
    EXPECT_FLOAT_EQ(l2_squared(a, b), 25.0F);  // 3^2 + 4^2 + 0^2
    EXPECT_FLOAT_EQ(l2_squared(a, a), 0.0F);
}

TEST(Distance, DotHandComputed) {
    const std::vector<float> a = {1, 2, 3};
    const std::vector<float> b = {4, -5, 6};
    EXPECT_FLOAT_EQ(dot(a, b), 12.0F);  // 4 - 10 + 18
}

TEST(Distance, CosineOfUnitVectors) {
    const std::vector<float> x = {1, 0};
    const std::vector<float> y = {0, 1};
    const std::vector<float> minus_x = {-1, 0};
    EXPECT_FLOAT_EQ(cosine_distance(x, x), 0.0F);
    EXPECT_FLOAT_EQ(cosine_distance(x, y), 1.0F);
    EXPECT_FLOAT_EQ(cosine_distance(x, minus_x), 2.0F);
}

TEST(Distance, DistancePicksMetric) {
    const std::vector<float> a = {1, 0};
    const std::vector<float> b = {0, 1};
    EXPECT_FLOAT_EQ(distance(Metric::L2, a, b), 2.0F);
    EXPECT_FLOAT_EQ(distance(Metric::Cosine, a, b), 1.0F);
}

// Odd lengths check the leftover elements after the SIMD part of the loop.
TEST(Distance, LongVectorsMatchDoublePrecision) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    for (const std::size_t n : {1, 3, 17, 128, 131, 784}) {
        std::vector<float> a(n);
        std::vector<float> b(n);
        double l2 = 0;
        double d = 0;
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = uniform(rng);
            b[i] = uniform(rng);
            l2 += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
            d += static_cast<double>(a[i]) * b[i];
        }
        EXPECT_NEAR(l2_squared(a, b), l2, 1e-4 * static_cast<double>(n)) << "n=" << n;
        EXPECT_NEAR(dot(a, b), d, 1e-4 * static_cast<double>(n)) << "n=" << n;
    }
}

TEST(Distance, NormalizeScalesToLengthOne) {
    std::vector<float> v = {3, 4};
    ASSERT_TRUE(normalize(v));
    EXPECT_FLOAT_EQ(v[0], 0.6F);
    EXPECT_FLOAT_EQ(v[1], 0.8F);
}

TEST(Distance, NormalizeHandlesHugeValues) {
    std::vector<float> v = {3e30F, 4e30F};  // squares overflow float, not double
    ASSERT_TRUE(normalize(v));
    EXPECT_FLOAT_EQ(v[0], 0.6F);
    EXPECT_FLOAT_EQ(v[1], 0.8F);
}

TEST(Distance, NormalizeRejectsZeroVector) {
    std::vector<float> v = {0, 0, 0};
    EXPECT_FALSE(normalize(v));
    EXPECT_EQ(v, (std::vector<float>{0, 0, 0}));
}

TEST(Distance, ParseMetric) {
    EXPECT_EQ(parse_metric("l2"), Metric::L2);
    EXPECT_EQ(parse_metric("cosine"), Metric::Cosine);
    EXPECT_THROW((void)parse_metric("dot"), std::invalid_argument);
    EXPECT_EQ(to_string(Metric::L2), "l2");
    EXPECT_EQ(to_string(Metric::Cosine), "cosine");
}

}  // namespace
}  // namespace vektor
