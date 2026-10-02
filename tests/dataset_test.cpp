#include "bench/dataset.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace vektor {
namespace {

std::filesystem::path temp_file(const std::string& name) {
    return std::filesystem::path(testing::TempDir()) / ("vektor_" + name);
}

// Writes rows in .fvecs format: an int32 length, then the floats.
void write_fvecs(const std::filesystem::path& path, const std::vector<std::vector<float>>& rows) {
    std::ofstream out(path, std::ios::binary);
    for (const auto& row : rows) {
        const auto len = static_cast<std::int32_t>(row.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof len);
        out.write(reinterpret_cast<const char*>(row.data()),
                  static_cast<std::streamsize>(row.size() * sizeof(float)));
    }
}

Vectors random_vectors(std::mt19937_64& rng, std::uint32_t dim, std::size_t rows) {
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    Vectors v{.dim = dim, .data = std::vector<float>(rows * dim)};
    for (float& x : v.data) {
        x = uniform(rng);
    }
    return v;
}

TEST(Fvecs, ReadsRowsUpToLimit) {
    const auto path = temp_file("read.fvecs");
    write_fvecs(path, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    const Vectors all = read_fvecs(path, 100);
    EXPECT_EQ(all.dim, 3U);
    EXPECT_EQ(all.rows(), 3U);
    EXPECT_EQ(all.data, (std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}));
    const Vectors two = read_fvecs(path, 2);
    EXPECT_EQ(two.rows(), 2U);
    EXPECT_EQ(two.row(1)[0], 4.0F);
}

TEST(Fvecs, RejectsBadFiles) {
    EXPECT_THROW((void)read_fvecs(temp_file("missing.fvecs"), 10), std::runtime_error);

    const auto mixed = temp_file("mixed.fvecs");
    write_fvecs(mixed, {{1, 2}, {1, 2, 3}});
    EXPECT_THROW((void)read_fvecs(mixed, 10), std::runtime_error);

    const auto empty = temp_file("empty.fvecs");
    write_fvecs(empty, {});
    EXPECT_THROW((void)read_fvecs(empty, 10), std::runtime_error);

    const auto cut = temp_file("cut.fvecs");
    write_fvecs(cut, {{1, 2, 3}, {4, 5, 6}});
    std::filesystem::resize_file(cut, std::filesystem::file_size(cut) - 2);
    EXPECT_THROW((void)read_fvecs(cut, 10), std::runtime_error);
}

TEST(Dataset, GroundTruthMatchesExactSearch) {
    for (const Metric metric : {Metric::L2, Metric::Cosine}) {
        std::mt19937_64 rng(11);
        const Vectors base = random_vectors(rng, 6, 300);
        const Vectors queries = random_vectors(rng, 6, 20);
        const Dataset ds = make_dataset(base, queries, metric, 10);
        ASSERT_EQ(ds.gt_k, 10U);
        ASSERT_EQ(ds.ground_truth.size(), 20U * 10U);

        Index index(6, metric);
        for (std::size_t i = 0; i < base.rows(); ++i) {
            index.add(base.row(i));
        }
        for (std::size_t q = 0; q < queries.rows(); ++q) {
            const std::vector<Result> want = index.search_exact(queries.row(q), 10);
            for (std::size_t i = 0; i < want.size(); ++i) {
                EXPECT_EQ(ds.truth(q)[i], want[i].row);
            }
        }
    }
}

TEST(Dataset, CapsGroundTruthAtBaseSize) {
    std::mt19937_64 rng(1);
    const Dataset ds =
        make_dataset(random_vectors(rng, 4, 5), random_vectors(rng, 4, 2), Metric::L2, 100);
    EXPECT_EQ(ds.gt_k, 5U);
}

TEST(Dataset, RejectsMismatchedDimensions) {
    std::mt19937_64 rng(1);
    EXPECT_THROW(
        (void)make_dataset(random_vectors(rng, 4, 5), random_vectors(rng, 3, 2), Metric::L2, 3),
        std::invalid_argument);
}

TEST(Vkd, SaveLoadRoundTrip) {
    std::mt19937_64 rng(5);
    const Dataset ds =
        make_dataset(random_vectors(rng, 7, 50), random_vectors(rng, 7, 4), Metric::Cosine, 8);
    const auto path = temp_file("round_trip.vkd");
    save_vkd(ds, path);
    const Dataset loaded = load_vkd(path);
    EXPECT_EQ(loaded.metric, Metric::Cosine);
    EXPECT_EQ(loaded.base.dim, 7U);
    EXPECT_EQ(loaded.queries.dim, 7U);
    EXPECT_EQ(loaded.gt_k, 8U);
    EXPECT_EQ(loaded.base.data, ds.base.data);
    EXPECT_EQ(loaded.queries.data, ds.queries.data);
    EXPECT_EQ(loaded.ground_truth, ds.ground_truth);
}

class VkdCorruption : public testing::Test {
protected:
    void SetUp() override {
        std::mt19937_64 rng(9);
        save_vkd(make_dataset(random_vectors(rng, 3, 20), random_vectors(rng, 3, 2), Metric::L2, 5),
                 path_);
    }

    // Overwrites 4 bytes at the given offset.
    void poke(std::streamoff offset, std::uint32_t value) const {
        std::fstream file(path_, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(offset);
        file.write(reinterpret_cast<const char*>(&value), sizeof value);
    }

    std::filesystem::path path_ = temp_file("corrupt.vkd");
};

TEST_F(VkdCorruption, WrongMagic) {
    poke(0, 0x12345678);
    EXPECT_THROW((void)load_vkd(path_), std::runtime_error);
}

TEST_F(VkdCorruption, Truncated) {
    std::filesystem::resize_file(path_, std::filesystem::file_size(path_) - 1);
    EXPECT_THROW((void)load_vkd(path_), std::runtime_error);
}

TEST_F(VkdCorruption, HeaderSaysMoreRows) {
    poke(8, 1'000'000);  // n_base
    EXPECT_THROW((void)load_vkd(path_), std::runtime_error);
}

TEST_F(VkdCorruption, GroundTruthOutOfRange) {
    const auto last = static_cast<std::streamoff>(std::filesystem::file_size(path_)) - 4;
    poke(last, 20);  // there are only 20 base rows: 0..19
    EXPECT_THROW((void)load_vkd(path_), std::runtime_error);
}

TEST(Recall, CountsTrueNeighboursInTopK) {
    const std::vector<std::uint32_t> truth = {4, 7, 1, 9};
    const std::vector<Result> found = {
        {.row = 7, .distance = 0}, {.row = 2, .distance = 0}, {.row = 4, .distance = 0}};
    EXPECT_DOUBLE_EQ(recall_at_k(found, truth, 1), 0.0);  // 7 is not the true top 1
    EXPECT_DOUBLE_EQ(recall_at_k(found, truth, 2), 0.5);  // {7, 2} vs {4, 7}
    EXPECT_DOUBLE_EQ(recall_at_k(found, truth, 3), 2.0 / 3.0);
    EXPECT_DOUBLE_EQ(recall_at_k(found, truth, 4), 0.5);  // only 3 found, 2 are true
    EXPECT_THROW((void)recall_at_k(found, truth, 0), std::invalid_argument);
    EXPECT_THROW((void)recall_at_k(found, truth, 5), std::invalid_argument);
}

}  // namespace
}  // namespace vektor
