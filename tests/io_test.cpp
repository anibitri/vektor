#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/binary.hpp"
#include "core/index.hpp"

namespace vektor {
namespace {

std::filesystem::path temp_file(const std::string& name) {
    return std::filesystem::path(testing::TempDir()) / ("vektor_" + name);
}

std::vector<float> random_vector(std::mt19937_64& rng, std::size_t dim) {
    std::vector<float> v(dim);
    for (float& x : v) {
        x = (static_cast<float>(rng() >> 40U) * 0x1.0p-24F * 2.0F) - 1.0F;
    }
    return v;
}

Index build(std::size_t n, std::size_t dim, Metric metric, bool with_ids) {
    std::mt19937_64 rng(17);
    Index index(dim, metric, {.M = 6, .M_max0 = 12, .ef_construction = 40});
    for (std::size_t i = 0; i < n; ++i) {
        if (with_ids) {
            index.add("doc-" + std::to_string(i), random_vector(rng, dim),
                      R"({"n": )" + std::to_string(i) + "}");
        } else {
            index.add(random_vector(rng, dim));
        }
    }
    return index;
}

void expect_same(const Index& a, const Index& b) {
    ASSERT_EQ(a.size(), b.size());
    ASSERT_EQ(a.dim(), b.dim());
    EXPECT_EQ(a.metric(), b.metric());
    EXPECT_EQ(a.params().M, b.params().M);
    EXPECT_EQ(a.params().M_max0, b.params().M_max0);
    EXPECT_EQ(a.params().ef_construction, b.params().ef_construction);
    EXPECT_EQ(a.params().seed, b.params().seed);
    EXPECT_EQ(a.params().heuristic, b.params().heuristic);
    EXPECT_EQ(a.entry_point(), b.entry_point());
    EXPECT_EQ(a.max_level(), b.max_level());
    EXPECT_EQ(a.has_ids(), b.has_ids());
    for (std::uint32_t row = 0; row < a.size(); ++row) {
        ASSERT_TRUE(std::ranges::equal(a.vector(row), b.vector(row)));
        ASSERT_EQ(a.level(row), b.level(row));
        for (int layer = 0; layer <= a.level(row); ++layer) {
            ASSERT_TRUE(std::ranges::equal(a.neighbours(row, layer), b.neighbours(row, layer)));
        }
        EXPECT_EQ(a.id(row), b.id(row));
        EXPECT_EQ(a.meta(row), b.meta(row));
    }
}

TEST(Crc32, MatchesStandardCheckValue) {
    const std::string_view text = "123456789";
    const auto bytes = std::as_bytes(std::span<const char>(text));
    EXPECT_EQ(crc32(0, bytes), 0xCBF43926U);
    EXPECT_EQ(crc32(crc32(0, bytes.first(4)), bytes.subspan(4)), 0xCBF43926U);
}

TEST(Ids, AddWithIdsAndMetadata) {
    Index index(2, Metric::L2);
    EXPECT_EQ(index.add("a", std::vector<float>{1, 2}, R"({"x": 1})"), 0U);
    EXPECT_EQ(index.add("b", std::vector<float>{3, 4}), 1U);
    EXPECT_TRUE(index.has_ids());
    EXPECT_TRUE(index.contains("a"));
    EXPECT_FALSE(index.contains("c"));
    EXPECT_EQ(index.id(1), "b");
    EXPECT_EQ(index.meta(0), R"({"x": 1})");
    EXPECT_EQ(index.meta(1), "");
}

TEST(Ids, RejectsDuplicatesAndMixing) {
    Index with_ids(2, Metric::L2);
    with_ids.add("a", std::vector<float>{1, 2});
    EXPECT_THROW(with_ids.add("a", std::vector<float>{3, 4}), std::invalid_argument);
    EXPECT_THROW(with_ids.add(std::vector<float>{3, 4}), std::invalid_argument);
    EXPECT_THROW(with_ids.add("b", std::vector<float>{1}), std::invalid_argument);  // bad vector
    EXPECT_FALSE(with_ids.contains("b"));  // a failed add leaves nothing behind
    EXPECT_EQ(with_ids.size(), 1U);

    Index without_ids(2, Metric::L2);
    without_ids.add(std::vector<float>{1, 2});
    EXPECT_THROW(without_ids.add("a", std::vector<float>{3, 4}), std::invalid_argument);
    EXPECT_EQ(without_ids.id(0), "0");  // the row number
    EXPECT_EQ(without_ids.meta(0), "");
}

TEST(SaveLoad, RoundTripWithIds) {
    const Index index = build(500, 16, Metric::Cosine, true);
    const auto path = temp_file("with_ids.vkt");
    index.save(path);
    EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
    const Index loaded = Index::load(path);
    expect_same(index, loaded);
    EXPECT_TRUE(loaded.contains("doc-123"));

    std::mt19937_64 rng(5);
    for (int i = 0; i < 20; ++i) {
        const std::vector<float> query = random_vector(rng, 16);
        const auto a = index.search(query, 10, 32);
        const auto b = loaded.search(query, 10, 32);
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t j = 0; j < a.size(); ++j) {
            EXPECT_EQ(a[j].row, b[j].row);
            EXPECT_EQ(a[j].distance, b[j].distance);
        }
    }
}

TEST(SaveLoad, RoundTripWithoutIds) {
    const Index index = build(300, 8, Metric::L2, false);
    const auto path = temp_file("without_ids.vkt");
    index.save(path);
    expect_same(index, Index::load(path));
}

TEST(SaveLoad, EmptyIndex) {
    const Index index(5, Metric::L2, {.M = 4, .M_max0 = 8});
    const auto path = temp_file("empty.vkt");
    index.save(path);
    const Index loaded = Index::load(path);
    EXPECT_EQ(loaded.size(), 0U);
    EXPECT_EQ(loaded.dim(), 5U);
    EXPECT_TRUE(loaded.search(std::vector<float>(5, 1.0F), 3, 10).empty());
}

TEST(SaveLoad, CanAddAfterLoading) {
    const auto path = temp_file("grow.vkt");
    build(200, 8, Metric::L2, true).save(path);
    Index loaded = Index::load(path);
    std::mt19937_64 rng(99);
    std::vector<std::vector<float>> added;
    for (int i = 0; i < 50; ++i) {
        added.push_back(random_vector(rng, 8));
        loaded.add("new-" + std::to_string(i), added.back());
    }
    EXPECT_EQ(loaded.size(), 250U);
    for (std::size_t i = 0; i < added.size(); ++i) {
        const auto got = loaded.search(added[i], 1, 50);
        ASSERT_EQ(got.size(), 1U);
        EXPECT_EQ(loaded.id(got[0].row), "new-" + std::to_string(i));
    }
}

TEST(SaveLoad, OverwritesExistingFile) {
    const auto path = temp_file("overwrite.vkt");
    build(50, 4, Metric::L2, false).save(path);
    build(80, 4, Metric::L2, false).save(path);
    EXPECT_EQ(Index::load(path).size(), 80U);
}

class CorruptFile : public testing::Test {
protected:
    void SetUp() override { build(100, 8, Metric::L2, true).save(path_); }

    void flip_byte(std::uintmax_t offset) const {
        std::fstream file(path_, std::ios::binary | std::ios::in | std::ios::out);
        file.seekg(static_cast<std::streamoff>(offset));
        char c = 0;
        file.read(&c, 1);
        c = static_cast<char>(c ^ 0x5A);
        file.seekp(static_cast<std::streamoff>(offset));
        file.write(&c, 1);
    }

    // Message of the exception thrown by load, or "" if it did not throw.
    [[nodiscard]] std::string load_error() const {
        try {
            (void)Index::load(path_);
        } catch (const std::runtime_error& e) {
            return e.what();
        }
        return "";
    }

    std::filesystem::path path_ = temp_file("corrupt.vkt");
};

TEST_F(CorruptFile, FlippedByteInVectors) {
    flip_byte(200);  // inside the vector data
    EXPECT_NE(load_error().find("checksum"), std::string::npos) << load_error();
}

TEST_F(CorruptFile, FlippedByteNearEnd) {
    flip_byte(std::filesystem::file_size(path_) - 10);  // inside the IDs and metadata
    EXPECT_FALSE(load_error().empty());
}

TEST_F(CorruptFile, Truncated) {
    std::filesystem::resize_file(path_, std::filesystem::file_size(path_) - 7);
    EXPECT_FALSE(load_error().empty());
}

TEST_F(CorruptFile, ExtraBytes) {
    std::ofstream(path_, std::ios::binary | std::ios::app) << "extra";
    EXPECT_FALSE(load_error().empty());
}

TEST_F(CorruptFile, WrongMagic) {
    flip_byte(0);
    EXPECT_NE(load_error().find("not a Vektor index"), std::string::npos) << load_error();
}

TEST_F(CorruptFile, WrongVersion) {
    flip_byte(4);
    EXPECT_NE(load_error().find("unsupported file version"), std::string::npos) << load_error();
}

TEST(SaveLoad, MissingFile) {
    EXPECT_THROW((void)Index::load(temp_file("does_not_exist.vkt")), std::runtime_error);
}

}  // namespace
}  // namespace vektor
