#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "core/distance.hpp"
#include "core/index.hpp"

namespace vektor {

// Rows of equal-length vectors stored one after another.
struct Vectors {
    std::uint32_t dim = 0;
    std::vector<float> data;

    [[nodiscard]] std::size_t rows() const { return dim == 0 ? 0 : data.size() / dim; }
    [[nodiscard]] std::span<const float> row(std::size_t i) const {
        return {data.data() + (i * dim), dim};
    }
};

// A benchmark dataset: base vectors to index, query vectors, and the true
// nearest neighbours of each query (row numbers into base, closest first).
struct Dataset {
    Metric metric = Metric::L2;
    Vectors base;
    Vectors queries;
    std::uint32_t gt_k = 0;                   // true neighbours stored per query
    std::vector<std::uint32_t> ground_truth;  // one row of gt_k numbers per query

    [[nodiscard]] std::span<const std::uint32_t> truth(std::size_t query) const {
        return {ground_truth.data() + (query * gt_k), gt_k};
    }
};

// Reads up to max_rows vectors from a .fvecs file (the TEXMEX format used by
// SIFT: each vector is an int32 length followed by that many float32 values).
Vectors read_fvecs(const std::filesystem::path& path, std::size_t max_rows);

// Reads up to max_rows vectors from a word2vec text file (how gensim-data ships
// GloVe): a header line "<rows> <dim>", then per line a word and dim numbers.
Vectors read_word2vec(const std::filesystem::path& path, std::size_t max_rows);

// Reads up to max_rows images from an IDX file (the MNIST and Fashion-MNIST
// format: a big-endian header, then one byte per pixel). One image, one vector.
Vectors read_idx(const std::filesystem::path& path, std::size_t max_rows);

// Shuffles the rows in place with a seeded shuffle that gives the same order
// with every standard library (std::shuffle does not).
void shuffle_rows(Vectors& v, std::uint64_t seed);

// A copy of rows [first, first + n).
Vectors slice_rows(const Vectors& v, std::size_t first, std::size_t n);

// Builds a dataset and finds each query's gt_k true neighbours by brute force.
Dataset make_dataset(Vectors base, Vectors queries, Metric metric, std::uint32_t gt_k);

// .vkd file: [magic "VKD1"][u32 dim][u32 n_base][u32 n_query][u32 gt_k][u32 metric]
//            [f32 base][f32 queries][u32 ground truth], all little-endian.
void save_vkd(const Dataset& ds, const std::filesystem::path& path);
Dataset load_vkd(const std::filesystem::path& path);

// Share of the true top k (the first k of truth) that appear in the first k of found.
double recall_at_k(std::span<const Result> found, std::span<const std::uint32_t> truth,
                   std::size_t k);

}  // namespace vektor
