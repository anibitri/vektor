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

// Synthetic data with a chosen intrinsic dimension r: points z from an
// r-dimensional standard normal distribution, mapped into dim dimensions by
// one fixed random matrix, plus a little noise. Ground truth: top 100, L2.
Dataset make_synthetic(std::uint32_t intrinsic_dim, std::uint32_t dim, std::size_t n_base,
                       std::size_t n_queries, std::uint64_t seed);

// Opens a dataset by name. "synthetic-<r>" is generated in memory (128
// dimensions, 100,000 base vectors, 1,000 queries, seed 1); anything else is
// the path of a .vkd file.
Dataset load_dataset(const std::string& name);

// Two estimates of intrinsic dimension from the nearest neighbours of a random
// sample of points (exact, by brute force, among all the vectors):
//
// TwoNN (Facco et al., 2017, "Estimating the intrinsic dimension of datasets by
// a minimal neighborhood information"): mu = (distance to the 2nd nearest
// neighbour) / (distance to the 1st). If the data is locally d-dimensional,
// P(mu <= x) = 1 - x^-d; d is fitted by a line through the origin of
// -log(1 - F(mu)) against log(mu), leaving out the largest 10% of mu.
//
// MLE (Levina & Bickel, 2004, "Maximum likelihood estimation of intrinsic
// dimension") with k neighbours: per point, (k - 1) / sum over j < k of
// log(T_k / T_j), where T_j is the distance to the j-th neighbour; averaged over
// the points. It looks at a wider neighbourhood than TwoNN's two neighbours.
//
// Points with an exact duplicate (a neighbour at distance 0) are skipped.
//
// With within_sample, neighbours are searched only among the sampled points (as
// in the TwoNN paper and scikit-dimension): fewer points, so farther neighbours,
// so the estimate describes the data at a coarser scale.
struct IntrinsicDimension {
    double twonn = 0;
    double mle = 0;
    std::size_t points = 0;  // sample points without duplicates
};
IntrinsicDimension estimate_intrinsic_dimension(const Vectors& vectors, Metric metric,
                                                std::size_t sample, std::size_t k,
                                                std::uint64_t seed, bool within_sample = false);

// Share of the true top k (the first k of truth) that appear in the first k of found.
double recall_at_k(std::span<const Result> found, std::span<const std::uint32_t> truth,
                   std::size_t k);

}  // namespace vektor
