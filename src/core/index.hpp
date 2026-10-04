#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/distance.hpp"

namespace vektor {

// One search hit: the vector's row number (its position in insertion order)
// and its distance to the query.
struct Result {
    std::uint32_t row;
    float distance;
};

// Settings for the HNSW graph (Malkov & Yashunin, 2018).
struct HnswParams {
    std::size_t M = 16;                 // links per node on the upper layers
    std::size_t M_max0 = 32;            // links per node on layer 0
    std::size_t ef_construction = 200;  // how many candidates to consider when linking a new node
    std::uint64_t seed = 42;            // seeds the random layer of each node
    bool heuristic = true;  // pick links with the paper's heuristic, or just take the M closest
};

// How a search remembers which nodes it has already seen.
enum class Visited : std::uint8_t {
    Tags,     // a reused array of "visit tags" (the default)
    HashSet,  // a new std::unordered_set for every search (kept for comparison)
};

// Checks vec (right length, no NaN or Inf) and returns a copy, scaled to
// length 1 for cosine. Throws std::invalid_argument on bad input.
std::vector<float> prepare_vector(std::span<const float> vec, std::size_t dim, Metric metric);

// Exact top k: compares the query with every row of `rows` (dim floats each).
// The query must already be prepared (see prepare_vector). Closest first.
std::vector<Result> brute_force(std::span<const float> rows, std::size_t dim, Metric metric,
                                std::span<const float> query, std::size_t k);

class Index {
public:
    Index(std::size_t dim, Metric metric, HnswParams params = {});

    // Adds a copy of vec, links it into the HNSW graph and returns its row number.
    // Throws std::invalid_argument if vec has the wrong length, contains NaN or
    // Inf, or is all zeros under cosine. Benchmarks use this: no IDs, the row
    // number identifies the vector.
    std::uint32_t add(std::span<const float> vec);

    // Adds a vector with a string ID and metadata (any text; the server stores
    // JSON). Also throws std::invalid_argument if the ID is already used. An
    // index has IDs for all its vectors or for none, so mixing the two add()
    // calls throws too.
    std::uint32_t add(const std::string& id, std::span<const float> vec, std::string meta = {});

    // Reserves memory for n vectors in total, so adding them does not reallocate.
    void reserve(std::size_t n);

    // Approximate search with HNSW. A larger ef_search is slower but finds more
    // of the true nearest neighbours. Returns up to k results, closest first.
    [[nodiscard]] std::vector<Result> search(std::span<const float> query, std::size_t k,
                                             std::size_t ef_search,
                                             Visited visited = Visited::Tags) const;

    // Exact search: compares the query with every vector.
    [[nodiscard]] std::vector<Result> search_exact(std::span<const float> query,
                                                   std::size_t k) const;

    // Writes the index to a binary .vkt file (via a temporary file, so an
    // existing file is only replaced once the new one is complete).
    void save(const std::filesystem::path& path) const;

    // Reads a .vkt file. Throws std::runtime_error with a clear message if the
    // file is not a Vektor index, has an unknown version, or is corrupted
    // (checked with a CRC-32 checksum).
    static Index load(const std::filesystem::path& path);

    [[nodiscard]] std::size_t size() const { return vectors_.size() / dim_; }
    [[nodiscard]] std::size_t dim() const { return dim_; }
    [[nodiscard]] Metric metric() const { return metric_; }
    [[nodiscard]] const HnswParams& params() const { return p_; }

    // The stored vector (normalised if the metric is cosine).
    [[nodiscard]] std::span<const float> vector(std::uint32_t row) const {
        return {vectors_.data() + (static_cast<std::size_t>(row) * dim_), dim_};
    }

    [[nodiscard]] bool has_ids() const { return !ids_.empty(); }
    [[nodiscard]] bool contains(const std::string& id) const { return rows_by_id_.contains(id); }
    // The vector's ID; its row number as text if the index has no IDs.
    [[nodiscard]] std::string id(std::uint32_t row) const {
        return has_ids() ? ids_[row] : std::to_string(row);
    }
    // The vector's metadata; empty if it has none.
    [[nodiscard]] std::string_view meta(std::uint32_t row) const {
        return has_ids() ? std::string_view(meta_[row]) : std::string_view();
    }

    // Graph inspection, for tests and debugging.
    [[nodiscard]] int level(std::uint32_t node) const {
        return static_cast<int>(links_[node].size()) - 1;
    }
    [[nodiscard]] std::span<const std::uint32_t> neighbours(std::uint32_t node, int layer) const {
        return links_[node][static_cast<std::size_t>(layer)];
    }
    [[nodiscard]] std::size_t max_links(int layer) const { return layer == 0 ? p_.M_max0 : p_.M; }
    [[nodiscard]] std::uint32_t entry_point() const { return entry_point_; }
    [[nodiscard]] int max_level() const { return max_level_; }

    // Bytes used by the vectors, the graph, and the IDs and metadata. An
    // estimate: the ID lookup table and the memory allocator's own bookkeeping
    // are not counted.
    [[nodiscard]] std::size_t memory_bytes() const;

private:
    // A node and its distance to the query, ordered by distance, then node.
    struct Candidate {
        float distance;
        std::uint32_t node;
        auto operator<=>(const Candidate&) const = default;
    };

    std::uint32_t insert(std::span<const float> vec);
    int random_level();
    [[nodiscard]] Candidate greedy_closest(std::span<const float> query, Candidate start,
                                           int layer) const;
    template <class VisitedSet>
    std::vector<Candidate> search_layer(std::span<const float> query,
                                        std::span<const Candidate> entry, std::size_t ef, int layer,
                                        VisitedSet& visited) const;
    [[nodiscard]] std::vector<Candidate> select_neighbours(std::span<const Candidate> candidates,
                                                           std::size_t m) const;
    void link_back(std::uint32_t from, std::uint32_t to, int layer);

    std::size_t dim_;
    Metric metric_;
    HnswParams p_;
    std::vector<float> vectors_;  // row i is vectors_[i * dim_ .. (i + 1) * dim_)
    std::vector<std::vector<std::vector<std::uint32_t>>> links_;  // links_[node][layer]
    std::uint32_t entry_point_ = 0;
    int max_level_ = -1;  // -1 while the index is empty
    std::mt19937_64 rng_;
    std::vector<std::string> ids_;   // empty if the index has no IDs
    std::vector<std::string> meta_;  // one per ID
    std::unordered_map<std::string, std::uint32_t> rows_by_id_;
};

}  // namespace vektor
