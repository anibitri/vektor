// The HNSW graph: Malkov & Yashunin, "Efficient and robust approximate nearest
// neighbor search using Hierarchical Navigable Small World graphs" (2018).
// docs/hnsw.md maps each algorithm in the paper to the functions here.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "core/index.hpp"

namespace vektor {
namespace {

// Remembers which nodes one search has seen. Clearing an array before every
// search would cost time proportional to the index size, so instead each search
// gets a new tag number, and a node counts as visited if its slot holds the
// current tag.
class VisitTags {
public:
    void start(std::size_t n) {
        if (tags_.size() < n) {
            tags_.resize(n, 0);
        }
        if (++current_ == 0) {  // the counter wrapped around: clear old tags once
            std::ranges::fill(tags_, 0);
            current_ = 1;
        }
    }

    // True the first time a node is seen in this search.
    bool visit(std::uint32_t node) {
        if (tags_[node] == current_) {
            return false;
        }
        tags_[node] = current_;
        return true;
    }

private:
    std::vector<std::uint32_t> tags_;
    std::uint32_t current_ = 0;
};

// The obvious alternative, kept to measure the difference: a new hash set per search.
class HashVisited {
public:
    void start(std::size_t /*n*/) {}
    bool visit(std::uint32_t node) { return seen_.insert(node).second; }

private:
    std::unordered_set<std::uint32_t> seen_;
};

// One array per thread, so searches running at the same time never share it.
thread_local VisitTags visit_tags;

}  // namespace

int Index::random_level() {
    // u in (0, 1], made from the raw random bits so that every standard library
    // gives the same levels for the same seed.
    const double u = 1.0 - (static_cast<double>(rng_() >> 11U) * 0x1.0p-53);
    const double mult = 1.0 / std::log(static_cast<double>(p_.M));
    return static_cast<int>(std::floor(-std::log(u) * mult));
}

// Moves to a closer neighbour until none is closer. This is SEARCH-LAYER with
// ef = 1, written as a plain loop because only one result is needed.
Index::Candidate Index::greedy_closest(std::span<const float> query, Candidate start,
                                       int layer) const {
    Candidate best = start;
    bool moved = true;
    while (moved) {
        moved = false;
        const auto& links = links_[best.node][static_cast<std::size_t>(layer)];
        for (const std::uint32_t next : links) {
            const float d = distance(metric_, query, vector(next));
            if (d < best.distance) {
                best = {.distance = d, .node = next};
                moved = true;
            }
        }
    }
    return best;
}

// SEARCH-LAYER (Algorithm 2): best-first search that keeps the ef closest nodes
// found so far. Returns them closest first.
template <class VisitedSet>
std::vector<Index::Candidate> Index::search_layer(std::span<const float> query,
                                                  std::span<const Candidate> entry, std::size_t ef,
                                                  int layer, VisitedSet& visited) const {
    visited.start(size());
    // Nodes still to explore, closest on top.
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> to_explore;
    // Best ef nodes found so far, farthest on top so it can be replaced.
    std::priority_queue<Candidate> found;
    for (const Candidate& c : entry) {
        if (visited.visit(c.node)) {
            to_explore.push(c);
            found.push(c);
        }
    }
    while (found.size() > ef) {
        found.pop();
    }

    while (!to_explore.empty()) {
        const Candidate current = to_explore.top();
        if (current.distance > found.top().distance) {
            break;  // every node left to explore is farther than our worst result
        }
        to_explore.pop();
        for (const std::uint32_t next : links_[current.node][static_cast<std::size_t>(layer)]) {
            if (!visited.visit(next)) {
                continue;
            }
            const float d = distance(metric_, query, vector(next));
            if (found.size() < ef || d < found.top().distance) {
                to_explore.push({.distance = d, .node = next});
                found.push({.distance = d, .node = next});
                if (found.size() > ef) {
                    found.pop();
                }
            }
        }
    }

    std::vector<Candidate> out(found.size());
    for (auto i = out.size(); i-- > 0;) {
        out[i] = found.top();
        found.pop();
    }
    return out;
}

// Chooses up to m links from candidates (sorted closest first).
// Simple (Algorithm 3): the m closest.
// Heuristic (Algorithm 4, like hnswlib, without the optional extras): keep a
// candidate only if it is closer to the base node than to every link already
// kept. This spreads links in different directions instead of into one cluster.
std::vector<Index::Candidate> Index::select_neighbours(std::span<const Candidate> candidates,
                                                       std::size_t m) const {
    if (!p_.heuristic || candidates.size() <= m) {
        const auto first = candidates.first(std::min(m, candidates.size()));
        return {first.begin(), first.end()};
    }
    std::vector<Candidate> kept;
    kept.reserve(m);
    for (const Candidate& c : candidates) {
        if (kept.size() == m) {
            break;
        }
        const bool diverse = std::ranges::none_of(kept, [&](const Candidate& k) {
            return distance(metric_, vector(c.node), vector(k.node)) < c.distance;
        });
        if (diverse) {
            kept.push_back(c);
        }
    }
    return kept;
}

// Adds the link from -> to. If `from` now has too many links, keeps the best
// ones, chosen the same way as for a new node.
void Index::link_back(std::uint32_t from, std::uint32_t to, int layer) {
    std::vector<std::uint32_t>& links = links_[from][static_cast<std::size_t>(layer)];
    links.push_back(to);
    const std::size_t limit = max_links(layer);
    if (links.size() <= limit) {
        return;
    }
    std::vector<Candidate> candidates;
    candidates.reserve(links.size());
    for (const std::uint32_t n : links) {
        candidates.push_back({.distance = distance(metric_, vector(from), vector(n)), .node = n});
    }
    std::ranges::sort(candidates);
    links.clear();
    for (const Candidate& c : select_neighbours(candidates, limit)) {
        links.push_back(c.node);
    }
}

// INSERT (Algorithm 1).
std::uint32_t Index::add(std::span<const float> vec) {
    if (size() >= std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("index is full");
    }
    const std::vector<float> v = prepare_vector(vec, dim_, metric_);
    const auto node = static_cast<std::uint32_t>(size());
    vectors_.insert(vectors_.end(), v.begin(), v.end());

    const int level = random_level();
    auto& layers = links_.emplace_back(static_cast<std::size_t>(level) + 1);
    for (int layer = 0; layer <= level; ++layer) {
        // One spare slot: a list briefly holds limit + 1 links before it is trimmed.
        layers[static_cast<std::size_t>(layer)].reserve(max_links(layer) + 1);
    }
    if (max_level_ < 0) {  // first node
        entry_point_ = node;
        max_level_ = level;
        return node;
    }

    // Walk down the layers above the new node's level, keeping only the closest node.
    const std::span<const float> q = vector(node);
    Candidate closest = {.distance = distance(metric_, q, vector(entry_point_)),
                         .node = entry_point_};
    for (int layer = max_level_; layer > level; --layer) {
        closest = greedy_closest(q, closest, layer);
    }
    // On each layer the new node lives on: find candidates, choose links, link both ways.
    std::vector<Candidate> entry = {closest};
    for (int layer = std::min(level, max_level_); layer >= 0; --layer) {
        std::vector<Candidate> found =
            search_layer(q, entry, p_.ef_construction, layer, visit_tags);
        for (const Candidate& c : select_neighbours(found, p_.M)) {
            layers[static_cast<std::size_t>(layer)].push_back(c.node);
            link_back(c.node, node, layer);
        }
        entry = std::move(found);
    }
    if (level > max_level_) {
        entry_point_ = node;
        max_level_ = level;
    }
    return node;
}

// K-NN-SEARCH (Algorithm 5).
std::vector<Result> Index::search(std::span<const float> query, std::size_t k,
                                  std::size_t ef_search, Visited visited) const {
    const std::vector<float> q = prepare_vector(query, dim_, metric_);
    if (k == 0 || max_level_ < 0) {
        return {};
    }
    Candidate closest = {.distance = distance(metric_, q, vector(entry_point_)),
                         .node = entry_point_};
    for (int layer = max_level_; layer > 0; --layer) {
        closest = greedy_closest(q, closest, layer);
    }
    const std::size_t ef = std::max(ef_search, k);
    const std::span<const Candidate> entry(&closest, 1);
    std::vector<Candidate> found;
    if (visited == Visited::Tags) {
        found = search_layer(q, entry, ef, 0, visit_tags);
    } else {
        HashVisited fresh;
        found = search_layer(q, entry, ef, 0, fresh);
    }

    std::vector<Result> out;
    out.reserve(std::min(k, found.size()));
    for (const Candidate& c : found) {
        if (out.size() == k) {
            break;
        }
        out.push_back({.row = c.node, .distance = c.distance});
    }
    return out;
}

}  // namespace vektor
