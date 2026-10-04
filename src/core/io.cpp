// Saving and loading the index as a binary .vkt file. Layout (little-endian):
//
//   [magic "VKT1"][u32 version][u32 dim][u8 metric]
//   [u64 M][u64 M_max0][u64 ef_construction][u64 seed][u8 heuristic]
//   [u64 count][f32 x count x dim: the vectors]
//   per node: [u8 level], then per layer 0..level: [u16 n][u32 x n: links]
//   [u32 entry_point][i32 max_level]
//   [u8 has_ids], then if 1, per node: [u32 length][ID bytes][u32 length][metadata bytes]
//   [u32 CRC-32 of everything before it]

#include <array>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/binary.hpp"
#include "core/index.hpp"

namespace vektor {
namespace {

constexpr std::array<char, 4> kMagic = {'V', 'K', 'T', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kMaxDim = 1U << 16U;  // sanity limit, far above real embeddings
constexpr int kMaxLevel = 64;                 // real levels stay below 20

std::runtime_error file_error(const std::filesystem::path& path, const std::string& what) {
    return std::runtime_error(path.string() + ": " + what);
}

// Writes values and keeps a running checksum of everything written.
class Writer {
public:
    explicit Writer(const std::filesystem::path& path)
        : out_(path, std::ios::binary | std::ios::trunc) {}

    [[nodiscard]] bool ok() const { return static_cast<bool>(out_); }

    template <typename T>
    void array(std::span<const T> values) {
        crc_ = crc32(crc_, std::as_bytes(values));
        write_raw(out_, values);
    }

    template <typename T>
    void value(T v) {
        array(std::span<const T>(&v, 1));
    }

    void text(const std::string& s) {
        value(static_cast<std::uint32_t>(s.size()));
        array(std::span<const char>(s));
    }

    // Appends the checksum and closes the file.
    void finish() {
        write_raw(out_, std::span<const std::uint32_t>(&crc_, 1));
        out_.close();
    }

private:
    std::ofstream out_;
    std::uint32_t crc_ = 0;
};

// Reads values, keeps a running checksum, and refuses to read past the end of
// the file, so a corrupted length can never trigger a huge allocation.
class Reader {
public:
    explicit Reader(const std::filesystem::path& path) : path_(path), in_(path, std::ios::binary) {
        if (!in_) {
            throw file_error(path, "cannot open");
        }
        remaining_ = std::filesystem::file_size(path);
    }

    // Throws unless at least `bytes` more bytes are left in the file.
    void need(std::uint64_t bytes) const {
        if (bytes > remaining_) {
            throw error("file is cut short or corrupted");
        }
    }

    template <typename T>
    void array(std::span<T> values) {
        need(values.size_bytes());
        read_raw(in_, values);
        if (!in_) {
            throw error("read failed");
        }
        remaining_ -= values.size_bytes();
        crc_ = crc32(crc_, std::as_bytes(values));
    }

    template <typename T>
    T value() {
        T v{};
        array(std::span<T>(&v, 1));
        return v;
    }

    std::string text() {
        const auto length = value<std::uint32_t>();
        need(length);
        std::string s(length, '\0');
        array(std::span<char>(s));
        return s;
    }

    // Reads the stored checksum and compares it with the one computed while reading.
    void check_checksum() {
        const std::uint32_t computed = crc_;
        const auto stored = value<std::uint32_t>();
        if (stored != computed) {
            throw error("checksum mismatch: the file is corrupted");
        }
        if (remaining_ != 0) {
            throw error("unexpected data after the checksum");
        }
    }

    [[nodiscard]] std::runtime_error error(const std::string& what) const {
        return file_error(path_, what);
    }

private:
    std::filesystem::path path_;
    std::ifstream in_;
    std::uint64_t remaining_ = 0;
    std::uint32_t crc_ = 0;
};

}  // namespace

void Index::save(const std::filesystem::path& path) const {
    std::filesystem::path tmp = path;
    tmp += ".tmp";
    Writer w(tmp);
    if (!w.ok()) {
        throw file_error(tmp, "cannot open for writing");
    }
    w.array(std::span<const char>(kMagic));
    w.value(kVersion);
    w.value(static_cast<std::uint32_t>(dim_));
    w.value(static_cast<std::uint8_t>(metric_));
    w.value(static_cast<std::uint64_t>(p_.M));
    w.value(static_cast<std::uint64_t>(p_.M_max0));
    w.value(static_cast<std::uint64_t>(p_.ef_construction));
    w.value(p_.seed);
    w.value(static_cast<std::uint8_t>(p_.heuristic ? 1 : 0));

    w.value(static_cast<std::uint64_t>(size()));
    w.array(std::span<const float>(vectors_.data(), size() * dim_));
    for (const auto& layers : links_) {
        w.value(static_cast<std::uint8_t>(layers.size() - 1));
        for (const auto& list : layers) {
            w.value(static_cast<std::uint16_t>(list.size()));
            w.array(std::span<const std::uint32_t>(list));
        }
    }
    w.value(entry_point_);
    w.value(static_cast<std::int32_t>(max_level_));

    w.value(static_cast<std::uint8_t>(has_ids() ? 1 : 0));
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        w.text(ids_[i]);
        w.text(meta_[i]);
    }
    w.finish();
    if (!w.ok()) {
        throw file_error(tmp, "write failed");
    }
    std::filesystem::rename(tmp, path);
}

namespace {

using Links = std::vector<std::vector<std::vector<std::uint32_t>>>;  // links[node][layer]

// Reads the header and returns an empty index with its settings.
Index read_header(Reader& r) {
    std::array<char, 4> magic{};
    r.array(std::span<char>(magic));
    if (magic != kMagic) {
        throw r.error("not a Vektor index file (wrong magic number)");
    }
    if (const auto version = r.value<std::uint32_t>(); version != kVersion) {
        throw r.error("unsupported file version " + std::to_string(version) + " (expected " +
                      std::to_string(kVersion) + ")");
    }
    const auto dim = r.value<std::uint32_t>();
    const auto metric = r.value<std::uint8_t>();
    HnswParams params;
    params.M = r.value<std::uint64_t>();
    params.M_max0 = r.value<std::uint64_t>();
    params.ef_construction = r.value<std::uint64_t>();
    params.seed = r.value<std::uint64_t>();
    params.heuristic = r.value<std::uint8_t>() != 0;
    if (dim == 0 || dim > kMaxDim || metric > static_cast<std::uint8_t>(Metric::Cosine)) {
        throw r.error("header has invalid values");
    }
    try {
        return {dim, static_cast<Metric>(metric), params};
    } catch (const std::invalid_argument& e) {
        throw r.error(std::string("header has invalid values: ") + e.what());
    }
}

// Reads every node's links, checking each list against its length limit.
void read_links(Reader& r, Links& links, const HnswParams& p) {
    for (auto& layers : links) {
        const int level = r.value<std::uint8_t>();
        if (level > kMaxLevel) {
            throw r.error("node level out of range");
        }
        layers.resize(static_cast<std::size_t>(level) + 1);
        for (std::size_t layer = 0; layer < layers.size(); ++layer) {
            const std::size_t limit = layer == 0 ? p.M_max0 : p.M;
            const auto n = r.value<std::uint16_t>();
            if (n > limit) {
                throw r.error("a node has more links than M allows");
            }
            layers[layer].reserve(limit + 1);  // the spare slot, as when adding
            layers[layer].resize(n);
            r.array(std::span<std::uint32_t>(layers[layer]));
        }
    }
}

// Links must point at other nodes that exist on that layer.
void check_links(const Reader& r, const Links& links) {
    for (std::size_t node = 0; node < links.size(); ++node) {
        for (std::size_t layer = 0; layer < links[node].size(); ++layer) {
            for (const std::uint32_t next : links[node][layer]) {
                if (next >= links.size() || next == node || links[next].size() <= layer) {
                    throw r.error("a link points to a missing node");
                }
            }
        }
    }
}

void read_ids(Reader& r, std::size_t count, std::vector<std::string>& ids,
              std::vector<std::string>& meta,
              std::unordered_map<std::string, std::uint32_t>& rows_by_id) {
    ids.reserve(count);
    meta.reserve(count);
    for (std::uint32_t row = 0; row < count; ++row) {
        std::string id = r.text();
        if (!rows_by_id.emplace(id, row).second) {
            throw r.error("duplicate ID '" + id + "'");
        }
        ids.push_back(std::move(id));
        meta.push_back(r.text());
    }
}

}  // namespace

Index Index::load(const std::filesystem::path& path) {
    Reader r(path);
    Index index = read_header(r);
    const auto count = r.value<std::uint64_t>();
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw r.error("vector count out of range");
    }
    r.need(count * index.dim_ * sizeof(float));  // rules out absurd counts before allocating
    index.vectors_.resize(count * index.dim_);
    r.array(std::span<float>(index.vectors_));
    index.links_.resize(count);
    read_links(r, index.links_, index.p_);
    index.entry_point_ = r.value<std::uint32_t>();
    index.max_level_ = r.value<std::int32_t>();
    check_links(r, index.links_);
    const bool valid_entry = count == 0 ? index.max_level_ == -1
                                        : index.entry_point_ < count &&
                                              index.max_level_ == index.level(index.entry_point_);
    if (!valid_entry) {
        throw r.error("invalid entry point");
    }
    if (r.value<std::uint8_t>() != 0) {
        read_ids(r, count, index.ids_, index.meta_, index.rows_by_id_);
    }
    r.check_checksum();
    // New vectors get fresh random levels, not a replay of the first ones.
    index.rng_.seed(index.p_.seed + count);
    return index;
}

}  // namespace vektor
