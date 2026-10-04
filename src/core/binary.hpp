#pragma once

// Raw reads and writes and a CRC-32 checksum, shared by Vektor's binary file
// formats (.vkt index files and .vkd benchmark datasets).

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <span>

namespace vektor {

static_assert(std::endian::native == std::endian::little,
              "file formats assume a little-endian CPU");

template <typename T>
void write_raw(std::ostream& out, std::span<const T> values) {
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size_bytes()));
}

template <typename T>
void read_raw(std::istream& in, std::span<T> values) {
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size_bytes()));
}

inline constexpr std::array<std::uint32_t, 256> kCrc32Table = [] {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t i = 0; i < table.size(); ++i) {
        std::uint32_t c = i;
        for (int bit = 0; bit < 8; ++bit) {
            c = (c & 1U) != 0 ? 0xEDB88320U ^ (c >> 1U) : c >> 1U;
        }
        table[i] = c;
    }
    return table;
}();

// CRC-32, the checksum used by zip and PNG. Start with crc = 0; to continue
// over more data, pass the previous result back in.
inline std::uint32_t crc32(std::uint32_t crc, std::span<const std::byte> bytes) {
    crc = ~crc;
    for (const std::byte b : bytes) {
        crc = kCrc32Table[(crc ^ std::to_integer<std::uint32_t>(b)) & 0xFFU] ^ (crc >> 8U);
    }
    return ~crc;
}

}  // namespace vektor
