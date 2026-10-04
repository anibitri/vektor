#pragma once

// Command-line flags given as "--name value" pairs, shared by vektor-bench and
// vektor-server.

#include <charconv>
#include <cstdint>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vektor {

class Flags {
public:
    explicit Flags(std::span<char* const> args) {
        for (std::size_t i = 0; i < args.size(); i += 2) {
            const std::string name = args[i];
            if (!name.starts_with("--") || i + 1 >= args.size()) {
                throw std::invalid_argument("expected '--name value', got '" + name + "'");
            }
            values_[name.substr(2)] = args[i + 1];
        }
    }

    std::string get(const std::string& name) {
        const auto it = values_.find(name);
        if (it == values_.end()) {
            throw std::invalid_argument("missing --" + name);
        }
        std::string value = it->second;
        values_.erase(it);
        return value;
    }

    std::string get(const std::string& name, const std::string& fallback) {
        return values_.contains(name) ? get(name) : fallback;
    }

    std::uint64_t get_count(const std::string& name, std::uint64_t fallback) {
        return values_.contains(name) ? to_count(name, get(name)) : fallback;
    }

    // A comma-separated list, e.g. "--M 8,16,32".
    std::vector<std::string> get_list(const std::string& name, const std::string& fallback) {
        std::vector<std::string> items;
        std::stringstream text(get(name, fallback));
        for (std::string item; std::getline(text, item, ',');) {
            items.push_back(item);
        }
        return items;
    }

    std::vector<std::uint64_t> get_counts(const std::string& name, const std::string& fallback) {
        std::vector<std::uint64_t> counts;
        for (const std::string& item : get_list(name, fallback)) {
            counts.push_back(to_count(name, item));
        }
        return counts;
    }

    // Call after reading every flag, to catch typos.
    void check_all_used() const {
        if (!values_.empty()) {
            throw std::invalid_argument("unknown flag --" + values_.begin()->first);
        }
    }

private:
    static std::uint64_t to_count(const std::string& name, const std::string& text) {
        std::uint64_t value = 0;
        const auto [end, err] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (err != std::errc() || end != text.data() + text.size()) {
            throw std::invalid_argument("--" + name + " needs whole numbers, got '" + text + "'");
        }
        return value;
    }

    std::map<std::string, std::string> values_;
};

}  // namespace vektor
