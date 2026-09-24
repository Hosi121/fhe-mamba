#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <optional>
#include <utility>
#include <vector>

namespace fhemamba {

// Admit selection/scaling masks (zero plus at most one finite nonzero value).
// Dense weight diagonals are usually unique and must not evict reusable masks.
struct MaskFingerprint {
  auto operator()(const std::vector<double>& values) const -> std::optional<uint64_t> {
    if (values.empty()) return std::nullopt;
    uint64_t hash = UINT64_C(14695981039346656037), nonzero = 0;
    for (double value : values) {
      const auto bits = std::bit_cast<uint64_t>(value);
      if ((bits & UINT64_C(0x7fffffffffffffff)) != 0) {
        if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000))
          return std::nullopt;
        if (nonzero && bits != nonzero) return std::nullopt;
        nonzero = bits;
      }
      hash ^= bits + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

// One cache belongs to one fixed context/slot count and stores only degree-1
// multiplication plaintexts. The caller must synchronize GPU use before an
// entry can be evicted. Hashes only filter: coefficient bits and level must
// match exactly, even on collision. Retained backend handles are bounded by
// capacity; this is not an allocator/RSS byte limit.
template <class Plaintext, class Fingerprint = MaskFingerprint>
class MaskPlaintextCache {
 public:
  explicit MaskPlaintextCache(std::size_t capacity = 64) : capacity_(capacity) {}

  template <class Encode>
  auto get(const std::vector<double>& values, uint32_t level, Encode&& encode) -> Plaintext {
    const auto hash = fingerprint_(values);
    if (!capacity_ || !hash) { ++bypasses; return encode(); }
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->hash == *hash && it->level == level && it->values.size() == values.size() &&
          std::memcmp(it->values.data(), values.data(), values.size() * sizeof(double)) == 0) {
        ++hits;
        entries_.splice(entries_.begin(), entries_, it);
        return entries_.front().plaintext;
      }
    }
    ++misses;
    auto plaintext = encode();
    if (entries_.size() == capacity_) { entries_.pop_back(); ++evictions; }
    entries_.push_front({*hash, level, values, plaintext});
    return plaintext;
  }

  auto size() const -> std::size_t { return entries_.size(); }
  auto capacity() const -> std::size_t { return capacity_; }
  std::size_t hits = 0, misses = 0, bypasses = 0, evictions = 0;

 private:
  struct Entry {
    uint64_t hash;
    uint32_t level;
    std::vector<double> values;
    Plaintext plaintext;
  };
  std::size_t capacity_;
  Fingerprint fingerprint_;
  std::list<Entry> entries_;
};

}  // namespace fhemamba
