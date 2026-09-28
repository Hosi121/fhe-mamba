#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <iterator>
#include <optional>
#include <type_traits>
#include <unordered_map>
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

// Masks often contain long constant runs. Hash the transitions and their
// positions instead of extending a serial hash dependency at every slot.
// The admission rule and the final bitwise equality check remain unchanged.
struct RunMaskFingerprint {
  auto operator()(const std::vector<double>& values) const -> std::optional<uint64_t> {
    if (values.empty()) return std::nullopt;
    uint64_t hash = UINT64_C(14695981039346656037), nonzero = 0;
    uint64_t previous = std::bit_cast<uint64_t>(values.front()) ^ 1;
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto bits = std::bit_cast<uint64_t>(values[i]);
      if (bits == previous) continue;
      previous = bits;
      if ((bits & UINT64_C(0x7fffffffffffffff)) != 0) {
        if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000))
          return std::nullopt;
        if (nonzero && bits != nonzero) return std::nullopt;
        nonzero = bits;
      }
      hash ^= bits + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
      hash ^= i + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    }
    return hash ^ (values.size() * UINT64_C(0x9e3779b97f4a7c15));
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
  explicit MaskPlaintextCache(std::size_t capacity = 64, bool indexed = false)
      : capacity_(capacity), indexed_(indexed) {}
  MaskPlaintextCache(const MaskPlaintextCache&) = delete;
  MaskPlaintextCache& operator=(const MaskPlaintextCache&) = delete;
  MaskPlaintextCache(MaskPlaintextCache&&) = default;
  MaskPlaintextCache& operator=(MaskPlaintextCache&&) = default;

  template <class Encode>
  auto get(const std::vector<double>& values, uint32_t level, Encode&& encode) -> Plaintext {
    const auto hash = [&] {
      if constexpr (std::is_same_v<Fingerprint, MaskFingerprint>)
        if (indexed_) return RunMaskFingerprint{}(values);
      return fingerprint_(values);
    }();
    if (!capacity_ || !hash) { ++bypasses; return encode(); }
    const auto key = index_key(*hash, level);
    auto found = entries_.end();
    auto matches = [&](const auto& entry) {
      return entry.hash == *hash && entry.level == level && entry.values.size() == values.size() &&
          std::memcmp(entry.values.data(), values.data(), values.size() * sizeof(double)) == 0;
    };
    if (indexed_) {
      const auto [begin, end] = index_.equal_range(key);
      for (auto it = begin; it != end; ++it)
        if (matches(*it->second)) { found = it->second; break; }
    } else {
      for (auto it = entries_.begin(); it != entries_.end(); ++it)
        if (matches(*it)) { found = it; break; }
    }
    if (found != entries_.end()) {
      ++hits;
      entries_.splice(entries_.begin(), entries_, found);
      return entries_.front().plaintext;
    }
    ++misses;
    auto plaintext = encode();
    if (entries_.size() == capacity_) {
      if (indexed_) {
        const auto old = std::prev(entries_.end());
        const auto [begin, end] = index_.equal_range(index_key(old->hash, old->level));
        for (auto it = begin; it != end; ++it) if (it->second == old) { index_.erase(it); break; }
      }
      entries_.pop_back(); ++evictions;
    }
    entries_.push_front({*hash, level, values, plaintext});
    if (indexed_) index_.emplace(key, entries_.begin());
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
  bool indexed_;
  Fingerprint fingerprint_;
  std::list<Entry> entries_;
  std::unordered_multimap<uint64_t, typename std::list<Entry>::iterator> index_;
  static uint64_t index_key(uint64_t hash, uint32_t level) {
    return hash ^ (uint64_t(level) * UINT64_C(0x9e3779b97f4a7c15));
  }
};

}  // namespace fhemamba
