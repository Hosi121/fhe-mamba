#pragma once

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fhemamba {
// Signed powers of two, in ascending magnitude. Shared by both executors;
// negative digits avoid the long runs of keys in ordinary binary expansion.
inline auto naf_steps(int value) -> std::vector<int> {
  std::vector<int> steps;
  int64_t remaining = value, power = 1;
  while (remaining != 0) {
    if ((remaining & 1) != 0) {
      const int64_t digit = 2 - (remaining & 3);
      const auto step = digit * power;
      if (step < std::numeric_limits<int>::min() || step > std::numeric_limits<int>::max())
        throw std::overflow_error("NAF step exceeds signed rotation index");
      steps.push_back(static_cast<int>(step));
      remaining -= digit;
    }
    remaining /= 2;
    power *= 2;
  }
  return steps;
}

inline int canonical_rotation(int offset, int slots) {
  if (slots < 2 || !std::has_single_bit(static_cast<unsigned>(slots)))
    throw std::invalid_argument("rotation slots must be a power of two");
  offset %= slots;
  if (offset > slots / 2) offset -= slots;
  if (offset < -slots / 2) offset += slots;
  return offset;
}

inline auto rotation_steps(int offset, int slots, bool naf) -> std::vector<int> {
  offset = canonical_rotation(offset, slots);
  if (naf) return naf_steps(offset);
  std::vector<int> steps;
  const int sign = offset < 0 ? -1 : 1;
  for (int bits = std::abs(offset), power = 1; bits; bits >>= 1, power <<= 1)
    if (bits & 1) steps.push_back(sign * power);
  return steps;
}
}  // namespace fhemamba
