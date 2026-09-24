#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace fhemamba {
// Rotate public coefficients right without per-element integer division.
// Both spans are contiguous and nonoverlapping, allowing the standard library
// to use its vectorized copy implementation. No floating-point arithmetic is
// performed, so every coefficient bit (including signed zero) is preserved.
inline auto rotate_plaintext_mask(const std::vector<double>& mask, int offset)
    -> std::vector<double> {
  std::vector<double> out(mask.size());
  if (mask.empty()) return out;
  const auto size = static_cast<std::ptrdiff_t>(mask.size());
  auto shift = static_cast<std::ptrdiff_t>(offset) % size;
  if (shift < 0) shift += size;
  const auto split = mask.end() - shift;
  std::copy(split, mask.end(), out.begin());
  std::copy(mask.begin(), split, out.begin() + shift);
  return out;
}
} // namespace fhemamba
