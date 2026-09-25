#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace fhemamba {

// Each inverse CKKS transform coefficient is bounded by mean(abs(slots)).
// Leave a factor-two margin below q0/2, including FFT and rounding error.
// This also keeps the pinned 64-bit encoder below its approximate-scaling
// branch. The single q0 residue then uniquely determines the signed integer.
inline bool compact_plaintext_fits(const std::vector<double>& values, uint32_t slots,
                                 double scale, uint64_t q0) {
  if (!slots || values.size() > slots || !std::isfinite(scale) || scale <= 0 ||
      q0 < 4 || q0 >= (uint64_t{1} << 61)) return false;
  long double sum = 0;
  for (double value : values) {
    if (!std::isfinite(value)) return false;
    sum += std::abs(static_cast<long double>(value));
  }
  return sum * scale / slots + 1 < static_cast<long double>(q0) / 4;
}

}  // namespace fhemamba
