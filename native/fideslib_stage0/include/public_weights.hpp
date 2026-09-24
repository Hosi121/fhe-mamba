#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace fhemamba {
// Read-only matrix coefficients, expanded to double only at the point of use.
// The mask builders are shared by Mamba-2 and the architecture-neutral DAG.
class PublicWeightView {
 public:
  PublicWeightView() = default;
  PublicWeightView(const std::vector<double>& values)
      : doubles_(values.data()), size_(values.size()) {}
  PublicWeightView(const std::vector<uint16_t>& values)
      : bf16_(values.data()), size_(values.size()) {}
  auto size() const -> std::size_t { return size_; }
  auto operator[](std::size_t index) const -> double {
    return bf16_ ? std::bit_cast<float>(static_cast<uint32_t>(bf16_[index]) << 16)
                 : doubles_[index];
  }
 private:
  const double* doubles_ = nullptr;
  const uint16_t* bf16_ = nullptr;
  std::size_t size_ = 0;
};

// All-or-nothing, lossless storage selection. No rounding, coefficient floor
// or arithmetic change is allowed; a nonrepresentable matrix keeps doubles.
inline bool compact_exact_bf16(std::vector<double>& values, std::vector<uint16_t>& packed) {
  for (double value : values) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) return false;
    const float f = static_cast<float>(value);
    if (static_cast<double>(f) != value ||
        (std::bit_cast<uint32_t>(f) & 0xffff)) return false;
  }
  std::vector<uint16_t> out;
  out.reserve(values.size());
  for (double value : values) out.push_back(std::bit_cast<uint32_t>(static_cast<float>(value)) >> 16);
  packed = std::move(out);
  std::vector<double>{}.swap(values);
  return true;
}
}  // namespace fhemamba
