#pragma once

#include <algorithm>
#include <cmath>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fhemamba::stage1 {

struct VectorRmsFixture {
  int width = 0, cases = 0;
  double epsilon = 0;
  std::vector<double> gamma, inputs;
};

inline auto read_vector_rms_fixture(std::istream& in) -> VectorRmsFixture {
  VectorRmsFixture f;
  std::string magic;
  if (!(in >> magic >> f.width >> f.cases >> f.epsilon) ||
      magic != "fhemamba-vector-rms-v1" || f.width < 1 || f.width > 8192 ||
      f.cases < 1 || f.cases > 1024 || !std::isfinite(f.epsilon) || f.epsilon <= 0) {
    throw std::invalid_argument("invalid vector RMS fixture header");
  }
  f.gamma.resize(f.width);
  f.inputs.resize(static_cast<std::size_t>(f.width) * f.cases);
  for (auto* values : {&f.gamma, &f.inputs}) {
    for (auto& x : *values) {
      if (!(in >> x) || !std::isfinite(x))
        throw std::invalid_argument("invalid vector RMS fixture value");
    }
  }
  if (in >> magic) throw std::invalid_argument("trailing vector RMS fixture data");
  return f;
}

struct VectorRmsLayout {
  int width, padded_width, slots, lanes;
  explicit VectorRmsLayout(int dimension, int batch_slots = 8192)
      : width(dimension), padded_width(1), slots(batch_slots), lanes(0) {
    if (width < 1 || slots < width || (slots & (slots - 1)))
      throw std::invalid_argument("invalid vector RMS layout");
    while (padded_width < width) padded_width *= 2;
    lanes = slots / padded_width;
  }
  auto slot(int feature, int lane) const -> int { return feature * lanes + lane; }
  auto rotations() const -> std::vector<int> {
    std::vector<int> result;
    for (int shift = lanes; shift < slots; shift *= 2) result.push_back(shift);
    return result;
  }
};

// Public diagonal coordinates. Every exact RMS output component divided by
// this scale has magnitude <=1/1.1. The floor bounds the coordinate condition
// number by 64 and keeps zero-gamma/padding coordinates well-defined.
inline auto vector_rms_refresh_scales(const std::vector<double>& gamma) -> std::vector<double> {
  double largest = 0;
  for (double g : gamma) {
    if (!std::isfinite(g)) throw std::invalid_argument("nonfinite RMS gamma");
    largest = std::max(largest, std::abs(g));
  }
  if (gamma.empty()) throw std::invalid_argument("empty RMS gamma");
  std::vector<double> scales(gamma.size(), 1.0);
  if (largest == 0) return scales;
  const double factor = 1.1 * std::sqrt(gamma.size());
  for (std::size_t j = 0; j < gamma.size(); ++j)
    scales[j] = factor * std::max(std::abs(gamma[j]), largest / 64);
  return scales;
}

// Features, including zero padding, occupy strided cyclic groups. Rotations
// preserve each lane, so the reduction needs no masks or broadcast phase.
template <class Value, class Operations>
auto vector_rms_sum(Value squared, const VectorRmsLayout& layout, Operations& ops) -> Value {
  for (int shift : layout.rotations()) squared = ops.add(squared, ops.rotate(squared, shift));
  return squared;
}

}  // namespace fhemamba::stage1
