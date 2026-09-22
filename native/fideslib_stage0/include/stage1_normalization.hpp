#pragma once

#include <algorithm>
#include <cmath>
#include <istream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fhemamba::stage1 {

struct NormalizationSchedule {
  double lo = 0, hi = 0, seed = 0;
  std::vector<std::pair<double, double>> coefficients;
};

inline auto read_normalization_schedule(std::istream& in) -> NormalizationSchedule {
  NormalizationSchedule result;
  std::string magic;
  int count = 0;
  if (!(in >> magic >> result.lo >> result.hi >> result.seed >> count) ||
      magic != "fhemamba-invsqrt-v1" || count < 1 || count > 64 ||
      !std::isfinite(result.lo) || !std::isfinite(result.hi) ||
      !std::isfinite(result.seed) || !(0 < result.lo && result.lo < result.hi) ||
      !(result.seed > 0)) {
    throw std::invalid_argument("invalid normalization schedule header");
  }
  for (int i = 0; i < count; ++i) {
    double a = 0, b = 0;
    if (!(in >> a >> b) || !std::isfinite(a) || !std::isfinite(b) || a <= 0 || b <= 0) {
      throw std::invalid_argument("invalid normalization coefficient pair");
    }
    result.coefficients.emplace_back(a, b);
  }
  if (in >> magic) throw std::invalid_argument("trailing normalization schedule data");
  return result;
}

// This parser is not a range certifier. The exporter separately certifies
// the frozen coefficient recipe with exact-rational interval arithmetic.
template <class Value>
struct NormalizationResult {
  Value before_final;
  Value output;
};

// For a certified positive/nonexpansive recipe, y_i <= seed*prod(a_j)
// and y_i <= 1/sqrt(lo). The first bound is much tighter in early stages.
// Upward rounding protects the public coefficient computation; the caller
// must separately budget execution noise and refresh headroom.
inline auto normalization_inverse_bound(const NormalizationSchedule& recipe,
                                        std::size_t completed_stages) -> double {
  if (completed_stages < 1 || completed_stages > recipe.coefficients.size())
    throw std::invalid_argument("invalid completed normalization stage count");
  const double infinity = std::numeric_limits<double>::infinity();
  double bound = recipe.seed;
  for (std::size_t j = 0; j < completed_stages; ++j)
    bound = std::nextafter(bound * recipe.coefficients[j].first, infinity);
  return std::min(bound, std::nextafter(1.0 / std::sqrt(recipe.lo), infinity));
}

// Recompute v*y^2 after each checkpoint. Refreshing an independently carried
// residual would break that identity and conceal refresh error in y.
// A checkpoint may change the representation/noise of y, never the original v.
template <class Value, class Operations, class Checkpoint>
auto evaluate_balanced_normalization(const Value& v, const NormalizationSchedule& recipe,
                                    Operations& ops, Checkpoint checkpoint)
    -> NormalizationResult<Value> {
  const auto [a0, b0] = recipe.coefficients.at(0);
  auto u = ops.scale(ops.scale(v, recipe.seed), recipe.seed);
  auto y = ops.scale(ops.scalar_add(ops.scale(u, -b0), a0), recipe.seed);
  ops.stage(y);
  auto before = ops.snapshot(y);
  for (std::size_t i = 1; i <= recipe.coefficients.size(); ++i) {
    y = checkpoint(y, i);
    if (i == recipe.coefficients.size()) before = ops.snapshot(y);
    const auto [a, b] = i == recipe.coefficients.size()
        ? std::pair<double, double>{1.5, 0.5} : recipe.coefficients[i];
    auto square = ops.multiply(y, y);
    auto vy = ops.multiply(ops.scale(v, -b), y);
    y = ops.add(ops.scale(y, a), ops.multiply(vy, square));
    ops.stage(y);
  }
  return {before, y};
}

// Operations are injected so the same polynomial DAG can be tested without
// FIDESlib and lowered to real ciphertext operations without secret-key access.
template <class Value, class Operations>
auto evaluate_normalization(const Value& v, const NormalizationSchedule& recipe,
                            Operations& ops, bool balanced, bool weighted = false)
    -> NormalizationResult<Value> {
  if (balanced && weighted) throw std::invalid_argument("ambiguous normalization mode");
  if (balanced) return evaluate_balanced_normalization(v, recipe, ops,
      [](const Value& y, std::size_t) { return y; });
  auto u = ops.scale(ops.scale(v, recipe.seed), recipe.seed);
  const auto [a0, b0] = recipe.coefficients.at(0);
  // Keep the weighted residual negative. Besides avoiding a negate, this
  // bypasses pinned FIDESlib's reverse scalar subtraction (wrong sign and
  // two extra scalar products in its GPU implementation).
  if (weighted) u = ops.scale(u, -b0);
  auto factor = weighted ? ops.scalar_add(u, a0) : ops.scalar_add(ops.scale(u, -b0), a0);
  auto y = ops.scale(factor, recipe.seed);
  ops.stage(y);
  if (weighted) {
    for (std::size_t i = 1; i < recipe.coefficients.size(); ++i) {
      const auto [a, b] = recipe.coefficients[i];
      const double ratio = b / recipe.coefficients[i - 1].second;
      u = ops.multiply(ops.scale(u, ratio), ops.multiply(factor, factor));
      factor = ops.scalar_add(u, a);
      y = ops.multiply(y, factor);
      ops.stage(y);
    }
  } else {
    for (std::size_t i = 1; i < recipe.coefficients.size(); ++i) {
      u = ops.multiply(u, ops.multiply(factor, factor));
      const auto [a, b] = recipe.coefficients[i];
      factor = ops.scalar_add(ops.scale(u, -b), a);
      y = ops.multiply(y, factor);
      ops.stage(y);
    }
  }
  auto before = ops.snapshot(y);
  if (weighted) {
    auto residual = ops.multiply(ops.multiply(v, y), y);
    auto correction = ops.scalar_add(ops.scale(residual, -0.5), 1.5);
    y = ops.multiply(y, correction);
  } else {
    auto square = ops.multiply(y, y);
    auto residual = ops.multiply(v, square);
    auto correction = ops.scalar_add(ops.scale(residual, -0.5), 1.5);
    y = ops.multiply(y, correction);
  }
  ops.stage(y);
  return {before, y};
}

struct ScalarNormalizationOps {
  auto snapshot(double v) -> double { return v; }
  auto scale(double v, double c) -> double { return v * c; }
  auto scalar_add(double v, double c) -> double { return v + c; }
  auto multiply(double a, double b) -> double { return a * b; }
  auto add(double a, double b) -> double { return a + b; }
  void stage(double) {}
};

}  // namespace fhemamba::stage1
