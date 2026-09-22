#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fhemamba::stage1 {

struct JointGateSpec {
  // Degree-major, head-minor coefficients in the ordinary Chebyshev convention.
  std::vector<double> lo, hi, p, q, rates;
};

inline auto joint_coefficient_slots(const std::vector<double>& row, int batch,
                                   int stride, int offset, int streams) -> std::vector<double> {
  if (batch < 1 || stride < 1 || streams < 1 || streams * stride > batch ||
      offset < 0 || offset + static_cast<int>(row.size()) > stride)
    throw std::runtime_error("invalid joint coefficient slot layout");
  std::vector<double> slots(static_cast<std::size_t>(batch), 0.0);
  for (int stream = 0; stream < streams; ++stream)
    std::copy(row.begin(), row.end(), slots.begin() + stream * stride + offset);
  return slots;
}

// A periodic coefficient vector agrees with the sparse vector on every head
// lane. It may only multiply a value supported on those lanes: in particular,
// Chebyshev T0 and the even-degree recurrence must use the head mask, not 1.
inline auto joint_periodic_coefficient_slots(const std::vector<double>& row, int batch,
                                            int stride, int offset, int streams)
    -> std::vector<double> {
  if (row.empty() || batch < 1 || stride < 1 || streams < 1 ||
      streams > batch / stride || offset < 0 || offset > stride ||
      row.size() > static_cast<std::size_t>(stride - offset))
    throw std::runtime_error("invalid periodic joint coefficient layout");
  int period = 1;
  while (period < static_cast<int>(row.size())) period *= 2;
  if (batch % period != 0 || stride % period != 0)
    throw std::runtime_error("joint coefficient period must divide batch and stream stride");
  std::vector<double> slots(static_cast<std::size_t>(period), 0.0);
  for (std::size_t h = 0; h < row.size(); ++h)
    slots[(static_cast<std::size_t>(offset) + h) % period] = row[h];
  return slots;
}

inline void validate_joint_gate(const JointGateSpec& spec, int heads) {
  if (heads < 1 || spec.lo.size() != static_cast<std::size_t>(heads) ||
      spec.hi.size() != spec.lo.size() || spec.rates.size() != spec.lo.size())
    throw std::runtime_error("joint gate requires one domain/rate per head");
  for (const auto* values : {&spec.lo, &spec.hi, &spec.p, &spec.q, &spec.rates})
    if (!std::all_of(values->begin(), values->end(), [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("joint gate contains non-finite values");
  for (int h = 0; h < heads; ++h)
    if (spec.lo[h] >= spec.hi[h] || spec.rates[h] <= 0)
      throw std::runtime_error("invalid joint gate domain/rate");
  for (const auto* coefficients : {&spec.p, &spec.q})
    if (coefficients->empty() || coefficients->size() % heads != 0 ||
        coefficients->size() / heads > 4097)
      throw std::runtime_error("invalid joint gate coefficient matrix shape");
}

// Vector-coefficient Paterson-Stockmeyer decomposition. All heads share the
// encrypted Chebyshev basis while their domains/coefficients remain distinct.
// No coefficient threshold, interpolation, or domain clipping occurs here.
template <typename Value, typename Ops>
auto evaluate_joint_roots(const Value& input, const JointGateSpec& spec, Ops& ops)
    -> std::pair<Value, Value> {
  const int heads = static_cast<int>(spec.lo.size());
  validate_joint_gate(spec, heads);
  const int degree = static_cast<int>(std::max(spec.p.size(), spec.q.size())) / heads - 1;
  int log = 1;
  while ((1 << log) < degree + 1) ++log;
  const int baby = 1 << ((log + 1) / 2);
  std::map<int, Value> basis{{1, input}};
  std::function<Value(int)> get = [&](int n) -> Value {
    if (const auto found = basis.find(n); found != basis.end()) return found->second;
    auto product = ops.multiply(get((n + 1) / 2), get(n / 2));
    auto doubled = ops.add(product, product);
    auto value = n % 2 == 0 ? ops.scalar_add(doubled, -1.0) : ops.subtract(doubled, input);
    basis.emplace(n, value);
    return value;
  };
  std::function<Value(std::vector<double>)> evaluate = [&](std::vector<double> c) -> Value {
    const int n = static_cast<int>(c.size()) / heads - 1;
    if (n < baby) {
      // Sparse coefficient masks lose another log2(slots) bits during the
      // inverse FFT. Normalize a whole public PS block by an exact power of
      // two, then undo it once after accumulation. This keeps tiny nonzero
      // rows representable without changing or thresholding the polynomial.
      double minimum = 1.0;
      for (int i = 0; i <= n; ++i) {
        double maximum = 0.0;
        for (int h = 0; h < heads; ++h) maximum = std::max(maximum, std::abs(c[i * heads + h]));
        if (maximum > 0.0) minimum = std::min(minimum, maximum);
      }
      const int shift = minimum < 1e-8
          ? static_cast<int>(std::ceil(std::log2(1e-8) - std::log2(minimum))) : 0;
      if (shift > 50) throw std::runtime_error("joint PS block exceeds supported coefficient encoding range");
      if (shift) for (auto& value : c) value = std::ldexp(value, shift);
      std::vector<double> row(c.begin(), c.begin() + heads);
      auto result = ops.constant(row);
      for (int i = 1; i <= n; ++i) {
        row.assign(c.begin() + i * heads, c.begin() + (i + 1) * heads);
        if (std::all_of(row.begin(), row.end(), [](double x) { return x == 0; })) continue;
        result = ops.add(result, ops.coefficient_product(get(i), row));
      }
      return shift ? ops.scale(result, std::ldexp(1.0, -shift)) : result;
    }
    int k = baby;
    while (2 * k - 1 < n) k *= 2;
    std::vector<double> right(c.begin() + k * heads, c.end());
    for (std::size_t i = heads; i < right.size(); ++i) right[i] *= 2;
    std::vector<double> left(c.begin(), c.begin() + k * heads);
    for (int i = k + 1; i <= n; ++i)
      for (int h = 0; h < heads; ++h)
        left[(2 * k - i) * heads + h] -= c[i * heads + h];
    return ops.add(evaluate(left), ops.multiply(get(k), evaluate(right)));
  };
  // Sequence explicitly: both evaluations share a mutable basis cache.
  auto p = evaluate(spec.p);
  auto q = evaluate(spec.q);
  return {p, q};
}

}  // namespace fhemamba::stage1
