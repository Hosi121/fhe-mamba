#include "stage1_normalization.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

using namespace fhemamba::stage1;

namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct TraceValue { double value; int depth = 0; };
struct TraceOps {
  int ct_ct = 0, ct_pt = 0, scalar_adds = 0, ct_adds = 0;
  auto snapshot(TraceValue v) -> TraceValue { return v; }
  auto scale(TraceValue a, double c) -> TraceValue {
    ++ct_pt; return {a.value * c, a.depth};
  }
  auto scalar_add(TraceValue a, double c) -> TraceValue {
    ++scalar_adds; return {a.value + c, a.depth};
  }
  auto multiply(TraceValue a, TraceValue b) -> TraceValue {
    ++ct_ct; return {a.value * b.value, std::max(a.depth, b.depth) + 1};
  }
  auto add(TraceValue a, TraceValue b) -> TraceValue {
    ++ct_adds; return {a.value + b.value, std::max(a.depth, b.depth)};
  }
  void stage(TraceValue) {}
};
}  // namespace

auto main() -> int {
  std::istringstream good("fhemamba-invsqrt-v1 0.01 10 0.3 2 1.5 0.5 1.5 0.5");
  auto schedule = read_normalization_schedule(good);
  require(schedule.coefficients.size() == 2, "schedule parse");
  for (const auto* bad : {"unknown 0.01 10 0.3 1 1.5 0.5", "fhemamba-invsqrt-v1 0 10 0.3 1 1.5 0.5",
                         "fhemamba-invsqrt-v1 1 10 0.3 1 1.5", "fhemamba-invsqrt-v1 1 10 0.3 1 1.5 -0.5",
                         "fhemamba-invsqrt-v1 1 10 0.3 1 1.5 0.5 extra"}) {
    bool rejected = false;
    try { std::istringstream in(bad); read_normalization_schedule(in); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "bad schedule accepted");
  }
  schedule.coefficients.assign(20, {1.5, 0.5});
  ScalarNormalizationOps scalar;
  for (int i = 0; i <= 1000; ++i) {
    const double v = std::exp(std::log(0.01) + i * std::log(1000.) / 1000);
    const auto coupled = evaluate_normalization(v, schedule, scalar, false);
    const auto balanced = evaluate_normalization(v, schedule, scalar, true);
    const auto weighted = evaluate_normalization(v, schedule, scalar, false, true);
    require(std::abs(coupled.output * std::sqrt(v) - 1) < 1e-12, "coupled accuracy");
    require(std::abs(balanced.output * std::sqrt(v) - 1) < 1e-12, "balanced accuracy");
    require(std::abs(balanced.output - coupled.output) < 1e-12, "polynomial parity");
    require(std::abs(weighted.output - coupled.output) < 1e-12, "weighted polynomial parity");
    int checkpoints = 0;
    const auto repaired = evaluate_balanced_normalization(v, schedule, scalar,
        [&](double y, std::size_t stage) {
          ++checkpoints;
          require(y <= normalization_inverse_bound(schedule, stage) * (1 + 1e-14),
                  "public stage bound must cover the inverse iterate");
          return stage == schedule.coefficients.size() ? y * 1.001 : y;
        });
    require(checkpoints == 20, "one public checkpoint per cubic update");
    // Independent Newton error identity: e -> -1.5*e^2 - 0.5*e^3.
    require(std::abs(repaired.output * std::sqrt(v) - (1 - 1.5005e-6)) < 1e-12,
            "final recomputation must repair an error introduced by refresh");
  }
  for (int n : {1, 2, 7, 16}) {
    schedule.coefficients.assign(n, {1.5, 0.5});
    for (bool balanced : {false, true}) {
      TraceOps ops;
      const auto result = evaluate_normalization(TraceValue{1.0}, schedule, ops, balanced);
      require(ops.ct_ct == 3 * n, "ct-ct count");
      require(ops.ct_pt == (balanced ? 2 * n + 4 : n + 4), "ct-pt count");
      require(result.output.depth == (balanced ? 2 * n : (n == 1 ? 3 : 2 * n + 2)), "depth");
    }
    TraceOps ops;
    const auto weighted = evaluate_normalization(TraceValue{1.0}, schedule, ops, false, true);
    require(ops.ct_ct == 3 * n && ops.ct_pt == n + 4, "weighted operation count");
    require(weighted.output.depth == (n == 1 ? 3 : 2 * n + 2), "weighted depth");
  }
}
