#include "stage1_mamba2_depth.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fhemamba::stage1 {

auto ceil_log2(int value) -> int {
  if (value <= 0) {
    throw std::invalid_argument("log2 input must be positive");
  }
  int log = 0;
  while ((1 << log) < value) {
    ++log;
  }
  return log;
}

auto cheb_baby_size(int degree) -> int {
  if (degree < 0) {
    throw std::invalid_argument("Chebyshev degree must be non-negative");
  }
  const int levels = std::max(1, ceil_log2(degree + 1));
  return 1 << ((levels + 1) / 2);
}

auto cheb_ps_cost(const std::vector<double>& coeffs, int m) -> ChebPSCost {
  if (coeffs.empty() || m < 1 || (m & (m - 1)) != 0 ||
      !std::all_of(coeffs.begin(), coeffs.end(), [](double c) { return std::isfinite(c); })) {
    throw std::invalid_argument("finite Chebyshev coefficients and power-of-two baby size required");
  }
  ChebPSCost cost;
  if (coeffs.size() == 1) {
    cost.scalar_muls = 1;
    cost.depth = 1;
    return cost;
  }
  // eval_chebyshev always constructs -u, even if no odd basis uses it.
  cost.scalar_muls = 1;
  std::map<int, int> levels{{1, 0}};
  std::function<int(int)> basis = [&](int i) {
    const auto found = levels.find(i);
    if (found != levels.end()) return found->second;
    const int level = i % 2 == 0 ? basis(i / 2) + 1
        : std::max(basis((i + 1) / 2), basis(i / 2)) + 1;
    ++cost.ct_ct_muls;
    // The even double-angle identity adds scaled_clone(ones_ct, -1).
    if (i % 2 == 0) ++cost.scalar_muls;
    levels[i] = level;
    return level;
  };
  std::function<int(std::vector<double>)> rec = [&](std::vector<double> c) {
    const int n = static_cast<int>(c.size()) - 1;
    if (n < m) {
      int deepest = 0;
      bool has_term = false;
      for (int i = 1; i <= n; ++i) {
        if (std::abs(c[i]) < kChebCoefficientFloor) {
          cost.discarded_l1 += std::abs(c[i]);
          continue;
        }
        deepest = std::max(deepest, basis(i) + 1);
        ++cost.scalar_muls;
        has_term = true;
      }
      if (!has_term) {
        ++cost.scalar_muls;
        return 1;
      }
      if (std::abs(c[0]) < kChebCoefficientFloor) cost.discarded_l1 += std::abs(c[0]);
      return deepest;
    }
    int k = m;
    while (2 * k - 1 < n) k *= 2;
    std::vector<double> b(c.begin() + k, c.end());
    for (double& value : b) value *= 2;
    b[0] = c[k];
    std::vector<double> a(c.begin(), c.begin() + k);
    for (int i = k + 1; i <= n; ++i) a[2 * k - i] -= c[i];
    const int giant = basis(k);
    const int right = rec(b);
    ++cost.ct_ct_muls;
    const int left = rec(a);
    return std::max(left, std::max(giant, right) + 1);
  };
  cost.depth = rec(coeffs);
  return cost;
}

auto plan_cheb_ps(const std::vector<double>& coeffs) -> ChebPSPlan {
  if (coeffs.empty()) throw std::invalid_argument("empty Chebyshev coefficients");
  const int degree = static_cast<int>(coeffs.size()) - 1;
  ChebPSPlan plan;
  plan.baseline_baby_size = cheb_baby_size(degree);
  plan.baby_size = plan.baseline_baby_size;
  plan.baseline = plan.cost = cheb_ps_cost(coeffs, plan.baby_size);
  // Preserve the baseline depth/scalar-work ceilings. The fixed 1e-10
  // coefficient-drop budget is a real-arithmetic bound on [-1,1] only;
  // it does not certify domain membership or CKKS evaluation noise.
  for (int m = 1; m <= degree + 1; m *= 2) {
    const auto candidate = cheb_ps_cost(coeffs, m);
    if (candidate.depth > plan.baseline.depth ||
        candidate.scalar_muls > plan.baseline.scalar_muls ||
        candidate.discarded_l1 > std::max(1e-10, plan.baseline.discarded_l1)) continue;
    if (std::tie(candidate.ct_ct_muls, candidate.scalar_muls) <
        std::tie(plan.cost.ct_ct_muls, plan.cost.scalar_muls)) {
      plan.baby_size = m;
      plan.cost = candidate;
    }
  }
  return plan;
}

auto cheb_clenshaw_host(const std::vector<double>& coeffs, double t) -> double {
  if (coeffs.empty()) {
    throw std::invalid_argument("Chebyshev coefficients must not be empty");
  }
  double b1 = 0.0;
  double b2 = 0.0;
  for (std::size_t index = coeffs.size(); index-- > 1;) {
    const double next = 2.0 * t * b1 - b2 + coeffs[index];
    b2 = b1;
    b1 = next;
  }
  return t * b1 - b2 + coeffs[0];
}

auto cheb_ps_host(const std::vector<double>& coeffs, double u, int m) -> double {
  if (coeffs.empty() || m <= 0) {
    throw std::invalid_argument("Chebyshev coefficients and baby size are invalid");
  }
  std::vector<double> t_values(std::max<std::size_t>(coeffs.size() + 1, 2), 0.0);
  t_values[0] = 1.0;
  t_values[1] = u;
  for (std::size_t i = 2; i < t_values.size(); ++i) {
    t_values[i] = 2.0 * u * t_values[i - 1] - t_values[i - 2];
  }
  std::function<double(std::vector<double>)> rec = [&](std::vector<double> c) -> double {
    const int n = static_cast<int>(c.size()) - 1;
    if (n < m) {
      double sum = 0.0;
      for (int i = 0; i <= n; ++i) {
        sum += c[static_cast<std::size_t>(i)] * t_values[static_cast<std::size_t>(i)];
      }
      return sum;
    }
    int k = m;
    while (2 * k - 1 < n) {
      k *= 2;
    }
    std::vector<double> btil(static_cast<std::size_t>(n - k + 1), 0.0);
    for (int j = 0; j <= n - k; ++j) {
      btil[static_cast<std::size_t>(j)] = 2.0 * c[static_cast<std::size_t>(k + j)];
    }
    btil[0] = c[static_cast<std::size_t>(k)];
    std::vector<double> aprime(c.begin(), c.begin() + k);
    for (int i = k + 1; i <= n; ++i) {
      aprime[static_cast<std::size_t>(2 * k - i)] -= c[static_cast<std::size_t>(i)];
    }
    return rec(aprime) + t_values[static_cast<std::size_t>(k)] * rec(btil);
  };
  return rec(coeffs);
}

void verify_cheb_ps_host(const std::string& name, const std::vector<double>& coeffs, int baby_size) {
  const int m = baby_size > 0 ? baby_size : cheb_baby_size(static_cast<int>(coeffs.size()) - 1);
  double max_error = 0.0;
  for (int sample = 0; sample <= 400; ++sample) {
    const double u = -1.0 + 2.0 * sample / 400.0;
    const double reference = cheb_clenshaw_host(coeffs, u);
    const double value = cheb_ps_host(coeffs, u, m);
    max_error = std::max(max_error, std::abs(reference - value));
  }
  if (max_error > 1e-6) {
    throw std::runtime_error("Chebyshev PS self-check failed for " + name + ": max error " +
                             std::to_string(max_error));
  }
}

// Level ledger of the PS recursion (levels above the level of u), used only
// for the pre-run depth estimate.
auto cheb_ps_depth(int degree) -> int {
  if (degree < 0) {
    throw std::invalid_argument("Chebyshev degree must be non-negative");
  }
  const int m = cheb_baby_size(degree);
  std::map<int, int> t_level;
  std::function<int(int)> level_of = [&](int i) -> int {
    if (i <= 1) {
      return 0;
    }
    auto found = t_level.find(i);
    if (found != t_level.end()) {
      return found->second;
    }
    int level = 0;
    if (i % 2 == 0) {
      level = level_of(i / 2) + 1;
    } else {
      level = std::max(level_of((i + 1) / 2), level_of(i / 2)) + 1;
    }
    t_level[i] = level;
    return level;
  };
  std::function<int(int)> rec = [&](int n) -> int {
    if (n == 0) {
      return 0;  // constant term: fresh low-level ciphertext, aligned upward
    }
    if (n < m) {
      int deepest = 0;
      for (int i = 1; i <= n; ++i) {
        deepest = std::max(deepest, level_of(i));
      }
      return deepest + 1;  // scalar coefficient multiply
    }
    int k = m;
    while (2 * k - 1 < n) {
      k *= 2;
    }
    const int giant_term = std::max(level_of(k), rec(n - k)) + 1;
    return std::max(rec(k - 1), giant_term);
  };
  return rec(degree);
}

auto estimate_levels(
    const M1Payload& payload,
    int tokens,
    const std::set<int>& bootstrap_before_token,
    const std::set<int>& debug_client_reencrypt_before_token,
    bool refresh_recurrent_state_post, int state_refresh_interval,
    bool replicated_state_blocks, bool shared_head_expansion,
    int streams) -> DepthEstimate {
  if (tokens <= 0 || state_refresh_interval < 0 || streams <= 0) {
    throw std::invalid_argument("depth estimate tokens and streams must be positive");
  }
  // streams > 1 costs one extra level in each RMS variance reduction (the
  // stream-base mask before the in-stride broadcast).
  const int norm_extra = streams > 1 ? 1 : 0;
  const auto& rms = payload.polys.at("rms_invsqrt");
  const auto& gated = payload.polys.at("gated_rms_invsqrt");
  if ((!rms.normalization && rms.iterations < 1) ||
      (!gated.normalization && gated.iterations < 1)) {
    throw std::invalid_argument("Newton polynomial iterations must be positive");
  }
  const int rms_depth = rms.normalization ? 0 : cheb_ps_depth(static_cast<int>(rms.coeffs.size()) - 1);
  const int conv_depth = cheb_ps_depth(
      static_cast<int>(payload.polys.at("conv_silu").coeffs.size()) - 1);
  const int gate_depth = cheb_ps_depth(
      static_cast<int>(payload.polys.at("gate_silu").coeffs.size()) - 1);
  const int dt_depth = payload.joint_gates
      ? 1 + cheb_ps_depth(static_cast<int>(std::max(payload.joint_gates->p.size(),
                    payload.joint_gates->q.size())) / payload.num_heads - 1)
      : cheb_ps_depth(static_cast<int>(payload.polys.at("dt_softplus").coeffs.size()) - 1);
  const auto& exp_spec = payload.polys.at("decay_exp");
  if (exp_spec.squarings < 0) {
    throw std::invalid_argument("exponential squarings must be non-negative");
  }
  const int exp_depth = cheb_ps_depth(static_cast<int>(exp_spec.coeffs.size()) - 1);

  const int inv1 = rms.normalization
      ? 6 + norm_extra + 2 * static_cast<int>(rms.normalization->coefficients.size())
      : 2 + norm_extra + rms_depth + 2 * rms.iterations;
  const int proj = std::max(inv1, 1) + 1;
  const int xconv = proj + 1 + conv_depth;
  const int gate_lvl = proj + 1 + gate_depth;
  const int dt_lvl = proj + 1 + dt_depth + (payload.joint_gates ? 2 : 1);
  const int decay_lvl = payload.joint_gates ? dt_lvl : dt_lvl + 1 + exp_depth + exp_spec.squarings;
  const int x_exp = xconv + 1;
  const int bc_exp = xconv + (replicated_state_blocks ? 2 : 1);
  const int head_group_extra = shared_head_expansion ? 1 : 0;
  const int dt_exp = dt_lvl + 1 + head_group_extra;
  const int decay_exp_lvl = decay_lvl + 1 + head_group_extra;
  const int dtx = std::max(x_exp, dt_exp) + 1;
  const int update = std::max(dtx, bc_exp) + 1;

  DepthEstimate estimate;
  estimate.proj_level = proj;
  estimate.update_level = update;
  estimate.req_residual = rms.normalization ? kScheduledNormInputRequirement + norm_extra : proj + 1;
  estimate.req_proj =
      1 + std::max(conv_depth, std::max(gate_depth + 2, dt_depth + (payload.joint_gates ? 2 : 1)));
  estimate.req_fifo = 2 + conv_depth;
  estimate.req_conv = replicated_state_blocks ? 7 : 6;
  estimate.req_dt = payload.joint_gates
      ? kJointWriteTailRequirement + head_group_extra : 2 + exp_depth + exp_spec.squarings;
  estimate.req_decay = 3;
  estimate.req_state_pre = 5 + head_group_extra;
  estimate.req_state_tail = 4;
  estimate.req_y = (gated.normalization ? kScheduledNormInputRequirement : 4) + norm_extra;
  estimate.req_out = 2;
  estimate.max_segment = std::max(
      {estimate.req_residual, estimate.req_proj, estimate.req_fifo, estimate.req_conv,
       estimate.req_dt, estimate.req_decay, estimate.req_state_pre,
       estimate.req_state_tail, estimate.req_y, kNewtonSegmentEstimate});
  int state = 0;
  bool has_state = false;
  for (int token = 0; token < tokens; ++token) {
    if (has_state && debug_client_reencrypt_before_token.count(token) > 0) {
      state = 0;
    } else if (has_state && bootstrap_before_token.count(token) > 0) {
      state = std::min(state, kAssumedBootstrapOutputLevel);
    }
    const bool recurrent_update = has_state;
    state = recurrent_update ? std::max(decay_exp_lvl, state) + 1 : update;
    const bool periodic_refresh =
        state_refresh_interval > 0 && token % state_refresh_interval == 0;
    if (recurrent_update &&
        (refresh_recurrent_state_post || periodic_refresh)) {
      state = kAssumedBootstrapOutputLevel;
    }
    has_state = true;
    const int readout = std::max(state, bc_exp) + 2;  // *C then packed mask
    const int y = std::max(readout, gate_lvl) + 1;
    const int variance = y + 1 + norm_extra;
    const int inv2 = gated.normalization
        ? variance + 5 + 2 * static_cast<int>(gated.normalization->coefficients.size())
        : variance + 1 + 2 * (gated.iterations - 1);
    const int out = std::max(y + 1, inv2) + 1;
    estimate.token_output_levels.push_back(out);
    estimate.required_depth = std::max(estimate.required_depth, out);
  }
  return estimate;
}

}  // namespace fhemamba::stage1
