#include "stage1_mamba2_depth.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_invalid(const std::function<void()>& operation) {
  try {
    operation();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("expected invalid_argument");
}

auto depth_payload() -> fhemamba::stage1::M1Payload {
  using fhemamba::stage1::PolySpec;
  fhemamba::stage1::M1Payload payload;
  auto polynomial = [](int degree) {
    PolySpec spec;
    spec.coeffs.assign(static_cast<std::size_t>(degree + 1), 0.01);
    return spec;
  };
  payload.polys["conv_silu"] = polynomial(8);
  payload.polys["gate_silu"] = polynomial(8);
  payload.polys["dt_softplus"] = polynomial(8);
  payload.polys["decay_exp"] = polynomial(8);
  payload.polys["decay_exp"].squarings = 4;
  payload.polys["rms_invsqrt"] = polynomial(8);
  payload.polys["rms_invsqrt"].iterations = 4;
  payload.polys["gated_rms_invsqrt"] = polynomial(8);
  payload.polys["gated_rms_invsqrt"].iterations = 4;
  return payload;
}

}  // namespace

auto main() -> int {
  using namespace fhemamba::stage1;

  const std::vector<double> coeffs = {0.5, 0.25, -0.1, 0.05, -0.025};
  const int baby_size = cheb_baby_size(static_cast<int>(coeffs.size()) - 1);
  for (int sample = 0; sample <= 20; ++sample) {
    const double value = -1.0 + sample / 10.0;
    require(std::abs(cheb_clenshaw_host(coeffs, value) -
                     cheb_ps_host(coeffs, value, baby_size)) < 1e-12,
            "Paterson-Stockmeyer evaluation differs from Clenshaw");
  }
  verify_cheb_ps_host("test", coeffs);
  require(cheb_ps_depth(0) == 0, "constant polynomial consumes a level");
  require(cheb_ps_depth(8) > 0, "non-constant polynomial has zero depth");

  // A coefficient-aware split preserves the polynomial, including asymmetric
  // and sparse series, while obeying baseline depth/scalar-work ceilings.
  bool saved_products = false;
  for (const int degree : {1, 7, 24, 31, 47, 64, 96, 192, 384, 768}) {
    for (const bool sparse : {false, true}) {
      std::vector<double> series(degree + 1);
      for (int i = 0; i <= degree; ++i) {
        series[i] = sparse && i > 1 && i % 2 ? 0.0 : std::cos(i * 0.7) / ((i + 1.) * (i + 1.));
      }
      const auto plan = plan_cheb_ps(series);
      require(plan.cost.depth <= plan.baseline.depth, "planner increased depth");
      require(plan.cost.scalar_muls <= plan.baseline.scalar_muls, "planner increased scalar work");
      require(plan.cost.ct_ct_muls <= plan.baseline.ct_ct_muls, "planner increased ct-ct work");
      saved_products |= plan.cost.ct_ct_muls < plan.baseline.ct_ct_muls;
      for (int sample = 0; sample <= 100; ++sample) {
        const double u = -1.0 + sample / 50.0;
        require(std::abs(cheb_ps_host(series, u, plan.baby_size) -
                         cheb_clenshaw_host(series, u)) < 1e-12,
                "planned evaluation changed the polynomial");
      }
    }
  }
  require(saved_products, "planner never reduced multiplication work");
  require(cheb_ps_cost({1., 0., 1.}, 4).ct_ct_muls == 1, "square basis count is wrong");
  require(cheb_ps_cost({1., 0., 1.}, 4).scalar_muls == 3, "basis constant scale omitted");
  require(cheb_ps_cost({1., 1e-14}, 4).discarded_l1 >= 1e-14, "coefficient drop not counted");
  require(plan_cheb_ps({2.}).cost.ct_ct_muls == 0, "constant polynomial requires products");
  require_invalid([] { plan_cheb_ps({}); });
  require_invalid([] { cheb_ps_cost({1.}, 3); });
  require_invalid([] { cheb_ps_cost({NAN}, 2); });

  const auto estimate =
      estimate_levels(depth_payload(), 3, {}, {}, false, 0, false, false, 1);
  require(estimate.token_output_levels.size() == 3,
          "depth estimate omitted token outputs");
  require(estimate.required_depth > 0 && estimate.max_segment > 0,
          "depth estimate is not positive");
  auto scheduled = depth_payload();
  for (const auto* name : {"rms_invsqrt", "gated_rms_invsqrt"}) {
    scheduled.polys[name] = PolySpec{};
    scheduled.polys[name].normalization = NormalizationSchedule{
        0.00001, 100., 0.1, std::vector<std::pair<double, double>>(16, {1.5, 0.5})};
  }
  const auto scheduled_depth = estimate_levels(scheduled, 2, {}, {}, false, 0, false, false, 1);
  require(scheduled_depth.req_residual >= 18 && scheduled_depth.req_y >= 18,
          "scheduled norms must retain depth for the live variance, seed and refresh");
  require(44 - 35 < scheduled_depth.req_residual &&
          44 - 21 >= scheduled_depth.req_residual,
          "a carried level-35 input must refresh; a level-21 input must fit");
  require(scheduled_depth.required_depth > estimate.required_depth,
          "uninterrupted scheduled depth must include all cubic updates");

  auto joint = scheduled;
  joint.num_heads = 1;
  joint.joint_gates = JointGateSpec{{-1.}, {1.}, {0.5, 0.1}, {1., 0.1}, {1.}};
  const auto joint_depth = estimate_levels(joint, 2, {}, {}, false, 0, true, false, 1);
  const auto joint_shared = estimate_levels(joint, 2, {}, {}, false, 0, true, true, 1);
  // Replay the failing L08 route: head placement, two update products,
  // readout product/mask and output gate consume six levels before y refresh.
  const int write_level = 34;
  const int y_level = write_level + 1 + 2 + 2 + 1;
  require(y_level + 4 == 44, "regression no longer reaches the last Meta-BTS modulus");
  require(44 - write_level < joint_depth.req_dt + 1 + 4,
          "level-34 joint write must refresh before its six-level tail");
  require(44 - 21 >= joint_depth.req_dt + 1 + 4,
          "refreshed joint write must fit its tail");
  require(joint_shared.req_dt == joint_depth.req_dt + 1,
          "shared head extraction was omitted from joint write tail");

  require_invalid([] { ceil_log2(0); });
  require_invalid([] { cheb_baby_size(-1); });
  require_invalid([] { cheb_clenshaw_host({}, 0.0); });
  require_invalid([] { cheb_ps_host({1.0}, 0.0, 0); });
  require_invalid([] { cheb_ps_depth(-1); });
  const auto periodic =
      estimate_levels(depth_payload(), 4, {}, {}, false, 2, false, false, 1);
  require(periodic.token_output_levels[2] < estimate_levels(
             depth_payload(), 4, {}, {}, false, 0, false, false, 1).token_output_levels[2],
          "periodic state refresh was omitted from the depth estimate");
  const auto replicated =
      estimate_levels(depth_payload(), 1, {}, {}, false, 0, true, false, 1);
  require(replicated.req_conv == estimate.req_conv + 1,
          "replicated state layout segment depth was not modeled");
  require(replicated.update_level >= estimate.update_level,
          "replicated state layout reduced the modeled update level");
  const auto shared_heads =
      estimate_levels(depth_payload(), 1, {}, {}, false, 0, false, true, 1);
  require(shared_heads.req_state_pre == estimate.req_state_pre + 1,
          "shared head extraction level was omitted from state preflight");
  require(shared_heads.update_level == estimate.update_level + 1,
          "shared head extraction level was omitted from update depth");
  require_invalid([] {
    estimate_levels(depth_payload(), 0, {}, {}, false, 0, false, false, 1);
  });
  require_invalid([] {
    estimate_levels(depth_payload(), 1, {}, {}, false, -1, false, false, 1);
  });
  auto invalid_newton = depth_payload();
  invalid_newton.polys["rms_invsqrt"].iterations = 0;
  require_invalid([&] {
    estimate_levels(invalid_newton, 1, {}, {}, false, 0, false, false, 1);
  });
  return 0;
}
