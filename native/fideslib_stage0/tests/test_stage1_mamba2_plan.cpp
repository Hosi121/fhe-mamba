#include "stage1_mamba2_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
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

auto model_dims() -> fhemamba::stage1::M1Payload {
  fhemamba::stage1::M1Payload payload;
  payload.d_model = 768;
  payload.d_inner = 1536;
  payload.num_heads = 24;
  payload.head_dim = 64;
  payload.state_size = 64;
  payload.n_groups = 1;
  payload.conv_dim = 1664;
  payload.proj_dim = 3352;
  return payload;
}

}  // namespace

auto main() -> int {
  using namespace fhemamba::stage1;

  const auto payload = model_dims();
  const auto packing = derive_packing(payload, 32768);
  require(packing.group_heads == 8, "unexpected heads per state group");
  require(packing.group_count == 3, "unexpected state group count");
  require(packing.group_block == 512, "unexpected state group width");

  const auto rep_in =
      resolve_replicated_shape(payload.proj_dim, payload.d_model, 32768, 0);
  const auto rep_out =
      resolve_replicated_shape(payload.d_model, payload.d_inner, 32768, 0);
  require(rep_in.replicas == 7 && rep_in.window == 4608,
          "unexpected in-projection replicated shape");
  require(rep_out.replicas == 10 && rep_out.window == 3072,
          "unexpected out-projection replicated shape");
  const auto interleaved_in =
      resolve_interleaved_replicated_shape(payload.proj_dim, payload.d_model,
                                           32768, 0);
  const auto interleaved_out =
      resolve_interleaved_replicated_shape(payload.d_model, payload.d_inner,
                                           32768, 0);
  require(interleaved_in.replicas == 7 && interleaved_in.window == 3840 &&
              interleaved_in.per_replica == 110 &&
              interleaved_in.guard_windows == 1,
          "unexpected interleaved in-projection shape");
  require(interleaved_out.replicas == 20 && interleaved_out.window == 1536 &&
              interleaved_out.per_replica == 77 &&
              interleaved_out.guard_windows == 1,
          "unexpected interleaved out-projection shape");

  // Compare the executing rotate/add helper with the defining cyclic sum,
  // including odd counts, negative strides, and wraparound in every slot.
  const std::vector<double> original{1, -2, 3, 0, 5, 7, -4, 2, 8, 11, -3};
  const auto roll = [](const std::vector<double>& input, int shift) {
    std::vector<double> result(input.size());
    for (int i = 0; i < static_cast<int>(input.size()); ++i) {
      result[i] = input[python_mod(i + shift, static_cast<int>(input.size()))];
    }
    return result;
  };
  const auto add = [](const std::vector<double>& a, const std::vector<double>& b) {
    auto result = a;
    for (std::size_t i = 0; i < a.size(); ++i) result[i] += b[i];
    return result;
  };
  for (int count = 1; count <= 64; ++count) {
    for (int stride : {-13, -3, 0, 1, 7, 19}) {
      std::vector<double> expected(original.size(), 0.0);
      for (int j = 0; j < count; ++j) expected = add(expected, roll(original, j * stride));
      require(rotation_sum(original, count, stride, true, roll, add) == expected,
              "binary rotation sum differs from cyclic-sum definition");
    }
    require(rotation_sum_schedule(count, true).size() <= static_cast<std::size_t>(count - 1),
            "binary replication uses more rotations than linear replication");
  }
  require_invalid([] { rotation_sum_schedule(0, true); });
  auto log_in = interleaved_in;
  auto log_out = interleaved_out;
  log_in.baby_step = 10;
  log_out.baby_step = 8;
  const auto linear_keys = required_rotations(payload, packing, log_in, log_out, true);
  log_in.logarithmic_replication = log_out.logarithmic_replication = true;
  const auto log_keys = required_rotations(payload, packing, log_in, log_out, true);
  require(log_keys.size() < linear_keys.size(), "binary sums did not reduce key inventory");
  const std::set<int32_t> log_required(log_keys.begin(), log_keys.end());
  for (const auto& [index, frequency] : rotation_frequencies(
           payload, packing, 24, 1, 32768, log_in, log_out, true)) {
    require(log_required.count(index) == 1 && frequency > 0,
            "binary sum frequency contains an unplanned rotation");
  }

  const auto normalized =
      build_normalized_state_layout({0.0, 2.0, 4.0}, 4, 16);
  require(normalized.group_scales ==
              std::vector<double>({1.0e-6, 2.0, 4.0}),
          "unexpected normalized-state scales");
  require(normalized.update_masks[1][4] == 0.5 &&
              normalized.update_masks[1][7] == 0.5 &&
              normalized.update_masks[1][3] == 0.0,
          "normalized-state update mask has the wrong placement");
  require(normalized.readout_masks[2][0] == 4.0 &&
              normalized.readout_masks[2][3] == 4.0 &&
              normalized.readout_masks[2][4] == 0.0,
          "normalized-state readout mask has the wrong placement");
  require_invalid([] { build_normalized_state_layout({}, 4, 16); });
  require_invalid(
      [] { build_normalized_state_layout({1.0, 2.0}, 9, 16); });
  require_invalid([] {
    build_normalized_state_layout({std::numeric_limits<double>::quiet_NaN()},
                                  4, 16);
  });
  const auto row_normalized = build_row_normalized_state_layout(
      {0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0}, 4, 16);
  require(row_normalized.group_scales == std::vector<double>({8.0, 128.0}),
          "row scale summaries must report each group's maximum");
  require(row_normalized.row_scales[0][0] == 1.0e-6,
          "zero calibrated rows must have a finite scale floor");
  for (int group = 0; group < 2; ++group) {
    // Different row scales must survive state expansion and readout reduction.
    // Exercise a carried state, including a zero-decay reset, over four steps.
    std::vector<double> original(16, 0.0), normalized_state(16, 0.0);
    for (const double decay : {0.2, 0.9, 0.0, 1.0}) {
      for (int state = 0; state < 4; ++state) {
        for (int row = 0; row < 4; ++row) {
          const auto slot = state * 4 + row;
          const double update = (row + 1.0) * (state + 2.0);
          original[slot] = decay * original[slot] + update;
          normalized_state[slot] = decay * normalized_state[slot] +
              update * row_normalized.update_masks[group][group * 4 + row];
        }
      }
      for (int row = 0; row < 4; ++row) {
        double expected = 0.0, actual = 0.0;
        for (int state = 0; state < 4; ++state) {
          expected += (state + 1.0) * original[state * 4 + row];
          actual += (state + 1.0) * normalized_state[state * 4 + row];
        }
        actual *= row_normalized.readout_masks[group][row];
        require(std::abs(expected - actual) < 1.0e-10,
                "row normalization changed the recurrent readout");
      }
    }
  }
  require_invalid([] { build_row_normalized_state_layout({1.0, 2.0, 3.0}, 2, 16); });
  require_invalid([] { build_row_normalized_state_layout({1.0, -1.0}, 2, 16); });

  // Plain reference is [token, head, position, state], while one encrypted
  // group is packed as [state, local_head, position]. Pin both the axis map
  // and normalized-state scale restoration used by debug attribution.
  constexpr int kTokens = 2;
  constexpr int kHeads = 4;
  constexpr int kGroupHeads = 2;
  constexpr int kHeadDim = 3;
  constexpr int kState = 2;
  std::vector<double> state_reference(
      kTokens * kHeads * kHeadDim * kState);
  std::iota(state_reference.begin(), state_reference.end(), 1.0);
  std::vector<double> packed(kGroupHeads * kHeadDim * kState);
  constexpr int kToken = 1;
  constexpr int kGroup = 1;
  constexpr double kScale = 4.0;
  for (int state = 0; state < kState; ++state) {
    for (int local_head = 0; local_head < kGroupHeads; ++local_head) {
      const int head = kGroup * kGroupHeads + local_head;
      for (int position = 0; position < kHeadDim; ++position) {
        const auto packed_index = static_cast<std::size_t>(
            state * kGroupHeads * kHeadDim + local_head * kHeadDim + position);
        const auto reference_index = static_cast<std::size_t>(
            (((kToken * kHeads + head) * kHeadDim + position) * kState + state));
        packed[packed_index] = state_reference[reference_index] / kScale;
      }
    }
  }
  require(packed_state_max_abs_error(
              packed, state_reference, kToken, kGroup, kHeads, kGroupHeads,
              kHeadDim, kState, kScale) == 0.0,
          "packed-state comparison has the wrong axis map");
  const std::vector<double> row_scales{1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
  auto row_packed = packed;
  for (std::size_t slot = 0; slot < row_packed.size(); ++slot) {
    row_packed[slot] *= kScale / row_scales[slot % row_scales.size()];
  }
  require(packed_state_max_abs_error(
              row_packed, state_reference, kToken, kGroup, kHeads, kGroupHeads,
              kHeadDim, kState, row_scales) == 0.0,
          "packed-state comparison did not restore per-row scales");
  row_packed[5] += 0.25;
  require(packed_state_max_abs_error(
              row_packed, state_reference, kToken, kGroup, kHeads, kGroupHeads,
              kHeadDim, kState, row_scales) == 8.0,
          "row scale attribution used a group summary instead of the row");
  packed[5] += 0.25;
  require(packed_state_max_abs_error(
              packed, state_reference, kToken, kGroup, kHeads, kGroupHeads,
              kHeadDim, kState, kScale) == 1.0,
          "packed-state comparison did not restore normalized scale");
  require_invalid([&] {
    packed_state_max_abs_error({}, state_reference, kToken, kGroup, kHeads,
                               kGroupHeads, kHeadDim, kState, kScale);
  });

  std::vector<double> head_reference(kTokens * kHeads);
  std::iota(head_reference.begin(), head_reference.end(), 1.0);
  for (int state = 0; state < kState; ++state) {
    for (int local_head = 0; local_head < kGroupHeads; ++local_head) {
      const int head = kGroup * kGroupHeads + local_head;
      for (int position = 0; position < kHeadDim; ++position) {
        const auto packed_index = static_cast<std::size_t>(
            state * kGroupHeads * kHeadDim + local_head * kHeadDim + position);
        packed[packed_index] = head_reference[kToken * kHeads + head];
      }
    }
  }
  require(packed_head_max_abs_error(
              packed, head_reference, kToken, kGroup, kHeads, kGroupHeads,
              kHeadDim, kState) == 0.0,
          "packed-head comparison has the wrong axis map");
  packed[5] += 0.25;
  require(packed_head_max_abs_error(
              packed, head_reference, kToken, kGroup, kHeads, kGroupHeads,
              kHeadDim, kState) == 0.25,
          "packed-head comparison missed an expanded slot");
  require_invalid([&] {
    packed_head_max_abs_error({}, head_reference, kToken, kGroup, kHeads,
                              kGroupHeads, kHeadDim, kState);
  });

  const auto rotations = required_rotations(payload, packing, rep_in, rep_out);
  require(!rotations.empty(), "rotation plan is empty");
  require(std::is_sorted(rotations.begin(), rotations.end()),
          "rotation plan is not deterministic");
  verify_naf(rotations);
  const auto positive_steps = naf_steps(28);
  require(std::accumulate(positive_steps.begin(), positive_steps.end(), 0) == 28,
          "positive NAF decomposition is incorrect");
  const auto negative_steps = naf_steps(-13);
  require(std::accumulate(negative_steps.begin(), negative_steps.end(), 0) == -13,
          "negative NAF decomposition is incorrect");

  const auto frequencies = rotation_frequencies(
      payload, packing, 24, 1, 32768, rep_in, rep_out);
  const std::set<int32_t> required(rotations.begin(), rotations.end());
  for (const auto& [index, frequency] : frequencies) {
    require(required.count(index) == 1, "frequency contains an unplanned rotation");
    require(frequency > 0.0, "rotation frequency is not positive");
  }
  require(rotation_key_gib_estimate(65536, 44) > 0.0,
          "rotation key estimate is not positive");

  auto bsgs_in = rep_in;
  auto bsgs_out = rep_out;
  bsgs_in.baby_step = 10;
  bsgs_out.baby_step = 12;
  const auto bsgs_rotations = required_rotations(payload, packing, bsgs_in, bsgs_out);
  require(bsgs_rotations.size() < rotations.size(),
          "true replicated BSGS did not reduce the rotation-key inventory");
  const auto bsgs_frequencies = rotation_frequencies(
      payload, packing, 24, 1, 32768, bsgs_in, bsgs_out);
  const std::set<int32_t> bsgs_required(bsgs_rotations.begin(), bsgs_rotations.end());
  for (const auto& [index, frequency] : bsgs_frequencies) {
    require(bsgs_required.count(index) == 1,
            "BSGS frequency contains an unplanned rotation");
    require(frequency > 0.0, "BSGS rotation frequency is not positive");
  }
  const auto state_rotations =
      required_rotations(payload, packing, bsgs_in, bsgs_out, true);
  const std::set<int32_t> state_required(state_rotations.begin(),
                                         state_rotations.end());
  require(state_required.count(-(packing.group_block - 1)) == 1,
          "replicated state stride rotation is missing");
  require(state_required.count(-2 * (packing.group_block - 1)) == 1,
          "replicated state doubling rotation is missing");
  const auto state_frequencies = rotation_frequencies(
      payload, packing, 24, 1, 32768, bsgs_in, bsgs_out, true);
  for (const auto& [index, frequency] : state_frequencies) {
    require(state_required.count(index) == 1,
            "replicated-state frequency contains an unplanned rotation");
    require(frequency > 0.0,
            "replicated-state rotation frequency is not positive");
  }
  const auto shared_head_rotations =
      required_rotations(payload, packing, bsgs_in, bsgs_out, true, true);
  const std::set<int32_t> shared_head_required(
      shared_head_rotations.begin(), shared_head_rotations.end());
  require(shared_head_required.count(
              -(payload.head_dim - 1) * (payload.num_heads - 1)) == 1,
          "shared-head final seed rotation is missing");
  const auto shared_head_frequencies = rotation_frequencies(
      payload, packing, 24, 1, 32768, bsgs_in, bsgs_out, true, true);
  for (const auto& [index, frequency] : shared_head_frequencies) {
    require(shared_head_required.count(index) == 1,
            "shared-head frequency contains an unplanned rotation");
    require(frequency > 0.0,
            "shared-head rotation frequency is not positive");
  }

  const auto small_shape = resolve_replicated_shape(4, 4, 32, 0);
  std::vector<double> weights(16);
  std::iota(weights.begin(), weights.end(), 1.0);
  const auto mask = replicated_bsgs_mask(weights, 4, 4, 0, small_shape, 32);
  require(mask.size() == 32, "replicated mask has the wrong size");
  const auto small_bsgs_base = resolve_replicated_shape(4, 16, 128, 0);
  std::vector<double> bsgs_weights(64);
  std::iota(bsgs_weights.begin(), bsgs_weights.end(), 1.0);
  auto small_bsgs_shape = small_bsgs_base;
  small_bsgs_shape.baby_step = 2;
  const auto first_pre_mask =
      replicated_bsgs_pre_mask(bsgs_weights, 4, 16, 0, small_bsgs_shape, 128);
  const auto second_pre_mask =
      replicated_bsgs_pre_mask(bsgs_weights, 4, 16, 2, small_bsgs_shape, 128);
  require(first_pre_mask == replicated_bsgs_mask(bsgs_weights, 4, 16, 0,
                                                  small_bsgs_shape, 128),
          "zero-giant BSGS mask was unexpectedly shifted");
  require(second_pre_mask != replicated_bsgs_mask(bsgs_weights, 4, 16, 2,
                                                   small_bsgs_shape, 128),
          "nonzero-giant BSGS mask was not shifted");
  auto packed_source = bsgs_weights;
  std::vector<uint16_t> packed_weights;
  require(fhemamba::compact_exact_bf16(packed_source, packed_weights), "exact weights did not compact");
  for (int k = 0; k < small_bsgs_shape.per_replica; ++k)
    require(replicated_bsgs_pre_mask(packed_weights, 4, 16, k, small_bsgs_shape, 128) ==
            replicated_bsgs_pre_mask(bsgs_weights, 4, 16, k, small_bsgs_shape, 128),
            "compact weight storage changed a BSGS mask");

  require_invalid([] { resolve_replicated_shape(4, 0, 32, 0); });
  require_invalid([] { python_mod(1, 0); });
  require_invalid([] { int_log2(0); });
  require_invalid([&] {
    replicated_bsgs_mask({}, 4, 4, 0, small_shape, 32);
  });
  require_invalid([&] {
    rotation_frequencies(payload, packing, 0, 1, 32768, rep_in, rep_out);
  });

  using TestHandle = std::shared_ptr<int>;
  std::vector<CachedLevelHandle<TestHandle>> cached_handles = {
      {std::make_shared<int>(7), 4},
      {std::make_shared<int>(8), 4},
      {std::make_shared<int>(9), 4}};
  int builder_calls = 0;
  bool cache_hit = false;
  bool level_bypass = false;
  for (std::size_t index = 0; index < cached_handles.size(); ++index) {
    auto resolved = resolve_hit_first_handle(
        &cached_handles, index, 4,
        [&]() {
          ++builder_calls;
          return std::make_shared<int>(10);
        },
        cache_hit, level_bypass);
    require(cache_hit && !level_bypass &&
                *resolved == 7 + static_cast<int>(index),
            "fully cached projection did not resolve its persistent handle");
  }
  require(builder_calls == 0,
          "fully cached projection invoked the mask builder");
  auto resolved = resolve_hit_first_handle(
      &cached_handles, 0, 3,
      [&]() {
        ++builder_calls;
        return std::make_shared<int>(10);
      },
      cache_hit, level_bypass);
  require(!cache_hit && level_bypass && builder_calls == 1 && *resolved == 10,
          "level-incompatible replicated plaintext did not rebuild lazily");
  resolved = resolve_hit_first_handle(
      &cached_handles, cached_handles.size(), 4,
      [&]() {
        ++builder_calls;
        return std::make_shared<int>(11);
      },
      cache_hit, level_bypass);
  require(!cache_hit && !level_bypass && builder_calls == 2 && *resolved == 11,
          "missing replicated plaintext did not build lazily");
  return 0;
}
