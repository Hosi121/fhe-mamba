#pragma once

#include "stage1_mamba2_payload.hpp"
#include "rotation_steps.hpp"
#include "public_weights.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace fhemamba::stage1 {

inline constexpr double kPlaintextCoefficientFloor = 1e-8;
inline constexpr int kBabyStepIn = 64;
inline constexpr int kBabyStepOut = 64;

struct ReplicatedShape {
  int replicas = 1;
  int window = 0;
  int reps = 0;         // window / n input tiles per window
  int per_replica = 0;  // ceil(n / r) diagonals (= masks = encodes)
  int baby_step = 1;    // 1 = direct group rotations; >1 = true BSGS
  int guard_windows = 0;  // filled input windows excluded from masks/folding
  bool logarithmic_replication = false;
};

struct RotationSumStep {
  int offset = 0;  // in units of the caller's signed stride
  bool accumulated = false;  // rotate the partial sum, or the original input
};

// Sum count consecutive cyclic shifts. The binary schedule works for every
// count, including non-powers of two, without masks or multiplicative depth.
auto rotation_sum_schedule(int count, bool logarithmic)
    -> std::vector<RotationSumStep>;

template <typename Value, typename Rotate, typename Add>
auto rotation_sum(const Value& input, int count, int stride, bool logarithmic,
                  Rotate&& rotate, Add&& add) -> Value {
  auto sum = input;
  for (const auto& step : rotation_sum_schedule(count, logarithmic)) {
    sum = add(sum, rotate(step.accumulated ? sum : input, step.offset * stride));
  }
  return sum;
}

template <typename Handle>
struct CachedLevelHandle {
  Handle handle{};
  int encode_level = 0;
};

// Resolve a persistent plaintext handle before invoking the mask builder.
// The generic form keeps the hit-first contract testable without FIDESlib.
template <typename Handle, typename Builder>
auto resolve_hit_first_handle(
    const std::vector<CachedLevelHandle<Handle>>* table,
    std::size_t index,
    int consumption_level,
    Builder&& builder,
    bool& cache_hit,
    bool& level_bypass) -> Handle {
  cache_hit = false;
  level_bypass = false;
  if (table != nullptr && index < table->size()) {
    const auto& cached = (*table)[index];
    if (static_cast<bool>(cached.handle) &&
        cached.encode_level <= consumption_level) {
      cache_hit = true;
      return cached.handle;
    }
    level_bypass = static_cast<bool>(cached.handle);
  }
  return std::forward<Builder>(builder)();
}

struct PackingDims {
  int batch = 0;
  int group_count = 0;   // number of full-slot state ciphertexts
  int group_heads = 0;   // heads carried by one state ciphertext
  int group_block = 0;   // group_heads * head_dim
  int xbc0 = 0;          // start of xBC inside proj layout (d_inner)
  int dt0 = 0;           // start of dt inside proj layout (d_inner + conv_dim)
  int b_base = 0;        // start of B inside packed conv layout (d_inner)
  int c_base = 0;        // start of C inside packed conv layout (d_inner + state)
};

struct NormalizedStateLayout {
  // Group maxima are summaries. row_scales are the actual coordinate map.
  std::vector<double> group_scales;
  std::vector<std::vector<double>> row_scales;
  std::vector<std::vector<double>> update_masks;
  std::vector<std::vector<double>> readout_masks;
};


auto resolve_replicated_shape(int output_dim, int input_dim, int batch,
                              int force_r) -> ReplicatedShape;
auto resolve_interleaved_replicated_shape(int output_dim, int input_dim,
                                          int batch, int force_r)
    -> ReplicatedShape;
auto replicated_bsgs_mask(PublicWeightView weights, int output_dim,
                          int input_dim, int k,
                          const ReplicatedShape& shape, int batch_size,
                          double coefficient_floor = kPlaintextCoefficientFloor)
    -> std::vector<double>;
auto replicated_bsgs_pre_mask(PublicWeightView weights,
                              int output_dim, int input_dim, int k,
                              const ReplicatedShape& shape, int batch_size,
                              double coefficient_floor = kPlaintextCoefficientFloor)
    -> std::vector<double>;
auto python_mod(int value, int modulus) -> int;
auto slot_bsgs_giant_with_zero(int input_dim, int output_dim, int baby_step)
    -> std::vector<int>;
auto derive_packing(const M1Payload& payload, int batch) -> PackingDims;
auto build_normalized_state_layout(
    const std::vector<double>& state_group_abs_max, int group_block, int batch)
    -> NormalizedStateLayout;
auto build_row_normalized_state_layout(
    const std::vector<double>& state_row_abs_max, int group_block, int batch)
    -> NormalizedStateLayout;
auto packed_state_max_abs_error(
    const std::vector<double>& packed, const std::vector<double>& reference,
    int token, int group, int heads, int group_heads, int head_dim,
    int state_size, double scale) -> double;
auto packed_state_max_abs_error(
    const std::vector<double>& packed, const std::vector<double>& reference,
    int token, int group, int heads, int group_heads, int head_dim,
    int state_size, const std::vector<double>& row_scales) -> double;

auto packed_head_max_abs_error(
    const std::vector<double>& packed, const std::vector<double>& reference,
    int token, int group, int heads, int group_heads, int head_dim,
    int state_size) -> double;
auto int_log2(int value) -> int;
auto required_rotations(const M1Payload& payload, const PackingDims& dims,
                        const ReplicatedShape& rep_in,
                        const ReplicatedShape& rep_out,
                        bool replicated_state_blocks = false,
                        bool shared_head_expansion = false)
    -> std::vector<int32_t>;
using fhemamba::naf_steps;
void verify_naf(const std::vector<int32_t>& indices);
auto rotation_frequencies(const M1Payload& payload, const PackingDims& dims,
                          int layers, int streams, int stream_stride,
                          const ReplicatedShape& rep_in,
                          const ReplicatedShape& rep_out,
                          bool replicated_state_blocks = false,
                          bool shared_head_expansion = false)
    -> std::map<int32_t, double>;
auto rotation_key_gib_estimate(int ring_dim, int multiplicative_depth)
    -> double;

}  // namespace fhemamba::stage1
