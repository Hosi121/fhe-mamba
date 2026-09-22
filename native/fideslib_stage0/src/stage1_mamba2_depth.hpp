#pragma once

#include "stage1_mamba2_payload.hpp"

#include <set>
#include <string>
#include <vector>

namespace fhemamba::stage1 {

inline constexpr double kChebCoefficientFloor = 1e-12;
inline constexpr int kAssumedBootstrapOutputLevel = 18;
inline constexpr int kNewtonSegmentEstimate = 14;
// Budget the live variance branch as well as the four-level seed. If the
// input is allowed to reach level 35 at depth 44, refreshing only y forces
// every next product back to that exhausted variance branch. This headroom
// refreshes the input before that point (including Meta-BTS preparation).
inline constexpr int kScheduledNormInputRequirement = 18;
// Joint write -> head placement -> *x -> *B -> *C -> readout mask -> gate.
// The old five-level estimate let a level-34 write reach y at level 40;
// Meta-BTS preparation then reached the last modulus at depth 44 and
// corrupted its correction. Shared head extraction adds one more level.
inline constexpr int kJointWriteTailRequirement = 6;

struct DepthEstimate {
  std::vector<int> token_output_levels;
  int required_depth = 0;
  int proj_level = 0;    // fresh-input level of the in_proj output
  int update_level = 0;  // fresh-input level of the token-0 state update
  // Segment requirements for the mid-circuit bootstrap checkpoints (same
  // formulas as LayerPlan; used for the pre-run geometry warning).
  int req_residual = 0;
  int req_proj = 0;
  int req_fifo = 0;
  int req_conv = 0;
  int req_dt = 0;
  int req_decay = 0;
  int req_state_pre = 0;
  int req_state_tail = 0;
  int req_y = 0;
  int req_out = 0;
  int max_segment = 0;
};

struct ChebPSCost {
  int ct_ct_muls = 0;
  int scalar_muls = 0;
  int depth = 0;
  double discarded_l1 = 0.0;
};

struct ChebPSPlan {
  int baby_size = 1;
  int baseline_baby_size = 1;
  ChebPSCost cost;
  ChebPSCost baseline;
};

// Count the actual cached basis/recursive split, including coefficient-floor
// branches. Depth is relative to the normalized input; a fresh constant is
// conservatively charged one level. No ciphertexts or private data are used.
auto cheb_ps_cost(const std::vector<double>& coeffs, int baby_size) -> ChebPSCost;
auto plan_cheb_ps(const std::vector<double>& coeffs) -> ChebPSPlan;

auto ceil_log2(int value) -> int;
auto cheb_baby_size(int degree) -> int;
auto cheb_clenshaw_host(const std::vector<double>& coeffs, double t) -> double;
auto cheb_ps_host(const std::vector<double>& coeffs, double u, int m) -> double;
void verify_cheb_ps_host(const std::string& name,
                         const std::vector<double>& coeffs, int baby_size = 0);
auto cheb_ps_depth(int degree) -> int;
auto estimate_levels(
    const M1Payload& payload, int tokens,
    const std::set<int>& bootstrap_before_token,
    const std::set<int>& debug_client_reencrypt_before_token,
    bool refresh_recurrent_state_post, int state_refresh_interval,
    bool replicated_state_blocks, bool shared_head_expansion,
    int streams) -> DepthEstimate;

}  // namespace fhemamba::stage1
