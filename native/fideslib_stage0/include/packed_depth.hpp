#pragma once

#include "packed_program.hpp"
#include <algorithm>
#include <set>
#include <vector>

namespace fhemamba {

// Levels count consumed primes. -1 represents a fresh degree-one ciphertext;
// every multiplication then produces degree two at the predicted level.
struct PackedDepthPlan {
  std::vector<int> cost;
  std::vector<bool> refresh_after;
  std::vector<bool> defer_linear_mask;
  std::vector<bool> live;
  int refreshes = 0;
  int ceiling = 39;
  int refreshed = 22;
};

inline int packed_routing_depth(const std::vector<double>& indices, bool scatter) {
  bool monotone = true, identity = true;
  std::set<int> offsets;
  int maximum = 0;
  for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
    monotone = monotone && (!i || indices[i] > indices[i - 1]);
    identity = identity && indices[i] == i;
    const int offset = static_cast<int>(indices[i]) - i;
    offsets.insert(offset);
    maximum = std::max(maximum, offset);
  }
  if (scatter && identity) return 0;
  if (!monotone || offsets.size() <= 32) return 1;
  int depth = 0;
  for (; maximum; maximum >>= 3) ++depth;
  return std::max(1, depth);
}

inline bool packed_negation(const PackedNode& node) {
  return node.operation == "mulp" && !node.data.empty() &&
      std::all_of(node.data.begin(), node.data.end(), [](double v) { return v == -1; });
}

inline int packed_node_depth(const PackedNode& node) {
  const auto& op = node.operation;
  if (op == "input" || op == "public" || op == "feedback" || op == "add" || op == "addp") return 0;
  if (packed_negation(node)) return 0;
  if (op == "mul" || op == "mulp") return 1;
  if (op == "cheb") {
    int depth = 0;
    while ((1 << depth) < static_cast<int>(node.data.size()) - 2) ++depth;
    return depth + 2; // affine, polynomial, additive padding correction
  }
  if (op == "linear" || op == "linear_ref") return 2; // BSGS plus output mask
  if (op == "gather" || op == "scatter") return packed_routing_depth(node.data, op == "scatter");
  if (op == "repeat") {
    const int outer = node.data[0], inner = node.data[1], repeat = node.data[2];
    std::vector<double> indices(outer * inner);
    for (int g = 0; g < outer; ++g) for (int j = 0; j < inner; ++j)
      indices[g * inner + j] = g * inner * repeat + j;
    return packed_routing_depth(indices, true);
  }
  if (op == "sum") {
    std::vector<double> indices(node.size);
    for (int j = 0; j < node.size; ++j) indices[j] = j * node.data[0];
    return packed_routing_depth(indices, false);
  }
  throw std::invalid_argument("unsupported packed depth operation: " + op);
}

inline int simulate_packed_depth(const PackedProgram& program, const PackedDepthPlan& plan,
                                std::vector<int>* produced = nullptr) {
  std::vector<int> levels(program.nodes.size(), -1);
  int refreshes = 0;
  for (int i = 0; i < static_cast<int>(program.nodes.size()); ++i) {
    if (!plan.live[i]) continue;
    const auto& node = program.nodes[i];
    int level = -1;
    if (node.operation != "feedback") {
      for (int parent : node.parents) {
        if (levels[parent] + plan.cost[i] > plan.ceiling) {
          levels[parent] = plan.refreshed;
          ++refreshes;
        }
        level = std::max(level, levels[parent]);
      }
      level += plan.cost[i];
    }
    if (plan.refresh_after[i] && level > plan.refreshed) {
      level = plan.refreshed;
      ++refreshes;
    }
    levels[i] = level;
    if (produced) (*produced)[i] = level;
  }
  return refreshes;
}

inline auto plan_packed_depth(const PackedProgram& program, int ceiling = 39, int refreshed = 22) -> PackedDepthPlan {
  if (refreshed < 0 || ceiling <= refreshed)
    throw std::invalid_argument("invalid packed refresh level policy");
  PackedDepthPlan plan;
  plan.ceiling = ceiling;
  plan.refreshed = refreshed;
  const int count = program.nodes.size();
  plan.refresh_after.resize(count);
  plan.defer_linear_mask.resize(count);
  plan.live.resize(count);
  for (const auto& output : program.outputs) plan.live[output.node] = true;
  for (int i = count - 1; i >= 0; --i) if (plan.live[i])
    for (int parent : program.nodes[i].parents) plan.live[parent] = true;
  std::vector<int> uses(count), levels(count);
  for (int i = 0; i < count; ++i) {
    const auto& node = program.nodes[i];
    plan.defer_linear_mask[i] = node.operation == "linear" || node.operation == "linear_ref";
    plan.cost.push_back(packed_node_depth(node));
    if (plan.live[i]) for (int parent : node.parents) ++uses[parent];
  }
  for (const auto& node : program.nodes) for (int parent : node.parents)
    if (node.operation != "gather") plan.defer_linear_mask[parent] = false;
  for (const auto& output : program.outputs) plan.defer_linear_mask[output.node] = false;
  for (int i = 0; i < count; ++i) if (plan.defer_linear_mask[i]) --plan.cost[i];
  plan.refreshes = simulate_packed_depth(program, plan, &levels);
  // Hoist refreshes above branches only when a whole-graph simulation predicts
  // fewer refreshes. The bounded greedy search is deterministic, not optimal.
  for (int pass = 0; pass < 8; ++pass) {
    bool improved = false;
    for (int i = 0; i < count; ++i) {
      if (!plan.live[i] || uses[i] < 2 || plan.refresh_after[i] || levels[i] <= plan.refreshed) continue;
      plan.refresh_after[i] = true;
      const int trial = simulate_packed_depth(program, plan);
      if (trial < plan.refreshes) {
        plan.refreshes = trial;
        simulate_packed_depth(program, plan, &levels);
        improved = true;
      } else plan.refresh_after[i] = false;
    }
    if (!improved) break;
  }
  return plan;
}
} // namespace fhemamba
