#pragma once

#include "rotation_steps.hpp"
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <vector>

namespace fhemamba {
struct RotationBatchStats {
  long long requests = 0, edges = 0, preparations = 0, sibling_batches = 0;
};

// Preserve each output's original signed-digit path. Only identical prefixes
// and the decomposition of sibling edges are shared. Callbacks own/synchronize
// their outputs; a parent remains alive until all its children have completed.
template <class Value, class Rotate, class RotateSiblings>
auto rotate_prefix_batch(const Value& input, const std::vector<int>& offsets,
                         int slots, bool naf, Rotate rotate,
                         RotateSiblings siblings, RotationBatchStats& stats)
    -> std::vector<Value> {
  struct Node { std::map<int, int> children; std::vector<int> outputs; };
  std::vector<Node> tree(1);
  for (int i = 0; i < static_cast<int>(offsets.size()); ++i) {
    int parent = 0;
    for (int step : rotation_steps(offsets[i], slots, naf)) {
      auto found = tree[parent].children.find(step);
      int child;
      if (found == tree[parent].children.end()) {
        child = tree.size();
        tree[parent].children.emplace(step, child);
        tree.emplace_back();
      } else child = found->second;
      parent = child;
    }
    tree[parent].outputs.push_back(i);
  }
  stats.requests += offsets.size();
  std::vector<Value> results(offsets.size());
  std::function<void(int, const Value&)> visit = [&](int index, const Value& value) {
    const auto& node = tree[index];
    for (int output : node.outputs) results[output] = value;
    if (node.children.empty()) return;
    ++stats.preparations;
    stats.edges += node.children.size();
    if (node.children.size() == 1) {
      const auto [step, child] = *node.children.begin();
      auto next = rotate(value, step);
      visit(child, next);
    } else {
      ++stats.sibling_batches;
      std::vector<int32_t> steps;
      for (const auto& [step, child] : node.children) steps.push_back(step);
      auto next = siblings(value, steps);
      if (next.size() != steps.size()) throw std::runtime_error("rotation batch size mismatch");
      int i = 0;
      for (const auto& [step, child] : node.children) visit(child, next[i++]);
    }
  };
  visit(0, input);
  return results;
}
}  // namespace fhemamba
