#pragma once

#include "packed_program.hpp"
#include <utility>

namespace fhemamba {
struct PackedUsePlan {
  std::vector<int> last;
  std::vector<std::vector<int>> consumers;
};

inline auto plan_packed_uses(const PackedProgram& program,
                            const std::vector<bool>& live = {}) -> PackedUsePlan {
  PackedUsePlan plan{std::vector<int>(program.nodes.size()),
                    std::vector<std::vector<int>>(program.nodes.size())};
  for (int i = 0; i < static_cast<int>(program.nodes.size()); ++i)
    if (live.empty() || live[i]) for (int parent : program.nodes[i].parents) {
      plan.last[parent] = i; plan.consumers[parent].push_back(i);
    }
  for (const auto& out : program.outputs) plan.last[out.node] = program.nodes.size();
  return plan;
}

// Last DAG use alone is insufficient: an identity operation or another DAG
// node may still alias this ciphertext. Move only the unique owning handle.
// Otherwise the caller receives a private scratch value, as before.
template <typename Handle, typename Clone>
auto consume_or_clone(Handle& value, bool final_use, Clone&& clone,
                      long long& reused) -> Handle {
  if (final_use && value.use_count() == 1) {
    ++reused;
    return std::move(value);
  }
  return clone(value);
}
}  // namespace fhemamba
