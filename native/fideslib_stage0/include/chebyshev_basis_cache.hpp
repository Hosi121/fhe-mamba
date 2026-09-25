#pragma once

#include "packed_program.hpp"
#include <map>
#include <memory>
#include <tuple>

namespace fhemamba {
// Only nodes with the same encrypted parent, dimensions and normalization
// interval may share a basis. Coefficients and degree may differ.
struct ChebyshevSharingPlan {
  std::vector<int> group, remaining;
};
inline auto plan_chebyshev_sharing(const PackedProgram& program,
                                 const std::vector<bool>& live) -> ChebyshevSharingPlan {
  using Key = std::tuple<int, int, double, double>;
  std::map<Key, std::vector<int>> groups;
  for (int i = 0; i < static_cast<int>(program.nodes.size()); ++i) {
    const auto& n = program.nodes[i];
    if ((!live.empty() && !live[i]) || n.operation != "cheb") continue;
    groups[{n.parents.at(0), n.size, n.data.at(0), n.data.at(1)}].push_back(i);
  }
  ChebyshevSharingPlan out;
  out.group.assign(program.nodes.size(), -1);
  for (const auto& [key, nodes] : groups) if (nodes.size() > 1) {
    const int group = out.remaining.size();
    out.remaining.push_back(nodes.size());
    for (int i : nodes) out.group[i] = group;
  }
  return out;
}

template <class Ct> struct ChebyshevBasisState {
  std::weak_ptr<typename Ct::element_type> source;
  int level = -1, degree = -1;
  Ct normalized;
  std::map<int, Ct> basis;
  bool matches(const Ct& input) const {
    return source.lock() == input && level == static_cast<int>(input->GetLevel()) &&
           degree == static_cast<int>(input->GetNoiseScaleDeg());
  }
  void reset(const Ct& input) {
    source = input; level = input->GetLevel(); degree = input->GetNoiseScaleDeg();
    normalized.reset(); basis.clear();
  }
};
}  // namespace fhemamba
