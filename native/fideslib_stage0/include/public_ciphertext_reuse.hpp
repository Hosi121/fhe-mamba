#pragma once

#include "packed_program.hpp"

#include <bit>
#include <cstdint>
#include <map>
#include <utility>

namespace fhemamba {

struct PublicCiphertextPlan {
  std::vector<int> group;
  std::vector<std::size_t> uses;
};

// encrypt() appends positive zeros to the context's fixed slot count. Remove
// only those exact bits when identifying equal public vectors. In particular,
// negative zero, coefficient order and every nonzero bit remain significant.
// Input and feedback nodes must never enter this plan.
inline auto plan_public_ciphertexts(const PackedProgram& program,
                                   const std::vector<bool>& live = {}) -> PublicCiphertextPlan {
  if (!live.empty() && live.size() != program.nodes.size())
    throw std::invalid_argument("public ciphertext liveness size mismatch");
  PublicCiphertextPlan plan;
  plan.group.assign(program.nodes.size(), -1);
  std::map<std::vector<uint64_t>, int> groups;
  for (std::size_t i = 0; i < program.nodes.size(); ++i) {
    const auto& node = program.nodes[i];
    if (node.operation != "public" || (!live.empty() && !live[i])) continue;
    if (node.data.size() > static_cast<std::size_t>(program.slots))
      throw std::invalid_argument("public ciphertext exceeds slot count");
    std::vector<uint64_t> key;
    key.reserve(node.data.size());
    for (double value : node.data) key.push_back(std::bit_cast<uint64_t>(value));
    while (!key.empty() && key.back() == 0) key.pop_back();
    const auto [it, inserted] = groups.emplace(std::move(key), plan.uses.size());
    if (inserted) plan.uses.push_back(0);
    plan.group[i] = it->second;
    ++plan.uses[it->second];
  }
  return plan;
}

// Local to one evaluation/public key. Every returned handle owns an independent
// mutable ciphertext. The immutable seed is released after its final source
// node, regardless of scheduling order. No secret input encryption is reused.
template <class Ciphertext>
class PublicCiphertextReuse {
 public:
  explicit PublicCiphertextReuse(PublicCiphertextPlan plan)
      : plan_(std::move(plan)), seeds_(plan_.uses.size()) {}

  template <class Encrypt, class Clone>
  auto get(std::size_t node, Encrypt&& encrypt, Clone&& clone) -> Ciphertext {
    const int group = plan_.group.at(node);
    if (group < 0 || !plan_.uses[group])
      throw std::invalid_argument("public ciphertext source is not pending");
    auto& seed = seeds_[group];
    if (!seed) {
      ++encryptions;
      if (plan_.uses[group] == 1) {
        auto value = encrypt();
        --plan_.uses[group];
        return value;
      }
      seed = encrypt();
      peak_seeds = std::max(peak_seeds, ++live_seeds);
    } else ++reuses;
    auto value = clone(seed);
    if (!--plan_.uses[group]) { seed = {}; --live_seeds; }
    return value;
  }

  std::size_t encryptions = 0, reuses = 0, live_seeds = 0, peak_seeds = 0;

 private:
  PublicCiphertextPlan plan_;
  std::vector<Ciphertext> seeds_;
};

}  // namespace fhemamba
