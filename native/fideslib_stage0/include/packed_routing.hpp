#pragma once

#include "plaintext_mask.hpp"
#include "rotation_steps.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace fhemamba {

inline int packed_rotation_cost(int offset, int slots, bool naf = false) {
  return rotation_steps(offset, slots, naf).size();
}

struct PackedDiagonalPlan {
  std::vector<int> offsets;
  int baby_step = 0;
  int rotations = 0;
};

// Strided head gathers/repeats have arithmetic-progression diagonals. Reuse
// baby rotations, then rotate partial sums. Choose by actual power-of-two
// rotation count, including both traversal directions and wrapped offsets.
inline auto plan_packed_diagonals(std::vector<int> offsets, int slots, bool naf = false) -> PackedDiagonalPlan {
  PackedDiagonalPlan best{offsets};
  for (int offset : offsets) best.rotations += packed_rotation_cost(offset, slots, naf);
  if (offsets.size() < 3) return best;
  const int stride = offsets[1] - offsets[0];
  for (int i = 2; i < static_cast<int>(offsets.size()); ++i)
    if (offsets[i] - offsets[i - 1] != stride) return best;
  for (int direction = 0; direction < 2; ++direction) {
    const int step = offsets[1] - offsets[0];
    for (int baby = 2; baby < static_cast<int>(offsets.size()); ++baby) {
      int cost = 0;
      for (int j = 0; j < baby; ++j) cost += packed_rotation_cost(offsets[0] + j * step, slots, naf);
      for (int first = 0; first < static_cast<int>(offsets.size()); first += baby)
        cost += packed_rotation_cost(first * step, slots, naf);
      if (cost < best.rotations) best = {offsets, baby, cost};
    }
    std::reverse(offsets.begin(), offsets.end());
  }
  return best;
}

inline auto packed_diagonal_pre_mask(const std::vector<double>& mask, int giant)
    -> std::vector<double> {
  return rotate_plaintext_mask(mask, giant);
}

// Each stage partitions the live source coordinates by their rotation. All
// masks in one stage consume the same input level, regardless of radix.
using PackedRoutingStage = std::map<int, std::vector<int>>;

// Move a routing stage's source-coordinate selection to destination masks:
// rotate(x * source_mask, offset) == rotate(x, offset) * destination_mask.
// This exposes shared baby/giant rotations without adding a mask or depth.
inline auto packed_routing_masks(const PackedRoutingStage& stage, int slots)
    -> std::map<int, std::vector<double>> {
  if (slots <= 0) throw std::invalid_argument("invalid routing slot count");
  std::map<int, std::vector<double>> masks;
  for (const auto& [offset, positions] : stage) {
    auto& mask = masks[offset]; mask.resize(slots);
    for (int position : positions) {
      const auto destination = static_cast<int64_t>(position) - offset;
      if (position < 0 || position >= slots || destination < 0 || destination >= slots)
        throw std::invalid_argument("routing stage leaves the slot range");
      mask[destination] = 1;
    }
  }
  return masks;
}

inline auto monotone_routing(const std::vector<double>& indices, bool scatter,
                             int bits = 3) -> std::vector<PackedRoutingStage> {
  if (bits < 1 || bits > 4) throw std::invalid_argument("invalid routing radix");
  std::vector<int> positions(indices.size()), distances(indices.size());
  int maximum = 0;
  for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
    if (!std::isfinite(indices[i]) || indices[i] < i || indices[i] >= 32768 || indices[i] != static_cast<int>(indices[i]) ||
        (i && indices[i] <= indices[i - 1]))
      throw std::invalid_argument("routing indices must be strictly increasing");
    positions[i] = scatter ? i : static_cast<int>(indices[i]);
    maximum = std::max(maximum, distances[i] = static_cast<int>(indices[i]) - i);
  }
  std::vector<int> shifts;
  for (int shift = 0; (maximum >> shift) != 0; shift += bits) shifts.push_back(shift);
  if (shifts.empty()) shifts.push_back(0); // Also applies the selection mask.
  if (scatter) std::reverse(shifts.begin(), shifts.end());
  std::vector<PackedRoutingStage> stages;
  for (int shift : shifts) {
    PackedRoutingStage stage;
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      const int distance = ((distances[i] >> shift) & ((1 << bits) - 1)) << shift;
      stage[scatter ? -distance : distance].push_back(positions[i]);
      positions[i] += scatter ? distance : -distance;
    }
    stages.push_back(std::move(stage));
  }
  return stages;
}
}  // namespace fhemamba
