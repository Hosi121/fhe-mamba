// Static storage/range inventory only; no encoder or inference change.
#include "packed_depth.hpp"
#include "stage1_mamba2_plan.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1]);
  const auto program = fhemamba::read_packed_program(input, true);
  const auto plan = fhemamba::plan_packed_depth(program);
  std::map<int, int> uses;
  for (int i = 0; i < static_cast<int>(program.nodes.size()); ++i) {
    if (!plan.live[i]) continue;
    const auto& node = program.nodes[i];
    if (node.operation == "linear") ++uses[i];
    else if (node.operation == "linear_ref") ++uses[static_cast<int>(node.data[0])];
  }
  long long unique = 0, total = 0;
  double largest_weight = 0, largest_mean = 0;
  std::cout << std::setprecision(17) << "{\"matrices\":[";
  bool first = true;
  for (const auto& [id, count] : uses) {
    const auto& node = program.nodes[id];
    const int columns = program.nodes[node.parents[0]].size;
    auto shape = fhemamba::stage1::resolve_interleaved_replicated_shape(node.size, columns, program.slots, 0);
    if (shape.replicas <= 1) throw std::runtime_error("unmodeled dense fallback");
    shape.baby_step = std::max(2, static_cast<int>(std::sqrt(shape.per_replica)));
    double max_weight = 0, max_mean = 0;
    for (int k = 0; k < shape.per_replica; ++k) {
      const auto mask = fhemamba::stage1::replicated_bsgs_pre_mask(
          node.weights(), node.size, columns, k, shape, program.slots, 0.0);
      double sum = 0;
      for (const auto value : mask) {
        const auto magnitude = std::abs(value);
        max_weight = std::max(max_weight, magnitude);
        sum = std::nextafter(sum + magnitude, std::numeric_limits<double>::infinity());
      }
      max_mean = std::max(max_mean, std::nextafter(sum / program.slots,
                                                std::numeric_limits<double>::infinity()));
    }
    unique += shape.per_replica;
    total += shape.per_replica * count;
    largest_weight = std::max(largest_weight, max_weight);
    largest_mean = std::max(largest_mean, max_mean);
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"node\":" << id << ",\"rows\":" << node.size << ",\"columns\":" << columns
              << ",\"uses\":" << count << ",\"diagonals\":" << shape.per_replica
              << ",\"max_weight_abs\":" << max_weight << ",\"max_mask_mean_abs_upper\":" << max_mean << '}';
  }
  std::cout << "],\"unique_diagonals\":" << unique << ",\"total_diagonal_encodes\":" << total
            << ",\"slots\":" << program.slots << ",\"max_weight_abs\":" << largest_weight
            << ",\"max_mask_mean_abs_upper\":" << largest_mean
            << ",\"ifft_complex_double_cache_bytes\":" << unique * program.slots * 16
            << ",\"single_level_signed_word_cache_bytes\":" << unique * program.slots * 2 * 8
            << ",\"illustrative_level21_rns_bytes\":" << unique * program.slots * 2 * 8 * 24
            << ",\"scope\":\"Static public masks only; ideal inverse-FFT component bound is mean absolute input. No floating FFT error bound, encoded integer parity, cache implementation, or speed measurement is asserted.\"}\n";
}
