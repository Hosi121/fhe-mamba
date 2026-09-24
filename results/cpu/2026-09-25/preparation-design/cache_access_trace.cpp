// Static identity trace; no encode, cache implementation or GPU measurement.
#include "packed_depth.hpp"
#include "stage1_mamba2_plan.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1]);
  const auto program = fhemamba::read_packed_program(input, true);
  const auto plan = fhemamba::plan_packed_depth(program);
  std::map<int, std::tuple<int, int, int>> geometries;
  bool first = true;
  std::cout << "{\"slots\":" << program.slots << ",\"linear_calls\":[";
  for (int i = 0; i < static_cast<int>(program.nodes.size()); ++i) {
    if (!plan.live[i]) continue;
    const auto& node = program.nodes[i];
    if (node.operation != "linear" && node.operation != "linear_ref") continue;
    const int identity = node.operation == "linear" ? i : static_cast<int>(node.data[0]);
    const auto& weight = program.nodes[identity];
    const int columns = program.nodes[node.parents[0]].size;
    auto shape = fhemamba::stage1::resolve_interleaved_replicated_shape(
        node.size, columns, program.slots, 0);
    if (shape.replicas <= 1) throw std::runtime_error("unmodeled direct path");
    shape.logarithmic_replication = true;
    shape.baby_step = std::max(2, static_cast<int>(std::sqrt(shape.per_replica)));
    const auto geometry = std::tuple{node.size, columns, shape.per_replica};
    if (geometries.contains(identity) && geometries.at(identity) != geometry)
      throw std::runtime_error("one weight identity used with different geometry");
    geometries[identity] = geometry;
    if (weight.size != node.size) throw std::runtime_error("weight shape differs");
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"node\":" << i << ",\"weight_node\":" << identity
              << ",\"rows\":" << node.size << ",\"columns\":" << columns
              << ",\"diagonals\":" << shape.per_replica
              << ",\"baby_step\":" << shape.baby_step << '}';
  }
  std::cout << "],\"scope\":\"Program-order public-weight identities. Levels and scales are not observed; compatible-encoding hits are an upper bound. No runtime cache or speed measurement.\"}\n";
}
