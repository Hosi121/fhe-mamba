// Static DAG inventory; excludes all rotations inside runtime refreshes.
#include "packed_depth.hpp"
#include "packed_lifetime.hpp"
#include "packed_routing.hpp"
#include "stage1_mamba2_plan.hpp"
#include <fstream>
#include <iostream>
#include <set>

namespace {
long long sum_rotations(int count, int stride, int slots, bool naf) {
  long long result = 0;
  for (const auto& step : fhemamba::stage1::rotation_sum_schedule(count, true))
    result += fhemamba::packed_rotation_cost(step.offset * stride, slots, naf);
  return result;
}
long long gather_rotations(const std::vector<double>& indices, bool scatter, int slots, bool naf, bool bsgs_stages) {
  bool monotone = true, identity = true;
  std::set<int> offsets;
  for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
    monotone = monotone && (!i || indices[i] > indices[i - 1]);
    identity = identity && indices[i] == i;
    offsets.insert((scatter ? -1 : 1) * (static_cast<int>(indices[i]) - i));
  }
  if (scatter && identity) return 0;
  if (monotone && offsets.size() > 32) {
    long long result = 0;
    for (const auto& stage : fhemamba::monotone_routing(indices, scatter)) {
      std::vector<int> offsets;
      for (const auto& [offset, positions] : stage) {
        if (bsgs_stages) offsets.push_back(offset);
        else result += fhemamba::packed_rotation_cost(offset, slots, naf);
      }
      if (bsgs_stages) result += fhemamba::plan_packed_diagonals(offsets, slots, naf).rotations;
    }
    return result;
  }
  return fhemamba::plan_packed_diagonals({offsets.begin(), offsets.end()}, slots, naf).rotations;
}
long long node_rotations(const fhemamba::PackedProgram& program, int index, bool naf, bool bsgs_stages) {
  using namespace fhemamba;
  const auto& node = program.nodes[index];
  const auto& op = node.operation;
  const int slots = program.slots;
  if (op == "gather" || op == "scatter") return gather_rotations(node.data, op == "scatter", slots, naf, bsgs_stages);
  if (op == "repeat") {
    const int outer = node.data[0], inner = node.data[1], repeat = node.data[2];
    std::vector<double> indices(outer * inner);
    for (int group = 0; group < outer; ++group)
      for (int j = 0; j < inner; ++j) indices[group * inner + j] = group * inner * repeat + j;
    return gather_rotations(indices, true, slots, naf, bsgs_stages) + sum_rotations(repeat, -inner, slots, naf);
  }
  if (op == "sum") {
    const int width = node.data[0];
    long long result = 0;
    for (int step = 1; step < width; step *= 2) result += packed_rotation_cost(step, slots, naf);
    std::vector<double> indices(node.size);
    for (int j = 0; j < node.size; ++j) indices[j] = j * width;
    return result + gather_rotations(indices, false, slots, naf, bsgs_stages);
  }
  if (op != "linear" && op != "linear_ref") return 0;
  const auto& weight = op == "linear" ? node : program.nodes[static_cast<int>(node.data[0])];
  const int columns = program.nodes[node.parents[0]].size;
  if (node.size + columns <= slots) {
    auto shape = stage1::resolve_interleaved_replicated_shape(node.size, columns, slots, 0);
    if (shape.replicas > 1) {
      shape.baby_step = std::max(2, static_cast<int>(std::sqrt(shape.per_replica)));
      long long result = sum_rotations(shape.reps, -columns, slots, naf) +
          sum_rotations(shape.replicas + shape.guard_windows, -shape.window, slots, naf) +
          sum_rotations(shape.replicas, shape.window + 1, slots, naf);
      for (int j = 0; j < shape.baby_step; ++j) result += packed_rotation_cost(j * shape.replicas, slots, naf);
      for (int first = 0; first < shape.per_replica; first += shape.baby_step)
        result += packed_rotation_cost(first * shape.replicas, slots, naf);
      return result;
    }
  }
  std::set<int> offsets;
  for (int i = 0; i < node.size; ++i) for (int j = 0; j < columns; ++j)
    if (weight.weights()[i * columns + j] != 0) offsets.insert(j - i);
  return plan_packed_diagonals({offsets.begin(), offsets.end()}, slots, naf).rotations;
}
}

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--bsgs-routing-stages"))
      throw std::invalid_argument("usage: packed_resource_inventory PROGRAM [--bsgs-routing-stages]");
    const bool bsgs_stages = argc == 3;
    std::ifstream input(argv[1]);
    const auto program = fhemamba::read_packed_program(input, true);
    const auto depth = fhemamba::plan_packed_depth(program);
    const auto uses = fhemamba::plan_packed_uses(program, depth.live);
    long long binary = 0, naf = 0, arithmetic = 0, final_arithmetic = 0;
    long long weights = 0, weight_bytes = 0, compact_weights = 0;
    long long peak_handles = 0, peak_handle_slots = 0, live_slots = 0, peak_slots = 0;
    std::set<int> resident;
    for (int i = 0; i < static_cast<int>(program.nodes.size()); ++i) {
      const auto& node = program.nodes[i];
      if (node.operation == "linear") {
        weights += node.weights().size(); compact_weights += node.bf16_weights.size();
        weight_bytes += node.data.size() * sizeof(double) + node.bf16_weights.size() * sizeof(uint16_t);
      }
      if (!depth.live[i]) continue;
      binary += node_rotations(program, i, false, bsgs_stages); naf += node_rotations(program, i, true, bsgs_stages);
      if (node.operation == "add" || node.operation == "mul") {
        ++arithmetic;
        if (uses.last[node.parents[0]] == i || uses.last[node.parents[1]] == i) ++final_arithmetic;
      }
      resident.insert(i); live_slots += node.size;
      if (static_cast<long long>(resident.size()) > peak_handles) {
        peak_handles = resident.size(); peak_handle_slots = live_slots;
      }
      peak_slots = std::max(peak_slots, live_slots);
      for (int parent : node.parents) if (uses.last[parent] == i && resident.erase(parent))
        live_slots -= program.nodes[parent].size;
    }
    std::cout << "{\"schema\":\"fhemamba-packed-resource-inventory-v1\",\"encrypted\":false,"
        << "\"scope\":\"planned refresh, replicated linear, radix8; runtime refresh rotations excluded\","
        << "\"bsgs_routing_stages\":" << (bsgs_stages ? "true" : "false") << ","
        << "\"binary_rotations\":" << binary << ",\"naf_rotations\":" << naf
        << ",\"arithmetic_nodes\":" << arithmetic << ",\"arithmetic_with_final_input\":" << final_arithmetic
        << ",\"public_weights\":" << weights << ",\"double_weight_bytes\":" << weights * sizeof(double)
        << ",\"compact_weight_bytes\":" << weight_bytes << ",\"bf16_weights\":" << compact_weights
        << ",\"peak_live_dag_handles\":" << peak_handles << ",\"slots_at_peak_handles\":" << peak_handle_slots
        << ",\"peak_logical_slots\":" << peak_slots << ",\"ciphertext_slots\":" << program.slots << "}\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
