#pragma once

#include "public_weights.hpp"

#include <cmath>
#include <istream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fhemamba {
struct PackedNode {
  std::string operation;
  int size;
  std::vector<int> parents;
  std::vector<double> data;
  double bound = 0;
  std::vector<uint16_t> bf16_weights;
  auto weights() const -> PublicWeightView {
    return bf16_weights.empty() ? PublicWeightView(data) : PublicWeightView(bf16_weights);
  }
};
struct PackedOutput {
  int node;
  std::vector<double> polynomial, exact;
};
struct PackedProgram {
  int slots;
  double bound;
  std::vector<PackedNode> nodes;
  std::vector<PackedOutput> outputs;
};
inline auto read_packed_program(std::istream& stream, bool compact_weights = false) -> PackedProgram {
  PackedProgram p;
  std::string magic;
  int nodes, outputs;
  if (!(stream >> magic >> p.slots >> p.bound >> nodes >> outputs) ||
      (magic != "fhemamba-packed-v1" && magic != "fhemamba-packed-v2") || p.slots < 2 || p.slots > 32768 ||
      (p.slots & (p.slots - 1)) || !std::isfinite(p.bound) || p.bound <= 0 ||
      nodes < 1 || nodes > 100000 || outputs < 1 || outputs > nodes)
    throw std::invalid_argument("invalid packed program header");
  for (int i = 0; i < nodes; ++i) {
    PackedNode n;
    int parents, count;
    if (!(stream >> n.operation >> n.size)) throw std::invalid_argument("invalid packed node");
    n.bound = p.bound;
    if (magic == "fhemamba-packed-v2" &&
        (!(stream >> n.bound) || !std::isfinite(n.bound) || n.bound <= 0 || n.bound > p.bound))
      throw std::invalid_argument("invalid node refresh bound");
    if (!(stream >> parents) || n.size < 1 || n.size > p.slots ||
        parents < 0 || parents > 2) throw std::invalid_argument("invalid packed node");
    for (int j = 0, parent; j < parents; ++j) {
      if (!(stream >> parent) || parent < 0 || parent >= i)
        throw std::invalid_argument("invalid packed dependency");
      n.parents.push_back(parent);
    }
    if (!(stream >> count) || count < 0 || count > p.slots * p.slots)
      throw std::invalid_argument("invalid packed data size");
    for (int j = 0; j < count; ++j) {
      double value;
      if (!(stream >> value) || !std::isfinite(value))
        throw std::invalid_argument("invalid packed constant");
      n.data.push_back(value);
    }
    const auto& op = n.operation;
    const bool source = op == "input" || op == "public";
    const bool binary = op == "add" || op == "mul";
    if (parents != (source ? 0 : binary ? 2 : 1))
      throw std::invalid_argument("wrong operand count");
    const int input_size = source ? 0 : p.nodes[n.parents[0]].size;
    bool valid = false;
    if (source) valid = count == n.size;
    else if (binary) valid = count == 0 && input_size == n.size &&
                            p.nodes[n.parents[1]].size == n.size;
    else if (op == "addp" || op == "mulp") valid = count == n.size && input_size == n.size;
    else if (op == "linear") valid = count == n.size * input_size;
    else if (op == "feedback" && magic == "fhemamba-packed-v2") valid = count == 0 && n.size == input_size;
    else if (op == "linear_ref" && magic == "fhemamba-packed-v2") {
      valid = count == 1 && n.data[0] >= 0 && n.data[0] < i && n.data[0] == std::floor(n.data[0]);
      if (valid) {
        const auto& weight = p.nodes[static_cast<int>(n.data[0])];
        valid = weight.operation == "linear" && weight.size == n.size &&
                weight.weights().size() == static_cast<std::size_t>(n.size) * input_size;
      }
    }
    else if (op == "repeat") {
      valid = count == 3;
      for (double v : n.data) valid = valid && v >= 1 && v <= p.slots && v == std::floor(v);
      if (valid) {
        const int repeat = static_cast<int>(n.data[2]);
        valid = (magic == "fhemamba-packed-v2" || !(repeat & (repeat - 1))) && n.data[0] * n.data[1] == input_size &&
                input_size * repeat == n.size;
      }
    }
    else if (op == "cheb") valid = count >= 4 && count <= 1027 &&
        n.data[0] < n.data[1] && input_size == n.size;
    else if (op == "sum") valid = count == 1 && n.data[0] >= 1 &&
        n.data[0] == std::floor(n.data[0]) && n.data[0] <= p.slots &&
        (static_cast<int>(n.data[0]) & (static_cast<int>(n.data[0]) - 1)) == 0 &&
        n.size * n.data[0] == input_size;
    else if (op == "gather" || op == "scatter") {
      valid = count == (op == "gather" ? n.size : input_size);
      const int limit = op == "gather" ? input_size : n.size;
      std::vector<bool> seen(p.slots);
      for (double index : n.data) {
        if (index < 0 || index >= limit || index != std::floor(index)) valid = false;
        else if (op == "scatter") {
          if (seen[static_cast<int>(index)]) valid = false;
          seen[static_cast<int>(index)] = true;
        }
      }
    }
    if (!valid) throw std::invalid_argument("invalid packed operation: " + op);
    if (compact_weights && op == "linear") compact_exact_bf16(n.data, n.bf16_weights);
    p.nodes.push_back(std::move(n));
  }
  for (int i = 0; i < outputs; ++i) {
    PackedOutput out;
    int size;
    if (!(stream >> out.node >> size) || out.node < 0 || out.node >= nodes ||
        size != p.nodes[out.node].size) throw std::invalid_argument("invalid packed output");
    for (auto* values : {&out.polynomial, &out.exact}) {
      for (int j = 0; j < size; ++j) {
        double value;
        if (!(stream >> value) || !std::isfinite(value))
          throw std::invalid_argument("invalid packed reference");
        values->push_back(value);
      }
    }
    p.outputs.push_back(std::move(out));
  }
  if (stream >> magic) throw std::invalid_argument("trailing packed program data");
  return p;
}
}  // namespace fhemamba
