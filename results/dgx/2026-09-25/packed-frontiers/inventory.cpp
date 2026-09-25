#include "packed_optimization.hpp"
#include <fstream>
#include <iostream>
#include <map>
using namespace fhemamba;
int main(int argc, char** argv) {
  std::ifstream stream(argv[1]);
  auto p = read_packed_program(stream, true);
  auto report = [&](const PackedProgram& x) {
    const auto depth = plan_packed_depth(x);
    int contiguous = 0, gathered = 0, polynomials = 0;
    for (int i = 0; i < static_cast<int>(x.nodes.size()); ++i) if (depth.live[i]) {
      const auto& n = x.nodes[i];
      if (n.operation == "cheb") ++polynomials;
      if (n.operation == "gather") {
        ++gathered; bool yes = true;
        for (int j = 0; j < n.size; ++j) yes &= n.data[j] == n.data[0] + j;
        contiguous += yes;
      }
      if (n.operation == "linear_ref") {
        const auto& owner = x.nodes[n.data[0]];
        if (owner.operation != "linear" || owner.weights().size() != static_cast<std::size_t>(n.size) * x.nodes[n.parents[0]].size)
          throw std::runtime_error("invalid optimized weight reference");
      }
    }
    std::cout << "{\"nodes\":" << x.nodes.size() << ",\"scalar_polynomials\":" << polynomials
      << ",\"gathers\":" << gathered << ",\"contiguous_gathers\":" << contiguous
      << ",\"planned_logical_refreshes\":" << depth.refreshes << "}";
  };
  std::cout << "{\"original\":"; report(p);
  PackedOptimizationStats stats;
  auto opt = batch_packed_polynomials(std::move(p), stats);
  std::cout << ",\"batched\":"; report(opt);
  std::cout << ",\"batches\":" << stats.polynomial_batches << ",\"members\":" << stats.grouped_polynomials
    << ",\"maximum_batch\":" << stats.maximum_batch << "}\n";
}
