// Static call inventory. No encryption, parameter changes, or timing claims.
#include "packed_depth.hpp"
#include "packed_routing.hpp"
#include "stage1_mamba2_plan.hpp"
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <tuple>

struct Rotations {
  int slots;
  long long original = 0, prefix_edges = 0, hoisted_sources = 0, batches = 0;
  std::map<std::vector<int>, long long> groups;
  void group(const std::vector<int>& offsets) {
    ++batches;
    std::set<std::vector<int>> prefixes, sources;
    for (int offset : offsets) {
      std::vector<int> prefix;
      for (int step : fhemamba::rotation_steps(offset, slots, true)) {
        ++original;
        sources.insert(prefix);
        prefix.push_back(step);
        prefixes.insert(prefix);
      }
    }
    prefix_edges += prefixes.size();
    hoisted_sources += sources.size();
    ++groups[offsets];
  }
  void single(int offset) { group({offset}); }
  void sum(int count, int stride) {
    // Conservative: do not share non-accumulated rotation_sum branches.
    for (const auto& step : fhemamba::stage1::rotation_sum_schedule(count, true))
      single(step.offset * stride);
  }
  void transform(std::vector<int> offsets) {
    const auto plan = fhemamba::plan_packed_diagonals(offsets, slots, true);
    if (!plan.baby_step) { group(offsets); return; }
    group(std::vector<int>(plan.offsets.begin(), plan.offsets.begin() + plan.baby_step));
    const int stride = plan.offsets[1] - plan.offsets[0];
    for (int first = 0; first < static_cast<int>(plan.offsets.size()); first += plan.baby_step)
      single(first * stride); // Different inner accumulator for each giant.
  }
  void routing(const std::vector<double>& indices, bool scatter) {
    bool monotone = true, identity = true;
    std::set<int> offsets;
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      monotone &= !i || indices[i] > indices[i - 1];
      identity &= indices[i] == i;
      offsets.insert((scatter ? -1 : 1) * (static_cast<int>(indices[i]) - i));
    }
    if (scatter && identity) return;
    if (monotone && offsets.size() > 32) {
      for (const auto& stage : fhemamba::monotone_routing(indices, scatter)) {
        std::vector<int> shifts;
        for (const auto& [offset, positions] : stage) shifts.push_back(offset);
        const auto plan = fhemamba::plan_packed_diagonals(shifts, slots, true);
        if (plan.baby_step) transform(shifts);
        else for (int offset : shifts) single(offset); // Mask before rotation.
      }
    } else transform({offsets.begin(), offsets.end()});
  }
};

struct Polynomial {
  std::set<int> basis{1};
  long long recombines = 0;
};
Polynomial polynomial(const fhemamba::PackedNode& node) {
  Polynomial out;
  std::vector<double> coefficients(node.data.begin() + 2, node.data.end());
  double at_zero = 0;
  for (int i = 0; i < static_cast<int>(coefficients.size()); i += 2)
    at_zero += (i % 4 ? -1 : 1) * coefficients[i];
  coefficients[0] -= at_zero;
  int levels = 0;
  while ((1 << levels) < static_cast<int>(coefficients.size())) ++levels;
  const int baby = 1 << ((std::max(1, levels) + 1) / 2);
  std::function<void(int)> basis = [&](int i) {
    if (!out.basis.insert(i).second) return;
    basis(i / 2);
    if (i % 2) basis((i + 1) / 2);
  };
  std::function<void(std::vector<double>)> evaluate = [&](std::vector<double> c) {
    const int degree = static_cast<int>(c.size()) - 1;
    if (degree < baby) {
      for (int i = 1; i <= degree; ++i) if (std::abs(c[i]) >= 1e-14) basis(i);
      return;
    }
    int k = baby;
    while (2 * k - 1 < degree) k *= 2;
    std::vector<double> upper(c.begin() + k, c.end()), lower(c.begin(), c.begin() + k);
    for (int i = 1; i < static_cast<int>(upper.size()); ++i) upper[i] *= 2;
    for (int i = k + 1; i <= degree; ++i) lower[2 * k - i] -= c[i];
    basis(k); evaluate(lower); evaluate(upper); ++out.recombines;
  };
  evaluate(coefficients);
  return out;
}

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1]);
  const auto program = fhemamba::read_packed_program(input, true);
  const auto plan = fhemamba::plan_packed_depth(program);
  Rotations rotations{program.slots};
  long long chebs = 0, basis_multiplies = 0, basis_squares = 0, recombines = 0;
  long long dag_multiplies = 0, dag_squares = 0, linears = 0, diagonal_products = 0, babies = 0;
  std::map<std::tuple<int, int, double, double>, std::vector<std::pair<int, Polynomial>>> polys;
  std::vector<int> nonlinear_frontier(program.nodes.size());
  int epoch = 0;
  struct PolyNode { int id, epoch, frontier, size, degree, ctct, recombines; std::set<int> basis; };
  std::vector<PolyNode> poly_nodes;
  std::map<std::tuple<std::string, int, std::vector<int>, std::vector<double>>, int> routing_seen;
  std::map<std::string, int> duplicate_routing;
  for (int i = 0; i < static_cast<int>(program.nodes.size()); ++i) {
    if (!plan.live[i]) continue;
    const auto& n = program.nodes[i];
    const auto& op = n.operation;
    if (op == "feedback") ++epoch;
    for (int parent : n.parents)
      nonlinear_frontier[i] = std::max(nonlinear_frontier[i], nonlinear_frontier[parent]);
    nonlinear_frontier[i] += op == "cheb";
    if (op == "gather" || op == "scatter" || op == "repeat") {
      const auto key = std::tuple{op, n.size, n.parents, n.data};
      if (!routing_seen.emplace(key, i).second) ++duplicate_routing[op];
    }
    if (op == "linear" || op == "linear_ref") {
      ++linears;
      const int columns = program.nodes[n.parents[0]].size;
      const auto shape = fhemamba::stage1::resolve_interleaved_replicated_shape(n.size, columns, program.slots, 0);
      if (shape.replicas <= 1) throw std::runtime_error("Unmodeled linear fallback");
      const int baby = std::max(2, static_cast<int>(std::sqrt(shape.per_replica)));
      babies += baby; diagonal_products += shape.per_replica;
      rotations.sum(shape.reps, -columns);
      rotations.sum(shape.replicas + shape.guard_windows, -shape.window);
      std::vector<int> offsets;
      for (int j = 0; j < baby; ++j) offsets.push_back(j * shape.replicas);
      rotations.group(offsets);
      for (int first = 0; first < shape.per_replica; first += baby) rotations.single(first * shape.replicas);
      rotations.sum(shape.replicas, shape.window + 1);
    } else if (op == "gather" || op == "scatter") rotations.routing(n.data, op == "scatter");
    else if (op == "repeat") {
      const int outer = n.data[0], inner = n.data[1], count = n.data[2];
      std::vector<double> indices(outer * inner);
      for (int g = 0; g < outer; ++g) for (int j = 0; j < inner; ++j) indices[g * inner + j] = g * inner * count + j;
      rotations.routing(indices, true); rotations.sum(count, -inner);
    } else if (op == "sum") {
      const int width = n.data[0];
      for (int step = 1; step < width; step *= 2) rotations.single(step);
      std::vector<double> indices(n.size);
      for (int j = 0; j < n.size; ++j) indices[j] = j * width;
      rotations.routing(indices, false);
    } else if (op == "cheb") {
      auto p = polynomial(n); ++chebs;
      basis_multiplies += p.basis.size() - 1; recombines += p.recombines;
      for (int b : p.basis) basis_squares += b % 2 == 0;
      poly_nodes.push_back({i, epoch, nonlinear_frontier[i], n.size,
          static_cast<int>(n.data.size()) - 3, static_cast<int>(p.basis.size() - 1 + p.recombines),
          static_cast<int>(p.recombines), p.basis});
      polys[{n.parents[0], n.size, n.data[0], n.data[1]}].push_back({i, std::move(p)});
    } else if (op == "mul") {
      ++dag_multiplies; dag_squares += n.parents[0] == n.parents[1];
    }
  }
  std::cout << std::setprecision(17)
    << "{\"scope\":\"Static source calls; refresh excluded; not a speed result\","
    << "\"rotations\":{\"baseline\":" << rotations.original << ",\"prefix_edges\":" << rotations.prefix_edges
    << ",\"hoisted_sources\":" << rotations.hoisted_sources << ",\"groups\":[";
  bool first = true;
  for (const auto& [offsets, count] : rotations.groups) {
    if (offsets.size() < 2) continue;
    if (!first) std::cout << ','; first = false;
    std::cout << "{\"calls\":" << count << ",\"offsets\":[";
    for (int j = 0; j < static_cast<int>(offsets.size()); ++j) { if (j) std::cout << ','; std::cout << offsets[j]; }
    std::cout << "]}";
  }
  std::cout << "]},\"polynomials\":{\"calls\":" << chebs << ",\"basis_multiplies\":" << basis_multiplies
    << ",\"basis_squares\":" << basis_squares << ",\"recombines\":" << recombines
    << ",\"dag_multiplies\":" << dag_multiplies << ",\"dag_squares\":" << dag_squares << ",\"shared_input_groups\":[";
  first = true;
  for (const auto& [key, entries] : polys) {
    if (entries.size() < 2) continue;
    if (!first) std::cout << ','; first = false;
    auto [parent, size, lo, hi] = key;
    long long baseline = 0;
    std::set<int> joined;
    for (const auto& [id, poly] : entries) {
      baseline += poly.basis.size() - 1;
      joined.insert(poly.basis.begin(), poly.basis.end());
    }
    std::cout << "{\"parent\":" << parent << ",\"size\":" << size << ",\"lo\":" << lo << ",\"hi\":" << hi
      << ",\"basis_calls\":" << baseline << ",\"shared_basis_calls\":" << joined.size() - 1 << ",\"nodes\":[";
    for (int j = 0; j < static_cast<int>(entries.size()); ++j) { if (j) std::cout << ','; std::cout << entries[j].first; }
    std::cout << "]}";
  }
  std::cout << "]},\"linears\":{\"calls\":" << linears << ",\"diagonal_products\":" << diagonal_products
    << ",\"baby_ciphertexts\":" << babies << "},\"duplicate_routing_same_parent_and_map\":{";
  first = true;
  for (const auto& [op, count] : duplicate_routing) {
    if (!first) std::cout << ','; first = false;
    std::cout << '"' << op << "\":" << count;
  }
  std::cout << "},\"nonlinear_frontiers\":[";
  first = true;
  for (const auto& n : poly_nodes) {
    if (!first) std::cout << ','; first = false;
    std::cout << "{\"id\":" << n.id << ",\"epoch\":" << n.epoch << ",\"frontier\":" << n.frontier
      << ",\"size\":" << n.size << ",\"degree\":" << n.degree << ",\"ctct\":" << n.ctct
      << ",\"recombines\":" << n.recombines << ",\"basis\":[";
    bool first_basis = true;
    for (int b : n.basis) { if (!first_basis) std::cout << ','; first_basis = false; std::cout << b; }
    std::cout << "]}";
  }
  std::cout << "]}\n";
}
