#include "rotation_batch.hpp"
#include "chebyshev_basis_cache.hpp"
#include <stdexcept>
#include <memory>

static void check(bool value) { if (!value) throw std::runtime_error("shared plan contract"); }
struct FakeCiphertext {
  int level = 0, degree = 1;
  int GetLevel() const { return level; }
  int GetNoiseScaleDeg() const { return degree; }
};
int main() {
  const std::vector<int> offsets{0, 3, 7, 15, -3, 3, 32771, -32768, 16384, -16384};
  for (bool naf : {false, true}) {
    fhemamba::RotationBatchStats stats;
    auto append = [](std::vector<int> value, int step) { value.push_back(step); return value; };
    const auto results = fhemamba::rotate_prefix_batch(std::vector<int>{}, offsets, 32768, naf,
        append, [&](const auto& input, const auto& steps) {
          std::vector<std::vector<int>> output;
          for (int step : steps) output.push_back(append(input, step));
          return output;
        }, stats);
    long long scalar_edges = 0;
    for (std::size_t i = 0; i < offsets.size(); ++i) {
      auto expected = fhemamba::rotation_steps(offsets[i], 32768, naf);
      check(results[i] == expected); scalar_edges += expected.size();
    }
    check(stats.edges < scalar_edges && stats.preparations < stats.edges);
    check(stats.sibling_batches > 0 && stats.requests == offsets.size());
  }
  fhemamba::PackedProgram program;
  program.nodes = {
    {"input", 1, {}, {0.25}},
    {"cheb", 1, {0}, {-1, 1, 0.1, 0.5}},
    {"cheb", 1, {0}, {-1, 1, 0.4, 0, -0.2}},
    {"cheb", 1, {0}, {-2, 2, 0, 1}},
    {"cheb", 1, {0}, {-1, 1, 0, 1}}
  };
  auto plan = fhemamba::plan_chebyshev_sharing(program, {true, true, true, true, false});
  check(plan.group[1] >= 0 && plan.group[1] == plan.group[2]);
  check(plan.group[3] == -1 && plan.group[4] == -1 && plan.remaining.size() == 1);
  auto input = std::make_shared<FakeCiphertext>();
  fhemamba::ChebyshevBasisState<decltype(input)> state;
  state.reset(input); check(state.matches(input));
  input->level++; check(!state.matches(input));
  state.reset(input); input->degree++; check(!state.matches(input));
  state.reset(input); check(!state.matches(std::make_shared<FakeCiphertext>(*input)));
  std::weak_ptr<FakeCiphertext> weak = input;
  input.reset(); check(weak.expired());
}
