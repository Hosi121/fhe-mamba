#include "public_ciphertext_reuse.hpp"

#include <memory>
#include <stdexcept>

static void require(bool condition) {
  if (!condition) throw std::runtime_error("public ciphertext reuse contract failed");
}

int main() {
  using namespace fhemamba;
  PackedProgram program{8, 8, {
      {"public", 1, {}, {0}}, {"public", 4, {}, {0, 0, 0, 0}},
      {"public", 2, {}, {1, 0}}, {"public", 1, {}, {1}},
      {"public", 2, {}, {0, -0.0}}, {"public", 2, {}, {0, 1}},
      {"input", 1, {}, {0}}, {"feedback", 1, {6}, {}},
      {"public", 1, {}, {2}}, {"public", 1, {}, {0}},
  }, {}};
  std::vector<bool> live(program.nodes.size(), true);
  live.back() = false;
  const auto plan = plan_public_ciphertexts(program, live);
  require(plan.uses.size() == 5 && plan.uses[plan.group[0]] == 2);
  require(plan.group[0] == plan.group[1] && plan.group[2] == plan.group[3]);
  require(plan.group[4] != plan.group[0] && plan.group[5] != plan.group[2]);
  require(plan.group[6] == -1 && plan.group[7] == -1 && plan.group[9] == -1);
  int calls = 0;
  std::weak_ptr<int> first_seed;
  auto encrypt = [&] {
    auto value = std::make_shared<int>(++calls);
    if (calls == 1) first_seed = value;
    return value;
  };
  auto clone = [](const auto& value) { return std::make_shared<int>(*value); };
  PublicCiphertextReuse<std::shared_ptr<int>> reuse(plan);
  auto first = reuse.get(1, encrypt, clone);  // Reverse source-node order.
  *first = 99;                              // Caller mutations cannot alter seed.
  require(!first_seed.expired() && reuse.live_seeds == 1);
  auto second = reuse.get(0, encrypt, clone);
  require(*second == 1 && first != second && first_seed.expired());
  require(reuse.live_seeds == 0 && reuse.encryptions == 1 && reuse.reuses == 1);
  reuse.get(2, encrypt, clone);
  reuse.get(4, encrypt, clone);
  reuse.get(5, encrypt, clone);
  reuse.get(8, encrypt, clone);
  reuse.get(3, encrypt, clone);
  require(calls == 5 && reuse.live_seeds == 0 && reuse.reuses == 2);
  for (int invalid : {0, 6, 7, 9}) {
    bool rejected = false;
    try { reuse.get(invalid, encrypt, clone); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
  }
  PublicCiphertextReuse<std::shared_ptr<int>> next_request(plan);
  require(*next_request.get(0, encrypt, clone) == 6);  // New encryption per request.
}
