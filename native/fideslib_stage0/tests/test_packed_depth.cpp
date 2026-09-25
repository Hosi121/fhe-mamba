#include "packed_depth.hpp"
#include "packed_routing.hpp"
#include <iostream>

static void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  using namespace fhemamba;
  // Direct consumers of the same parent already share a cached refresh.
  PackedProgram p{32768, 64, {}, {}};
  p.nodes.push_back({"input", 1, {}, {1}, 64});
  for (int i = 1; i <= 35; ++i) p.nodes.push_back({"mulp", 1, {i - 1}, {1}, 64});
  for (int i = 0; i < 4; ++i) p.nodes.push_back({"cheb", 1, {35}, std::vector<double>(34), 64});
  for (int i = 36; i < 40; ++i) p.outputs.push_back({i, {1}, {1}});
  auto plan = plan_packed_depth(p);
  require(plan.refreshes == 1 && !plan.refresh_after[35], "redundant hoist before direct consumers");
  auto baseline = plan;
  baseline.refresh_after.assign(p.nodes.size(), false);
  // A parent refresh is cached, so all four direct children already share it.
  // Make the actual fork explicit with one identity-depth node per branch.
  p.nodes.resize(36);
  p.outputs.clear();
  for (int i = 0; i < 4; ++i) {
    const int fork = p.nodes.size();
    p.nodes.push_back({"addp", 1, {35}, {0}, 64});
    p.nodes.push_back({"cheb", 1, {fork}, std::vector<double>(34), 64});
    p.outputs.push_back({fork + 1, {1}, {1}});
  }
  plan = plan_packed_depth(p);
  baseline = plan;
  baseline.refresh_after.assign(p.nodes.size(), false);
  require(simulate_packed_depth(p, baseline) == 4, "fixture must refresh each fork separately");
  require(plan.refreshes == 1 && plan.refresh_after[35], "fork refreshes were not hoisted");
  auto again = plan_packed_depth(p);
  require(plan.refresh_after == again.refresh_after, "planning is not deterministic");

  // Feedback decrypts the client output and returns a fresh encryption. It
  // needs neither inherited depth nor an extra server refresh of that output.
  p.nodes.push_back({"feedback", 1, {35}, {}, 64});
  p.outputs.push_back({static_cast<int>(p.nodes.size()) - 1, {1}, {1}});
  plan = plan_packed_depth(p);
  std::vector<int> levels(p.nodes.size());
  simulate_packed_depth(p, plan, &levels);
  require(levels.back() == -1, "feedback did not reset depth");

  PackedProgram masks{1024, 64, {
      {"input", 2, {}, {1, 2}, 64},
      {"linear", 4, {0}, std::vector<double>(8, 1), 64},
      {"gather", 2, {1}, {0, 2}, 64},
      {"linear_ref", 4, {0}, {1}, 64},
      {"mulp", 4, {3}, std::vector<double>(4, -1), 64},
      {"cheb", 2, {0}, std::vector<double>(34), 64},
  }, {{2, {1, 1}, {1, 1}}, {4, std::vector<double>(4), std::vector<double>(4)}}};
  auto mask_plan = plan_packed_depth(masks);
  require(mask_plan.defer_linear_mask[1] && mask_plan.cost[1] == 1,
          "gather must absorb a deferred projection mask");
  require(!mask_plan.defer_linear_mask[3] && mask_plan.cost[3] == 2,
          "nongather consumer requires clean projection padding");
  require(mask_plan.cost[4] == 0, "negation must not consume a multiplication level");
  require(!mask_plan.live[5] && mask_plan.live[1], "dead code or weight reference liveness is wrong");
  masks.outputs.push_back({1, std::vector<double>(4), std::vector<double>(4)});
  require(!plan_packed_depth(masks).defer_linear_mask[1], "observable projection must be masked");

  // S2C-first spends four levels before raising the modulus, and returns four
  // extra levels. The first refresh moves earlier; the interval stays equal.
  PackedProgram chain{32768, 1, {{"input", 1, {}, {0.5}, 1}}, {}};
  for (int i = 1; i <= 37; ++i)
    chain.nodes.push_back({"mulp", 1, {i - 1}, {1}, 1});
  chain.outputs.push_back({37, {0.5}, {0.5}});
  auto ordinary = plan_packed_depth(chain);
  auto s2c = plan_packed_depth(chain, 35, 18);
  require(ordinary.refreshes == 0 && s2c.refreshes == 1,
          "S2C input transform's reserved levels were ignored");
  require(ordinary.ceiling - ordinary.refreshed == s2c.ceiling - s2c.refreshed,
          "usable depth changed after refresh");
  levels.resize(chain.nodes.size());
  simulate_packed_depth(chain, s2c, &levels);
  require(levels.back() == 19, "S2C refreshed level was ignored");
  bool rejected = false;
  try { (void)plan_packed_depth(chain, 18, 18); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "invalid refresh policy was accepted");

  // The estimator must account for each actual mask stage, including
  // reductions whose temporary padding is dirty and still needs selection.
  for (int stride : {1, 2, 64, 128}) {
    std::vector<double> indices(32768 / stride);
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) indices[i] = i * stride;
    require(packed_routing_depth(indices, false) == (stride == 1 ? 1 : static_cast<int>(monotone_routing(indices, false).size())),
            "gather depth differs from its routing stages");
    require(packed_routing_depth(indices, true) == (stride == 1 ? 0 : static_cast<int>(monotone_routing(indices, true).size())),
            "scatter depth differs from its routing stages");
  }
  std::cout << "depth planner shares refreshes and preserves depth limits\n";
}
