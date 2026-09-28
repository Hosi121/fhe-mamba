#include "packed_depth.hpp"
#include "packed_schedule.hpp"
#include <iostream>

using namespace fhemamba;
static void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

static void check_bounded_speculation() {
  PackedProgram chain{32, 1024, {{"input", 1, {}, {2.}, 2}}, {}};
  int state = 0;
  for (int step = 1; step <= 16; ++step) {
    const int input = chain.nodes.size();
    chain.nodes.push_back({"input", 1, {}, {static_cast<double>(step)}, 32});
    chain.nodes.push_back({"add", 1, {state, input}, {}, 1024});
    state = input + 1;
  }
  chain.outputs.push_back({state, {}, {}});
  const auto live = plan_packed_depth(chain).live;
  auto run = [&](std::size_t limit) {
    PackedReadySchedule schedule(chain, live);
    std::vector<double> values(chain.nodes.size());
    while (!schedule.empty()) {
      // The first update waits for refresh. Later inputs can be prepared
      // independently, but their values remain live until that update finishes.
      const int i = schedule.select([](int node) { return node != 2; }, limit);
      const auto& node = chain.nodes[i];
      for (int parent : node.parents) require(schedule.done(parent), "dependency was bypassed");
      values[i] = node.operation == "input" ? node.data[0]
                  : values[node.parents[0]] + values[node.parents[1]];
      schedule.complete(i);
    }
    require(values[state] == 138., "memory admission changed the result");
    require(schedule.live_values() == 1, "unpinned values were retained");
    if (limit) {
      require(schedule.limit_selections() == 1, "blocked oldest branch was not selected");
      require(schedule.peak_live_values() <= limit + 1, "speculative inputs exceeded the budget");
    } else {
      require(schedule.limit_selections() == 0, "default schedule changed");
      require(schedule.peak_live_values() == 18, "unlimited working set was not counted");
    }
  };
  run(0);
  run(4);

  PackedProgram pinned{8, 2, {{"input", 1, {}, {1.}, 2}, {"input", 1, {}, {1.}, 2}},
                        {{0, {}, {}}, {1, {}, {}}}};
  PackedReadySchedule outputs(pinned, plan_packed_depth(pinned).live);
  while (!outputs.empty()) outputs.complete(outputs.select([](int) { return true; }, 1));
  require(outputs.live_values() == 2, "soft limit must not discard pinned outputs");
}

int main() {
  check_bounded_speculation();
  PackedProgram branches{16, 64, {
    {"input", 1, {}, {.2}, 2},
    {"mulp", 1, {0}, {2.}, 2},
    {"input", 1, {}, {.5}, 2},
    {"mul", 1, {1, 1}, {}, 2},
    {"feedback", 1, {3}, {}, 2},
    {"public", 1, {}, {.3}, 2},
    {"add", 1, {4, 5}, {}, 2},
    {"feedback", 1, {6}, {}, 2},
    {"add", 1, {7, 0}, {}, 2},
    {"mulp", 1, {0}, {7.}, 8}, // dead; cannot hold a frontier open
  }, {{0, {}, {}}, {2, {}, {}}, {8, {}, {}}}};
  const auto depth = plan_packed_depth(branches);
  PackedReadySchedule schedule(branches, depth.live);
  require(schedule.ready() == std::set<int>({0, 2}), "incorrect initial frontier");
  schedule.complete(0);
  require(!schedule.final_use(0, 1), "pinned value can be consumed");
  schedule.complete(1);
  require(schedule.final_use(1, 3), "duplicate edges prevent final-use ownership");
  schedule.complete(3);
  require(schedule.releasable(1) && !schedule.releasable(0), "runtime lifetime is incorrect");
  require(schedule.ready() == std::set<int>({2}), "feedback bypassed independent earlier work");
  schedule.complete(2);
  require(schedule.ready() == std::set<int>({4, 5}), "feedback epoch did not advance");
  schedule.complete(5); schedule.complete(4); schedule.complete(6);
  require(schedule.ready() == std::set<int>({7}), "second feedback barrier is incorrect");
  schedule.complete(7); schedule.complete(8);
  require(schedule.empty() && schedule.ready().empty(), "frontier did not finish");
  require(schedule.next_consumer(0) == -1, "completed consumer remains pending");
  require(schedule.live_values() == 3, "duplicate edges or feedback corrupted live counts");
  std::cout << "ready scheduling preserves dependencies, feedback barriers and lifetimes\n";
}
