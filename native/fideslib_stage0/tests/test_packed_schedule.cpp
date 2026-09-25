#include "packed_depth.hpp"
#include "packed_schedule.hpp"
#include <iostream>

using namespace fhemamba;
static void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
int main() {
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
  std::cout << "ready scheduling preserves dependencies, feedback barriers and lifetimes\n";
}
