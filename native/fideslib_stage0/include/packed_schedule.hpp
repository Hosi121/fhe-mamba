#pragma once

#include "packed_lifetime.hpp"
#include <algorithm>
#include <set>

namespace fhemamba {
// A runtime frontier keeps ciphertext levels authoritative. Feedback epochs
// are barriers, including independent work that was originally before a
// client callback. Uses count edges (including x*x), not just distinct users.
class PackedReadySchedule {
 public:
  PackedReadySchedule(const PackedProgram& program, const std::vector<bool>& live)
      : program_(program), uses_(plan_packed_uses(program, live)),
        pending_(program.nodes.size()), remaining_uses_(program.nodes.size()),
        epoch_(program.nodes.size()), done_(program.nodes.size()),
        pinned_(program.nodes.size()) {
    int epoch = 0;
    for (int i = 0; i < static_cast<int>(program.nodes.size()); ++i) {
      if (program.nodes[i].operation == "feedback") ++epoch;
      epoch_[i] = epoch;
      if (epoch >= static_cast<int>(epoch_remaining_.size())) epoch_remaining_.resize(epoch + 1);
      done_[i] = !live[i];
      if (live[i]) {
        pending_[i] = program.nodes[i].parents.size();
        ++epoch_remaining_[epoch]; ++remaining_;
      }
      remaining_uses_[i] = uses_.consumers[i].size();
    }
    for (const auto& output : program.outputs) pinned_[output.node] = true;
    activate_epoch();
  }
  const std::set<int>& ready() const { return ready_; }
  bool done(int i) const { return done_[i]; }
  bool empty() const { return remaining_ == 0; }
  bool final_use(int parent, int operation) const {
    const auto& parents = program_.nodes[operation].parents;
    return !pinned_[parent] && remaining_uses_[parent] == std::count(parents.begin(), parents.end(), parent);
  }
  bool releasable(int i) const { return !pinned_[i] && remaining_uses_[i] == 0; }
  int next_consumer(int i) const {
    for (int consumer : uses_.consumers[i]) if (!done_[consumer]) return consumer;
    return -1;
  }
  void complete(int i) {
    if (!ready_.erase(i)) throw std::logic_error("completed node was not ready");
    done_[i] = true; --remaining_; --epoch_remaining_[active_epoch_];
    for (int parent : program_.nodes[i].parents) --remaining_uses_[parent];
    for (int consumer : uses_.consumers[i]) {
      if (!--pending_[consumer] && epoch_[consumer] == active_epoch_) ready_.insert(consumer);
    }
    if (!epoch_remaining_[active_epoch_]) activate_epoch();
    if (remaining_ && ready_.empty()) throw std::logic_error("packed frontier cannot advance");
  }
 private:
  void activate_epoch() {
    while (active_epoch_ < static_cast<int>(epoch_remaining_.size()) && !epoch_remaining_[active_epoch_])
      ++active_epoch_;
    for (int i = 0; i < static_cast<int>(program_.nodes.size()); ++i)
      if (!done_[i] && epoch_[i] == active_epoch_ && !pending_[i]) ready_.insert(i);
  }
  const PackedProgram& program_;
  PackedUsePlan uses_;
  std::vector<int> pending_, remaining_uses_, epoch_, epoch_remaining_;
  std::vector<bool> done_, pinned_;
  std::set<int> ready_;
  int active_epoch_ = 0, remaining_ = 0;
};
}  // namespace fhemamba
