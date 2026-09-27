#include "bounded_prefetch.hpp"
#include <atomic>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>

static void require(bool value) {
  if (!value) throw std::runtime_error("prefetch contract failed");
}
int main() {
  {
    fhemamba::BoundedPrefetch<int> queue(1, 1,
        [owned = std::make_unique<int>(7)](std::size_t) { return *owned; });
    require(queue.pop() == 7);
  }
  for (std::size_t workers : {1u, 2u, 4u})
  for (std::size_t capacity : {1u, 2u, 4u}) {
    fhemamba::BoundedPrefetch<std::unique_ptr<int>> queue(100, capacity,
        [](std::size_t i) { return std::make_unique<int>(i); }, workers);
    for (int i = 0; i < 100; ++i) require(*queue.pop() == i);
    require(queue.peak_size() <= capacity);
    bool exhausted = false;
    try { queue.pop(); } catch (const std::out_of_range&) { exhausted = true; }
    require(exhausted);
  }
  {
    fhemamba::BoundedPrefetch<int> queue(10, 2, [](std::size_t) -> int {
      throw std::runtime_error("producer error");
    });
    bool propagated = false;
    try { queue.pop(); }
    catch (const std::runtime_error& e) { propagated = std::string(e.what()) == "producer error"; }
    require(propagated);
  }
  // Cancel while the producer is back-pressured. Destruction must wake/join it
  // and destroy every owning payload; no polling or timed sleeps in the test.
  std::promise<void> first;
  std::weak_ptr<int> live;
  {
    fhemamba::BoundedPrefetch<std::shared_ptr<int>> queue(100, 1, [&](std::size_t i) {
      auto item = std::make_shared<int>(i);
      if (i == 0) { live = item; first.set_value(); }
      return item;
    });
    first.get_future().wait();
  }
  require(live.expired());
  bool rejected = false;
  try { fhemamba::BoundedPrefetch<int> queue(1, 0, [](std::size_t) { return 1; }); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected);
  // Force reverse completion while retaining ordered consumption. The second
  // worker must progress while item zero is still being prepared.
  {
    std::promise<void> second_started, release_first;
    auto release = release_first.get_future().share();
    std::atomic<int> started = 0;
    fhemamba::BoundedPrefetch<int> queue(20, 2, [&](std::size_t i) {
      ++started;
      if (i == 0) release.wait();
      if (i == 1) second_started.set_value();
      return static_cast<int>(i);
    }, 2);
    second_started.get_future().wait();
    require(started == 2); // Both admitted slots include in-flight items.
    release_first.set_value();
    for (int i = 0; i < 20; ++i) require(queue.pop() == i);
    require(queue.peak_size() <= 2);
  }
  {
    std::promise<void> second_started;
    auto started = second_started.get_future().share();
    fhemamba::BoundedPrefetch<int> queue(10, 2, [&](std::size_t i) {
      if (i == 0) { started.wait(); throw std::runtime_error("parallel producer error"); }
      if (i == 1) second_started.set_value();
      return static_cast<int>(i);
    }, 2);
    bool propagated = false;
    try { queue.pop(); }
    catch (const std::runtime_error& e) { propagated = std::string(e.what()) == "parallel producer error"; }
    require(propagated);
  }
  // Cancellation joins both producers and releases all retained owning items.
  {
    std::promise<void> a, b;
    std::weak_ptr<int> first_live, second_live;
    {
      fhemamba::BoundedPrefetch<std::shared_ptr<int>> queue(100, 2, [&](std::size_t i) {
        auto item = std::make_shared<int>(i);
        if (i == 0) { first_live = item; a.set_value(); }
        if (i == 1) { second_live = item; b.set_value(); }
        return item;
      }, 2);
      a.get_future().wait(); b.get_future().wait();
    }
    require(first_live.expired() && second_live.expired());
  }
  {
    fhemamba::BoundedPrefetch<int> queue(0, 2, [](std::size_t) { return 1; }, 2);
    bool exhausted = false;
    try { queue.pop(); } catch (const std::out_of_range&) { exhausted = true; }
    require(exhausted);
  }
  rejected = false;
  try { fhemamba::BoundedPrefetch<int> queue(1, 2, [](std::size_t) { return 1; }, 0); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected);
}
