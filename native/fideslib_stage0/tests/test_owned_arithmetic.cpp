#include "owned_arithmetic.hpp"
#include <algorithm>
#include <cassert>
#include <memory>

struct Value { int number, level; };
using Handle = std::shared_ptr<Value>;

int main() {
  fhemamba::OwnedArithmeticStats stats;
  int completions = 0;
  std::weak_ptr<Value> pending_read;
  auto subtract = [&](Handle left, Handle right) {
    return fhemamba::evaluate_owned_binary(std::move(left), std::move(right),
        [](const Handle& x) { return std::make_shared<Value>(*x); },
        [](Handle& a, Handle& b) { a->level = b->level = std::max(a->level, b->level); },
        [&](Handle& a, Handle& b) {
          assert(a.get() != b.get()); // Mutable operations need distinct buffers.
          a->number -= b->number;
          pending_read = b;
        },
        [&] { assert(!pending_read.expired()); ++completions; }, stats);
  };
  auto a = std::make_shared<Value>(Value{7, 2});
  auto b = std::make_shared<Value>(Value{3, 5});
  auto original = a.get();
  auto result = subtract(std::move(a), std::move(b));
  assert(!a && !b && result.get() == original);
  assert(result->number == 4 && result->level == 5 && pending_read.expired());
  assert(stats.reused_inputs == 2 && stats.cloned_inputs == 0);

  // Both original live-outs survive a mutating alignment and subtraction.
  a = std::make_shared<Value>(Value{9, 1});
  b = std::make_shared<Value>(Value{2, 6});
  result = subtract(a, b);
  assert(a->number == 9 && a->level == 1 && b->number == 2 && b->level == 6);
  assert(result->number == 7 && result->level == 6);
  assert(stats.reused_inputs == 2 && stats.cloned_inputs == 2);

  // Two dying handles can alias even when no caller retains their value.
  b = a;
  result = subtract(std::move(a), std::move(b));
  assert(result->number == 0 && stats.reused_inputs == 3 && stats.cloned_inputs == 3);

  // A square/double with another live alias must detach both operands.
  a = std::make_shared<Value>(Value{11, 4});
  result = subtract(a, a);
  assert(a->number == 11 && a->level == 4 && result->number == 0);
  assert(stats.calls == 4 && completions == 4 && stats.cloned_inputs == 5);
  bool rejected = false;
  try { subtract({}, a); } catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected && stats.calls == 4 && completions == 4);
}
