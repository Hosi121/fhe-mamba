#include "packed_lifetime.hpp"
#include <cassert>
#include <memory>

int main() {
  using Value = std::shared_ptr<int>;
  const auto clone = [](const Value& v) { return std::make_shared<int>(*v); };
  long long reused = 0;
  auto input = std::make_shared<int>(7);
  // A later branch or an output still needs its original input.
  auto branch = fhemamba::consume_or_clone(input, false, clone, reused);
  *branch *= 3;
  assert(*input == 7 && *branch == 21 && reused == 0);
  // Last use in one DAG node cannot consume another node's alias.
  auto alias = input;
  auto next = fhemamba::consume_or_clone(input, true, clone, reused);
  *next += 4;
  assert(*alias == 7 && *input == 7 && reused == 0);
  alias.reset();
  // x*x: clone the second operand before taking the now unique first one.
  auto right = clone(input);
  const auto original = input.get();
  auto square = fhemamba::consume_or_clone(input, true, clone, reused);
  *square *= *right;
  assert(!input && square.get() == original && *square == 49 && *right == 7 && reused == 1);
}
