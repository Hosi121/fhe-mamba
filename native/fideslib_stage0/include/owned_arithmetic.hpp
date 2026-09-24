#pragma once

#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace fhemamba {

struct OwnedArithmeticStats {
  long long calls = 0, reused_inputs = 0, cloned_inputs = 0;
};

// Consume scratch handles, detaching any aliases before level alignment can
// mutate either operand. Completion must precede release of the right operand:
// an asynchronous evaluator may still be reading its storage.
template <class Handle, class Clone, class Align, class Evaluate, class Complete>
auto evaluate_owned_binary(Handle left, Handle right, Clone&& clone, Align&& align,
                           Evaluate&& evaluate, Complete&& complete,
                           OwnedArithmeticStats& stats) -> Handle {
  if (!left || !right) throw std::invalid_argument("owned arithmetic needs two operands");
  for (auto* value : {&left, &right}) {
    if (value->use_count() == 1) ++stats.reused_inputs;
    else { *value = clone(*value); ++stats.cloned_inputs; }
  }
  align(left, right);
  evaluate(left, right);
  complete();
  ++stats.calls;
  return left;
}

}  // namespace fhemamba
