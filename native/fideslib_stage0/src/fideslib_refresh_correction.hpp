#pragma once

#include <fideslib.hpp>
#include <CKKS/ApproxModEval.cuh>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/Context.cuh>
#include <cmath>

namespace fhemamba {

// Produce 4096*first + second, keeping the original CKKS scale and level.
// The caller must divide its existing output multiplier/mask by 4096. This
// integer RNS operation must not be replaced by EvalMult(double): that would
// consume another scale level. Incompatible metadata keeps the old two paths.
template <class Synchronize>
bool merge_refresh_correction(fideslib::Ciphertext<fideslib::DCRTPoly>& first,
                              const fideslib::Ciphertext<fideslib::DCRTPoly>& second,
                              Synchronize&& synchronize) {
  if (!first || !second || first == second || !first->loaded || !second->loaded ||
      first->parent_context != second->parent_context) return false;
  using DeviceCiphertext = FIDESlib::CKKS::Ciphertext;
  auto device = [](const auto& value) {
    return std::static_pointer_cast<DeviceCiphertext>(
        value->parent_context->GetDeviceCiphertext(value->gpu));
  };
  auto a = device(first), b = device(second);
  if (a->getLevel() != b->getLevel() || a->NoiseLevel != b->NoiseLevel ||
      a->NoiseFactor != b->NoiseFactor || !std::isfinite(a->NoiseFactor) || a->NoiseFactor <= 0 ||
      a->slots != b->slots || a->keyID != b->keyID ||
      a->c0.isModUp() || a->c1.isModUp() || b->c0.isModUp() || b->c1.isModUp()) return false;
  if (!first.unique()) {
    first = first->Clone();
    a = device(first);
  }
  synchronize();
  FIDESlib::CKKS::SetCurrentContext(a->cc_);
  FIDESlib::CKKS::multIntScalar(*a, 4096);
  synchronize();
  first->parent_context->EvalAddInPlace(first, second);
  synchronize();
  return true;
}
}  // namespace fhemamba
