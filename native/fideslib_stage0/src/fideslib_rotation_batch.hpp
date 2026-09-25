#pragma once

#include "rotation_batch.hpp"
#include <fideslib.hpp>

namespace fhemamba {
template <class Synchronize>
auto hoisted_rotation_batch(fideslib::CryptoContext<fideslib::DCRTPoly>& cc,
    const fideslib::Ciphertext<fideslib::DCRTPoly>& input,
    const std::vector<int>& offsets, int slots, bool naf,
    Synchronize synchronize, RotationBatchStats& stats)
    -> std::vector<fideslib::Ciphertext<fideslib::DCRTPoly>> {
  return rotate_prefix_batch(input, offsets, slots, naf,
      [&](const auto& value, int step) {
        auto out = value->Clone();
        cc->EvalRotateInPlace(out, step); synchronize(); return out;
      },
      [&](const auto& value, const auto& steps) {
        auto precompute = cc->EvalFastRotationPrecompute(value);
        auto out = cc->EvalFastRotation(value, steps, 2 * cc->GetRingDimension(), precompute);
        synchronize(); return out;
      }, stats);
}
}  // namespace fhemamba
