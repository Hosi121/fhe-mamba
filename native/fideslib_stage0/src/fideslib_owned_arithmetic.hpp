#pragma once

#include <fideslib.hpp>
#include "owned_arithmetic.hpp"

namespace fhemamba {

enum class CiphertextBinaryOp { Add, Subtract, Multiply };

template <class Complete>
auto owned_ciphertext_square(fideslib::CryptoContext<fideslib::DCRTPoly>& context,
                             fideslib::Ciphertext<fideslib::DCRTPoly> value,
                             Complete&& complete, OwnedArithmeticStats& stats)
    -> fideslib::Ciphertext<fideslib::DCRTPoly> {
  return evaluate_owned_unary(std::move(value),
      [](const auto& input) { return input->Clone(); },
      [&](auto& input) { context->EvalSquareInPlace(input); },
      std::forward<Complete>(complete), stats);
}

// Keep ownership and the backend operation shared; each architecture supplies
// its existing level-alignment and synchronization policy.
template <class Align, class Complete>
auto owned_ciphertext_binary(fideslib::CryptoContext<fideslib::DCRTPoly>& context,
                             fideslib::Ciphertext<fideslib::DCRTPoly> left,
                             fideslib::Ciphertext<fideslib::DCRTPoly> right,
                             CiphertextBinaryOp operation, Align&& align, Complete&& complete,
                             OwnedArithmeticStats& stats)
    -> fideslib::Ciphertext<fideslib::DCRTPoly> {
  return evaluate_owned_binary(std::move(left), std::move(right),
      [](const auto& value) { return value->Clone(); }, std::forward<Align>(align),
      [&](auto& a, auto& b) {
        switch (operation) {
          case CiphertextBinaryOp::Add: context->EvalAddInPlace(a, b); break;
          case CiphertextBinaryOp::Subtract: context->EvalSubInPlace(a, b); break;
          case CiphertextBinaryOp::Multiply: context->EvalMultMutableInPlace(a, b); break;
        }
      }, std::forward<Complete>(complete), stats);
}

}  // namespace fhemamba
