#pragma once

#include <cstdint>
#include <vector>

namespace fhemamba::stage1 {

// FIDESlib v2.1 addPt aligns a plaintext when levels differ, but skips
// alignment for equal levels even if the ciphertext has scale degree 2.
// Adding a degree-1 plaintext then silently loses the intended constant in
// release builds. All ordinary cached vectors in our evaluator have degree 1.
// Re-encode only this incompatible case at the ciphertext's actual degree.
template <class Context, class Ciphertext, class Plaintext>
auto additive_plaintext(Context& context, const Ciphertext& ciphertext,
                        Plaintext plaintext, const std::vector<double>& values,
                        uint32_t slots, long long& reencodes) -> Plaintext {
  if (ciphertext->GetNoiseScaleDeg() > 1 &&
      plaintext->GetLevel() == ciphertext->GetLevel()) {
    ++reencodes;
    return context->MakeCKKSPackedPlaintext(
        values, ciphertext->GetNoiseScaleDeg(), ciphertext->GetLevel(), nullptr,
        slots);
  }
  return plaintext;
}

}  // namespace fhemamba::stage1
