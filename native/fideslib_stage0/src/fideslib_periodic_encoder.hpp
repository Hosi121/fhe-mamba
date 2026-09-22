#pragma once

#include <fideslib.hpp>
#include <openfhe.h>
#include <math/dftransform.h>

#include <algorithm>
#include <any>
#include <bit>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fhemamba::stage1 {

// Encode an s-slot periodic public plaintext using a 2s-point NTT, then
// embed it in the original N-dimensional ring. No ciphertext uses a small
// ring. All scaling, rounding, and RNS conversion remain OpenFHE operations.
// Construct before starting encoder workers: FFT initialization is global.
class PeriodicPlaintextEncoder {
 public:
  using Context = fideslib::CryptoContext<fideslib::DCRTPoly>;
  using Params = lbcrypto::DCRTPoly::Params;

  PeriodicPlaintextEncoder(Context context, uint32_t slots)
      : context_(std::move(context)),
        cpu_(std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(context_->cpu)),
        slots_(slots), small_n_(2 * slots), n_(cpu_->GetRingDimension()) {
    if (!std::has_single_bit(slots_) || slots_ > n_ / 2)
      throw std::invalid_argument("periodic encoding requires a power-of-two slot count dividing N/2");
    repeat_ = n_ / small_n_;
    const auto original = cpu_->GetElementParams();
    const auto& towers = original->GetParams();
    std::vector<lbcrypto::NativeInteger> moduli, identity_roots;
    for (const auto& tower : towers) {
      const auto q = tower->GetModulus();
      const auto root = tower->GetRootOfUnity().ModExp(lbcrypto::NativeInteger(repeat_), q);
      lbcrypto::NativeVector table(small_n_, q), precon(small_n_, q);
      lbcrypto::NativeInteger power(1);
      for (uint32_t j = 0; j < small_n_; ++j) {
        const auto reversed = lbcrypto::ReverseBits(j, std::bit_width(small_n_ - 1));
        table[reversed] = power;
        precon[reversed] = power.PrepModMulConst(q);
        power = power.ModMul(root, q);
      }
      roots_.push_back(std::move(table));
      precon_.push_back(std::move(precon));
      moduli.push_back(q);
      identity_roots.emplace_back(1);
    }
    for (std::size_t level = 0; level < towers.size(); ++level) {
      auto full = std::make_shared<Params>(*original);
      for (std::size_t drop = 0; drop < level; ++drop) full->PopLastParam();
      full_params_.push_back(std::move(full));
      // Root 1 deliberately makes OpenFHE's format switch a no-op. The
      // temporary contains rounded coefficients, despite its EVALUATION tag.
      // Using ordinary small-ring NTT params would overwrite OpenFHE's global
      // NTT tables, which are keyed by modulus rather than (modulus, size).
      coefficient_params_.push_back(
          std::make_shared<Params>(2 * small_n_, moduli, identity_roots));
      moduli.pop_back();
      identity_roots.pop_back();
    }
    lbcrypto::DiscreteFourierTransform::Initialize(2 * small_n_, slots_);
  }

  auto encode(const std::vector<double>& values, uint32_t level,
              std::size_t noise_scale_degree = 1) const -> fideslib::Plaintext {
    if (values.size() > slots_ || level >= full_params_.size())
      throw std::invalid_argument("periodic plaintext values or level out of range");
    lbcrypto::Plaintext plaintext;
    {
      // The temporary's identity-root format switches do no arithmetic.
      // Suppress their OpenMP teams through this task's ICV, not OpenFHE's
      // process-wide controls; restore it even if the encoder throws.
#ifdef _OPENMP
      struct SerialTemporary {
        int previous = omp_get_max_active_levels();
        SerialTemporary() { omp_set_max_active_levels(0); }
        ~SerialTemporary() { omp_set_max_active_levels(previous); }
      } serial;
#endif
      plaintext = cpu_->MakeCKKSPackedPlaintext(
          values, noise_scale_degree, level, coefficient_params_[level], slots_);
    }
    auto& compact = plaintext->GetElement<lbcrypto::DCRTPoly>();
    lbcrypto::DCRTPoly expanded(full_params_[level], ::Format::EVALUATION, false);
    auto& output = expanded.GetAllElements();
    for (std::size_t tower = 0; tower < output.size(); ++tower) {
      auto transformed = compact.GetElementAtIndex(tower).GetValues();
      intnat::NumberTheoreticTransformNat<lbcrypto::NativeVector>()
          .ForwardTransformToBitReverseInPlace(roots_[tower], precon_[tower], &transformed);
      lbcrypto::NativeVector values_n(n_, transformed.GetModulus());
      // For f(X) = g(X^(N/2s)), evaluation at the N roots depends only on
      // the root exponent modulo 4s. In bit-reversed NTT order each of the
      // 2s small evaluations occupies one consecutive run of N/2s entries.
      for (uint32_t j = 0; j < small_n_; ++j)
        for (uint32_t k = j * repeat_; k < (j + 1) * repeat_; ++k)
          values_n[k] = transformed[j];
      output[tower].SetValues(std::move(values_n), ::Format::EVALUATION);
    }
    compact = std::move(expanded);
    auto context_copy = context_;
    auto result = std::make_shared<fideslib::PlaintextImpl>(std::move(context_copy));
    result->cpu = std::make_any<lbcrypto::Plaintext>(std::move(plaintext));
    if (context_->auto_load_plaintexts) context_->LoadPlaintext(result);
    return result;
  }

  auto slots() const -> uint32_t { return slots_; }

 private:
  Context context_;
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cpu_;
  uint32_t slots_, small_n_, n_, repeat_;
  std::vector<std::shared_ptr<Params>> full_params_, coefficient_params_;
  std::vector<lbcrypto::NativeVector> roots_, precon_;
};

}  // namespace fhemamba::stage1
