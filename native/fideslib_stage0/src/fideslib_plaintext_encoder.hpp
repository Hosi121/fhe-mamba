#pragma once

#include <fideslib.hpp>
#include <openfhe.h>
#include <any>
#include <memory>
#include <vector>
#include "fideslib_periodic_encoder.hpp"

namespace fhemamba {

// Preserve OpenFHE FFT, rounding and scale handling while deferring its final
// RNS NTT to the GPU. Instances have fixed context/slots and private params.
class CoefficientPlaintextEncoder {
 public:
  using Context = fideslib::CryptoContext<fideslib::DCRTPoly>;
  using Params = lbcrypto::DCRTPoly::Params;
  CoefficientPlaintextEncoder(Context context, uint32_t slots,
                              bool move_coefficients = false);
  auto encode(const std::vector<double>& values, uint32_t level,
              std::size_t degree = 1) const -> fideslib::Plaintext;

 private:
  Context context_;
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cpu_;
  uint32_t slots_;
  bool move_coefficients_;
  std::vector<std::shared_ptr<Params>> full_params_, coefficient_params_;
};

// Internal FIDESlib bridge for the pinned backend. Avoid pass-by-value limb
// copies and, for coefficient-format inputs, apply its existing GPU NTT.
// Returns only after upload/NTT complete, preserving all buffer lifetimes.
inline constexpr int kPlaintextNttBatch = 16;
enum class PlaintextUploadMode { AlreadyLoaded, Staged, Direct, Borrowed };
// Borrowed reads the pinned OpenFHE word layout until synchronization. Unsupported
// layouts fall back to Direct, then Staged. Report the path actually executed.
auto load_plaintext(fideslib::CryptoContext<fideslib::DCRTPoly>& context,
                    fideslib::Plaintext& plaintext, int ntt_batch = kPlaintextNttBatch,
                    PlaintextUploadMode mode = PlaintextUploadMode::Staged) -> PlaintextUploadMode;

// Probe-only readback of the loaded GPU polynomial in ordinary OpenFHE form.
auto readback_plaintext(const fideslib::Plaintext& plaintext) -> lbcrypto::DCRTPoly;

// Shared serial evaluator policy. CPU cache workers and client encryption can
// retain the stock encoder; all evaluator uploads must pass through load().
// Periodic coefficients keep their small CPU NTT even when GPU NTT is enabled.
struct PlaintextPreparationOptions {
  bool fast_upload = false;
  bool gpu_ntt = false;
  bool profile = false;
  bool direct_upload = false;
  bool move_coefficients = false;
  bool borrow_upload = false;
};

class PlaintextPreparation {
 public:
  using Context = CoefficientPlaintextEncoder::Context;
  using Ciphertext = fideslib::Ciphertext<fideslib::DCRTPoly>;
  PlaintextPreparation(Context context, uint32_t slots, PlaintextPreparationOptions options);
  void enable_periodic_encoding(uint32_t slots);
  auto encode(const std::vector<double>& values, uint32_t level,
              std::size_t degree = 1, uint32_t packing_slots = 0) -> fideslib::Plaintext;
  auto encode_addend(const Ciphertext& ciphertext, const std::vector<double>& values,
                     long long& reencodes) -> fideslib::Plaintext;
  auto align_addend(const Ciphertext& ciphertext, fideslib::Plaintext plaintext,
                    const std::vector<double>& values, long long& reencodes)
      -> fideslib::Plaintext;
  void load(fideslib::Plaintext& plaintext);

  long long gpu_ntt_encodes = 0, fast_uploads = 0, subring_encodes = 0;
  long long direct_uploads = 0, borrowed_uploads = 0, moved_coefficient_encodes = 0;
  double upload_seconds = 0;

 private:
  Context context_;
  uint32_t slots_;
  bool fast_upload_, profile_, move_coefficients_;
  PlaintextUploadMode upload_mode_;
  std::unique_ptr<CoefficientPlaintextEncoder> coefficient_;
  std::unique_ptr<stage1::PeriodicPlaintextEncoder> periodic_;
};

}  // namespace fhemamba
