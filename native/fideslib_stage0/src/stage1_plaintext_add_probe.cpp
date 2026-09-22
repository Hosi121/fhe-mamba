// Regression probe for same-level plaintext addition after a ciphertext square.
#include <fideslib.hpp>

#include "fideslib_plaintext_ops.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fideslib;
extern "C" int cudaDeviceSynchronize(void);

auto main(int argc, char* argv[]) -> int {
  try {
    if (argc != 4) {
      throw std::invalid_argument("usage: stage1_plaintext_add_probe OUTPUT REPO_COMMIT BINARY_SHA256");
    }
    constexpr int slots = 8192;
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetSecretKeyDist(SPARSE_TERNARY);
    parameters.SetRingDim(2 * slots);
    parameters.SetBatchSize(slots);
    parameters.SetMultiplicativeDepth(8);
    parameters.SetScalingModSize(40);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetNumLargeDigits(3);
    parameters.SetDevices({0});
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->LoadContext(keys.publicKey);

    std::vector<double> values(slots), constants(slots);
    for (int slot = 0; slot < slots; ++slot) {
      values[slot] = (slot % 17 - 8) / 16.0;
      constants[slot] = (slot % 5 - 2) / 8.0;
    }
    auto input = cc->MakeCKKSPackedPlaintext(values);
    auto encrypted = cc->Encrypt(keys.publicKey, input);
    auto error = [&](const Ciphertext<DCRTPoly>& result, bool squared) {
      cudaDeviceSynchronize();
      Plaintext decrypted;
      auto handle = result->Clone();
      cc->Decrypt(keys.secretKey, handle, &decrypted);
      decrypted->SetLength(slots);
      const auto actual = decrypted->GetRealPackedValue();
      double maximum = 0;
      for (int slot = 0; slot < slots; ++slot) {
        const double expected =
            (squared ? values[slot] * values[slot] : values[slot]) + constants[slot];
        if (!std::isfinite(actual[slot])) {
          throw std::runtime_error("nonfinite probe output");
        }
        maximum = std::max(maximum, std::abs(actual[slot] - expected));
      }
      return maximum;
    };

    double fixed_error = 0, unaligned_error = 0;
    long long reencodes = 0;
    std::vector<int> tested_levels;
    // Check fresh degree-1 and degree-2 values at several consumed levels.
    for (int step = 0; step < 3; ++step) {
      tested_levels.push_back(static_cast<int>(encrypted->GetLevel()));
      auto constant = cc->MakeCKKSPackedPlaintext(
          constants, 1, encrypted->GetLevel(), nullptr, slots);
      auto fresh_constant = fhemamba::stage1::additive_plaintext(
          cc, encrypted, constant, constants, slots, reencodes);
      fixed_error = std::max(fixed_error, error(cc->EvalAdd(encrypted, fresh_constant), false));
      auto squared = cc->EvalMult(encrypted, encrypted);
      constant = cc->MakeCKKSPackedPlaintext(
          constants, 1, squared->GetLevel(), nullptr, slots);
      unaligned_error = std::max(unaligned_error, error(cc->EvalAdd(squared, constant), true));
      auto corrected = fhemamba::stage1::additive_plaintext(
          cc, squared, constant, constants, slots, reencodes);
      fixed_error = std::max(fixed_error, error(cc->EvalAdd(squared, corrected), true));
      cc->EvalMultInPlace(encrypted, 1.0);
      cc->RescaleInPlace(encrypted);
    }
    const bool passed = fixed_error < 1e-6 && reencodes == 3;
    std::ofstream out(argv[1]);
    if (!out) throw std::runtime_error("cannot write output");
    out << std::setprecision(12)
        << "{\"stage\":\"plaintext-add-scale-probe\",\"version\":\"" << FHEMAMBA_VERSION
        << "\",\"repo_commit\":\"" << argv[2] << "\",\"binary_sha256\":\"" << argv[3]
        << "\",\"backend\":\"FIDESlib\",\"encrypted\":true,\"passed\":"
        << (passed ? "true" : "false") << ",\"status\":\"" << (passed ? "passed" : "failed")
        << "\",\"parameters\":{\"ring_dim\":16384,\"depth\":8,\"scale\":40,\"security\":\"not-set\"}"
        << ",\"measurements\":{\"max_abs_error\":" << fixed_error
        << ",\"uncorrected_max_abs_error\":" << unaligned_error
        << ",\"add_scale_reencodes\":" << reencodes << ",\"levels\":["
        << tested_levels[0] << "," << tested_levels[1] << "," << tested_levels[2] << "]}"
        << ",\"measurement_scope\":{\"artifact_level_report\":true,"
           "\"full_model_correctness_claimed\":false,\"claim\":"
           "\"Encrypted same-level plaintext-addition regression; not a model or security gate.\"}}\n";
    std::cout << "fixed_error=" << fixed_error << " uncorrected_error=" << unaligned_error << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
