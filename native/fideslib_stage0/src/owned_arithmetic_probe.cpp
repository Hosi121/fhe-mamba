// Same-input encrypted arithmetic and live-out gates for scratch ownership.
#include "fideslib_owned_arithmetic.hpp"
#include <CKKS/Ciphertext.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

using namespace fideslib;
using Ct = Ciphertext<DCRTPoly>;
using Op = fhemamba::CiphertextBinaryOp;
static void sync_gpu() {
  if (cudaDeviceSynchronize()) throw std::runtime_error("CUDA synchronization failed");
}
static bool same_bits(const std::vector<std::complex<double>>& a,
                      const std::vector<std::complex<double>>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
}
static bool same_rns(const FIDESlib::CKKS::RawCipherText& a,
                     const FIDESlib::CKKS::RawCipherText& b) {
  return a.sub_0 == b.sub_0 && a.sub_1 == b.sub_1 && a.moduli == b.moduli &&
         a.numRes == b.numRes && a.N == b.N && a.format == b.format &&
         a.Noise == b.Noise && a.NoiseLevel == b.NoiseLevel && a.slots == b.slots;
}

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--mamba2"))
      throw std::invalid_argument("usage: owned_arithmetic_probe OUTPUT [--mamba2]");
    const bool mamba2 = argc == 3;
    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecurityLevel(HEStd_NotSet);
    p.SetSecretKeyDist(mamba2 ? SPARSE_TERNARY : UNIFORM_TERNARY);
    p.SetCKKSDataType(mamba2 ? COMPLEX : REAL);
    p.SetRingDim(65536); p.SetBatchSize(32768);
    p.SetMultiplicativeDepth(44); p.SetScalingModSize(59); p.SetFirstModSize(60);
    p.SetScalingTechnique(FLEXIBLEAUTO); p.SetKeySwitchTechnique(HYBRID);
    p.SetNumLargeDigits(3); p.SetDevices({0});
    p.SetPlaintextAutoload(false); p.SetCiphertextAutoload(true);
    auto cc = GenCryptoContext(p);
    for (auto feature : {PKE, KEYSWITCH, LEVELEDSHE}) cc->Enable(feature);
    auto keys = cc->KeyGen(); cc->EvalMultKeyGen(keys.secretKey);
    cc->LoadContext(keys.publicKey); sync_gpu();
    auto decrypt = [&](Ct value) {
      Plaintext plain; cc->Decrypt(keys.secretKey, value, &plain);
      plain->SetLength(32768);
      auto values = plain->GetCKKSPackedValue();
      for (const auto& x : values)
        if (!std::isfinite(x.real()) || !std::isfinite(x.imag()))
          throw std::runtime_error("non-finite output");
      return values;
    };
    auto align = [](Ct& a, Ct& b) {
      const auto level = std::max(a->GetLevel(), b->GetLevel());
      a->SetLevel(level); b->SetLevel(level);
    };
    auto snapshot = [&](const Ct& value) {
      sync_gpu();
      FIDESlib::CKKS::RawCipherText raw{};
      auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(cc->GetDeviceCiphertext(value->gpu));
      gpu->store(raw); sync_gpu();
      return raw;
    };
    std::vector<std::complex<double>> a_values(32768), b_values(32768);
    for (int i = 0; i < 32768; ++i) {
      a_values[i] = {std::sin(i * 0.7 + 0.1) / 8, mamba2 ? std::cos(i * 0.3) / 16 : 0};
      b_values[i] = {std::cos(i * 0.2 + 0.4) / 4, mamba2 ? std::sin(i * 0.1) / 32 : 0};
    }
    auto plain_a = cc->MakeCKKSPackedPlaintext(a_values);
    auto plain_b = cc->MakeCKKSPackedPlaintext(b_values);
    auto original_a = cc->Encrypt(keys.publicKey, plain_a);
    auto original_b = cc->Encrypt(keys.publicKey, plain_b);
    sync_gpu();
    fhemamba::OwnedArithmeticStats stats;
    int cases = 0;
    bool repeated_decrypt_bit_equal = true;
    for (const auto [la, lb] : {std::pair{0u, 0u}, {21u, 21u}, {21u, 24u}, {34u, 32u}})
      for (bool scaled : {false, true}) for (auto op : {Op::Add, Op::Subtract, Op::Multiply})
        for (int alias = 0; alias < 4; ++alias) {
          auto a = original_a->Clone(), b = original_b->Clone();
          a->SetLevel(la); b->SetLevel(lb);
          if (scaled) cc->EvalMultInPlace(a, 1.0);
          sync_gpu();
          if (alias >= 2) b = a;
          const auto before_a = snapshot(a), before_b = snapshot(b);
          const auto level_a = a->GetLevel(), level_b = b->GetLevel();
          auto left = a->Clone(), right = b->Clone();
          align(left, right);
          auto expected = op == Op::Add ? cc->EvalAdd(left, right) :
                          op == Op::Subtract ? cc->EvalSub(left, right) : cc->EvalMult(left, right);
          sync_gpu();
          auto input_a = alias == 0 || alias == 3 ? a->Clone() : a;
          auto input_b = alias == 3 ? input_a : alias == 2 ? a : b->Clone();
          auto actual = fhemamba::owned_ciphertext_binary(cc, std::move(input_a), std::move(input_b),
              op, align, sync_gpu, stats);
          if (!same_rns(snapshot(expected), snapshot(actual)) ||
              expected->GetLevel() != actual->GetLevel() ||
              expected->GetNoiseScaleDeg() != actual->GetNoiseScaleDeg())
            throw std::runtime_error("result or metadata mismatch in case " + std::to_string(cases));
          if (!same_rns(before_a, snapshot(a)) || !same_rns(before_b, snapshot(b)) ||
              a->GetLevel() != level_a || b->GetLevel() != level_b)
            throw std::runtime_error("live input changed in case " + std::to_string(cases));
          // OpenFHE's CKKS Decode adds random Gaussian noise. Record this
          // diagnostic, but gate exact ciphertext coefficients instead.
          if (cases == 0) repeated_decrypt_bit_equal = same_bits(decrypt(expected), decrypt(expected));
          ++cases;
        }
    if (cases != 96 || stats.calls != 96 || stats.reused_inputs != 96 || stats.cloned_inputs != 96)
      throw std::runtime_error("unexpected ownership dispatch count");
    fhemamba::OwnedArithmeticStats square_stats;
    int square_cases = 0;
    for (const auto level : {0u, 21u, 34u, 39u})
      for (int degree_case = 0; degree_case < 3; ++degree_case)
        for (int alias = 0; alias < 3; ++alias) {
          auto a = original_a->Clone();
          a->SetLevel(level);
          if (degree_case == 1) cc->EvalMultInPlace(a, 0.375);
          if (degree_case == 2) a = cc->EvalMult(a, a);
          sync_gpu();
          const auto before = snapshot(a);
          auto left = a->Clone(), right = a->Clone();
          align(left, right);
          auto expected = cc->EvalMult(left, right);
          // Cover both the allocating Mamba-2 and in-place Mamba-3 baselines.
          auto mutable_left = a->Clone(), mutable_right = a->Clone();
          cc->EvalMultMutableInPlace(mutable_left, mutable_right);
          auto direct = cc->EvalMult(a, a);
          sync_gpu();
          auto value = alias == 0 ? a : a->Clone();
          Ct live_alias = alias == 2 ? value : Ct{};
          auto actual = fhemamba::owned_ciphertext_square(cc, std::move(value), sync_gpu, square_stats);
          const auto reference = snapshot(expected);
          if (!same_rns(reference, snapshot(actual)) ||
              !same_rns(reference, snapshot(mutable_left)) ||
              !same_rns(reference, snapshot(direct)) ||
              expected->GetLevel() != actual->GetLevel() ||
              expected->GetNoiseScaleDeg() != actual->GetNoiseScaleDeg())
            throw std::runtime_error("square result or metadata mismatch in case " + std::to_string(square_cases));
          if (!same_rns(before, snapshot(a)) || (live_alias && !same_rns(before, snapshot(live_alias))))
            throw std::runtime_error("square live input changed in case " + std::to_string(square_cases));
          ++square_cases;
        }
    if (square_cases != 36 || square_stats.calls != 36 ||
        square_stats.reused_inputs != 12 || square_stats.cloned_inputs != 24)
      throw std::runtime_error("unexpected square dispatch count");
    std::ofstream out(argv[1]);
    out << "{\"passed\":true,\"cases\":" << cases
        << ",\"exact_rns_results\":true,\"live_inputs_unchanged\":true"
        << ",\"repeated_decrypt_bit_equal\":" << (repeated_decrypt_bit_equal ? "true" : "false")
        << ",\"reused_inputs\":" << stats.reused_inputs << ",\"cloned_inputs\":" << stats.cloned_inputs
        << ",\"square_cases\":" << square_cases
        << ",\"square_exact_rns_results\":true,\"square_live_inputs_unchanged\":true"
        << ",\"square_reused_inputs\":" << square_stats.reused_inputs
        << ",\"square_cloned_inputs\":" << square_stats.cloned_inputs
        << ",\"mamba2\":" << (mamba2 ? "true" : "false")
        << ",\"ring_dimension\":65536,\"depth\":44,\"scale_bits\":59,\"security\":\"not-set\"}\n";
    std::cout << "passed cases=" << cases << std::endl;
  } catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 2; }
}
