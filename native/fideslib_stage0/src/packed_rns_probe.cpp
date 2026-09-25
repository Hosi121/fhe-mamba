// Qualification of compact host coefficients: exact residues, fallback and
// encrypted arithmetic. Verification is outside the preparation timers.
#include "fideslib_plaintext_encoder.hpp"
#include "plaintext_rns.hpp"
#include <cuda_runtime_api.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace fideslib;
using Clock = std::chrono::steady_clock;
static void sync_gpu() { if (cudaDeviceSynchronize()) throw std::runtime_error("CUDA sync failed"); }
static double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static auto cpu(const Plaintext& p) -> const lbcrypto::Plaintext& {
  return std::any_cast<const lbcrypto::Plaintext&>(p->cpu);
}

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--mamba2"))
      throw std::invalid_argument("usage: packed_rns_probe OUTPUT [--mamba2]");
    const bool mamba2 = argc == 3;
    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetSecretKeyDist(mamba2 ? SPARSE_TERNARY : UNIFORM_TERNARY);
    params.SetCKKSDataType(mamba2 ? COMPLEX : REAL);
    params.SetRingDim(65536); params.SetBatchSize(32768);
    params.SetMultiplicativeDepth(44); params.SetScalingModSize(59); params.SetFirstModSize(60);
    params.SetScalingTechnique(FLEXIBLEAUTO); params.SetKeySwitchTechnique(HYBRID);
    params.SetNumLargeDigits(3); params.SetDevices({0});
    params.SetPlaintextAutoload(false); params.SetCiphertextAutoload(true);
    auto cc = GenCryptoContext(params);
    for (auto feature : {PKE, KEYSWITCH, LEVELEDSHE}) cc->Enable(feature);
    auto keys = cc->KeyGen(); cc->EvalMultKeyGen(keys.secretKey);
    cc->LoadContext(keys.publicKey); sync_gpu();
    auto cpu_context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(cc->cpu);
    const auto cp = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cpu_context->GetCryptoParameters());
    const auto q0 = cp->GetElementParams()->GetParams()[0]->GetModulus().ConvertToInt();
    int exact_cases = 0, compact_cases = 0, fallback_cases = 0;
    for (uint32_t slots : {32u, 32768u}) {
      fhemamba::PlaintextPreparation preparation(cc, slots,
          {.gpu_ntt = true, .move_coefficients = true, .borrow_upload = true, .gpu_rns = true});
      for (uint32_t level : {0u, 18u, 22u, 29u, 34u, 44u}) for (std::size_t degree : {1u, 2u}) {
        for (int pattern = 0; pattern < 10; ++pattern) {
          std::vector<double> values(slots);
          for (uint32_t i = 0; i < slots; ++i) {
            switch (pattern) {
              case 0: values[i] = i % 2 ? -0.0 : 0.0; break;
              case 1: values[i] = -0.125; break;
              case 2: values[i] = 0.25 * std::sin(i * 1.71 + 0.3); break;
              case 3: values[i] = i < slots / 4 ? 1 : 0; break;
              case 4: values[i] = i % 2 ? -0.499 : 0.499; break;
              case 5: values[i] = i % 2 ? -0.501 : 0.501; break;
              case 6: values[i] = 128 * std::sin(i * 1.71 + 0.3); break;
              case 7: values[i] = 1e-8 * std::cos(i + 0.25); break;
              case 8: values[i] = i == slots / 2 ? -4 : 0; break;
              case 9: values[i] = 1; break;
            }
          }
          auto reference = cc->MakeCKKSPackedPlaintext(values, degree, level, nullptr, slots);
          const bool expected_compact = slots == 32768 && degree == 1 && level < 44 &&
              fhemamba::compact_plaintext_fits(values, slots, cp->GetScalingFactorReal(level), q0);
          const auto prior = preparation.compact_rns_encodes;
          auto result = preparation.encode(values, level, degree);
          if ((preparation.compact_rns_encodes != prior) != expected_compact)
            throw std::runtime_error("compact dispatch mismatch");
          preparation.load(result);
          if (fhemamba::readback_plaintext(result) != cpu(reference)->GetElement<lbcrypto::DCRTPoly>() ||
              result->GetLevel() != reference->GetLevel() ||
              cpu(result)->GetScalingFactor() != cpu(reference)->GetScalingFactor() ||
              cpu(result)->GetNoiseScaleDeg() != degree || cpu(result)->GetSlots() != slots)
            throw std::runtime_error("RNS or metadata mismatch at slots=" + std::to_string(slots) +
                " level=" + std::to_string(level) + " degree=" + std::to_string(degree) +
                " pattern=" + std::to_string(pattern));
          ++exact_cases;
          expected_compact ? ++compact_cases : ++fallback_cases;
        }
      }
      std::cout << "exact_slots=" << slots << " cases=" << exact_cases << std::endl;
    }
    std::vector<double> input(32768);
    for (int i = 0; i < 32768; ++i) input[i] = std::sin(i * 0.71 + 0.1) / 8;
    auto input_plaintext = cc->MakeCKKSPackedPlaintext(input, 1, 0, nullptr, 32768);
    auto encrypted = cc->Encrypt(keys.publicKey, input_plaintext);
    double max_error = 0;
    int encrypted_cases = 0;
    fhemamba::PlaintextPreparation preparation(cc, 32768,
        {.gpu_ntt = true, .move_coefficients = true, .borrow_upload = true, .gpu_rns = true});
    for (uint32_t level : {0u, 18u, 34u}) {
      auto x = encrypted->Clone(); x->SetLevel(level);
      for (int operation = 0; operation < 3; ++operation) {
        std::vector<double> v(32768, operation == 2 ? 0.75 : -0.125);
        auto operand = operation == 2 ? cc->EvalMult(x, x) : x;
        long long reencodes = 0;
        auto p = operation == 0 ? preparation.encode(v, operand->GetLevel())
                              : preparation.encode_addend(operand, v, reencodes);
        preparation.load(p);
        auto result = operation == 0 ? cc->EvalMult(operand, p) : cc->EvalAdd(operand, p);
        sync_gpu();
        Plaintext output; cc->Decrypt(keys.secretKey, result, &output); output->SetLength(32768);
        const auto actual = output->GetRealPackedValue();
        for (int i = 0; i < 32768; ++i) {
          const double expected = operation == 0 ? input[i] * v[i] :
              operation == 1 ? input[i] + v[i] : input[i] * input[i] + v[i];
          if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite encrypted output");
          max_error = std::max(max_error, std::abs(actual[i] - expected));
        }
        ++encrypted_cases;
      }
    }
    struct Sample { int level, block, iteration; bool compact; double encode_ms, upload_ms; };
    std::vector<Sample> samples;
    for (uint32_t level : {18u, 26u, 34u}) for (int block = 0; block < 4; ++block) {
      const bool compact = block == 1 || block == 2;
      fhemamba::PlaintextPreparation timed(cc, 32768,
          {.gpu_ntt = true, .move_coefficients = true, .borrow_upload = true, .gpu_rns = compact});
      for (int iteration = -1; iteration < 8; ++iteration) {
        std::vector<double> v(32768);
        for (int i = 0; i < 32768; ++i) v[i] = 0.25 * std::cos(i * 0.27 + iteration + 2);
        const auto start = Clock::now();
        auto p = timed.encode(v, level);
        const double encode_ms = milliseconds(start);
        const auto upload = Clock::now(); timed.load(p); sync_gpu();
        const double upload_ms = milliseconds(upload);
        samples.push_back({static_cast<int>(level), block, iteration, compact, encode_ms, upload_ms});
      }
      if (compact && timed.compact_rns_uploads != 9) throw std::runtime_error("timed path not reached");
    }
    const bool passed = exact_cases == 240 && compact_cases > 0 && fallback_cases > 0 &&
                        encrypted_cases == 9 && max_error < 1e-6;
    std::ofstream out(argv[1]);
    if (!out) throw std::runtime_error("cannot write report");
    out << std::setprecision(15) << "{\"schema\":\"fhemamba-compact-rns-probe-v1\",\"passed\":"
        << (passed ? "true" : "false") << ",\"exact_rns_cases\":" << exact_cases
        << ",\"compact_cases\":" << compact_cases << ",\"fallback_cases\":" << fallback_cases
        << ",\"encrypted_cases\":" << encrypted_cases << ",\"max_abs_error\":" << max_error
        << ",\"tolerance\":1e-6,\"mamba2\":" << (mamba2 ? "true" : "false") << ",\"samples\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const auto& s = samples[i]; if (i) out << ',';
      out << "{\"level\":" << s.level << ",\"block\":" << s.block << ",\"iteration\":" << s.iteration
          << ",\"compact\":" << (s.compact ? "true" : "false") << ",\"encode_ms\":" << s.encode_ms
          << ",\"upload_ms\":" << s.upload_ms << '}';
    }
    out << "]}\n";
    std::cout << "exact_cases=" << exact_cases << " compact_cases=" << compact_cases
              << " fallback_cases=" << fallback_cases << " max_error=" << max_error << std::endl;
    return passed ? 0 : 1;
  } catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 2; }
}
