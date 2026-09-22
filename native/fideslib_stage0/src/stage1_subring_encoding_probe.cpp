// Compare full-ring and subring NTTs for the same periodic coefficients.
#include "fideslib_periodic_encoder.hpp"
#include <future>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fideslib;
extern "C" int cudaDeviceSynchronize(void);

namespace {
constexpr int kSlots = 32768, kHeads = 24, kOffset = 3328, kPeriod = 32;
using Encoder = fhemamba::stage1::PeriodicPlaintextEncoder;
using Clock = std::chrono::steady_clock;
auto elapsed_ms(Clock::time_point start) -> double {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void synchronize_gpu() {
  if (cudaDeviceSynchronize() != 0) throw std::runtime_error("CUDA synchronization failed");
}
struct Sample {
  int level, block, iteration, packing_slots;
  bool subring;
  double coefficient_scale, encode_ms, multiply_ms, active_error, inactive_error;
};
}  // namespace

auto main(int argc, char* argv[]) -> int {
  try {
    if (argc != 5 || (std::string(argv[4]) != "abba" && std::string(argv[4]) != "baab"))
      throw std::invalid_argument(
          "usage: stage1_subring_encoding_probe OUTPUT REPO_COMMIT BINARY_SHA256 abba|baab");
    const std::string order(argv[4]);
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetSecretKeyDist(SPARSE_TERNARY);
    parameters.SetRingDim(2 * kSlots);
    parameters.SetBatchSize(kSlots);
    parameters.SetMultiplicativeDepth(44);
    parameters.SetScalingModSize(59);
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
    // Exact RNS equality is stronger than approximate decrypted agreement.
    std::size_t exact_cases = 0;
    auto require_equal = [&](const Plaintext& actual, const Plaintext& reference) {
      const auto& a = std::any_cast<const lbcrypto::Plaintext&>(actual->cpu);
      const auto& b = std::any_cast<const lbcrypto::Plaintext&>(reference->cpu);
      if (a->GetElement<lbcrypto::DCRTPoly>() != b->GetElement<lbcrypto::DCRTPoly>() ||
          a->GetScalingFactor() != b->GetScalingFactor() ||
          a->GetNoiseScaleDeg() != b->GetNoiseScaleDeg() ||
          a->GetLevel() != b->GetLevel() || a->GetSlots() != b->GetSlots())
        throw std::runtime_error("subring plaintext differs from standard RNS encoding");
    };
    for (uint32_t period : {1u, 2u, 8u, 32u, 64u, 128u}) {
      Encoder encoder(cc, period);
      for (uint32_t level : {0u, 1u, 21u, 26u, 34u, 44u}) {
        for (std::size_t degree : {1u, 2u}) {
          for (double scale : {0.0, 1e-14, 1e-8, 1e-4, 1.0, 128.0}) {
            std::vector<double> coefficients(period);
            for (uint32_t j = 0; j < period; ++j)
              coefficients[j] = scale * std::sin((j + 1) * 1.25);
            require_equal(encoder.encode(coefficients, level, degree),
                          cc->MakeCKKSPackedPlaintext(coefficients, degree, level, nullptr, period));
            ++exact_cases;
          }
        }
      }
    }
    Encoder encoder(cc, kPeriod);
    // Repeated concurrent encoding must not replace global full-ring NTT tables.
    std::vector<std::future<void>> workers;
    for (int worker = 0; worker < 8; ++worker) {
      workers.push_back(std::async(std::launch::async, [&, worker] {
        std::vector<double> row(kPeriod, (worker - 3) * 0.03125);
        require_equal(encoder.encode(row, 26), cc->MakeCKKSPackedPlaintext(row, 1, 26, nullptr, kPeriod));
      }));
    }
    for (auto& worker : workers) { worker.get(); ++exact_cases; }
#ifdef _OPENMP
    const auto previous_active_levels = omp_get_max_active_levels();
    encoder.encode(std::vector<double>(kPeriod, 1.0), 26);
    if (omp_get_max_active_levels() != previous_active_levels)
      throw std::runtime_error("periodic encoder changed the caller's OpenMP controls");
    bool rejected_tiny_input = false;
    try { encoder.encode(std::vector<double>(kPeriod, 1e-100), 26); }
    catch (const std::exception&) { rejected_tiny_input = true; }
    if (!rejected_tiny_input || omp_get_max_active_levels() != previous_active_levels)
      throw std::runtime_error("exception path failed to restore OpenMP controls");
#endif
    std::cerr << "exact_encoding_cases=" << exact_cases << '\n';
    auto keys = cc->KeyGen();
    cc->LoadContext(keys.publicKey);

    std::vector<double> values(kSlots, 0.0);
    for (int h = 0; h < kHeads; ++h) values[kOffset + h] = (h - 11.0) / 16.0;
    auto input = cc->MakeCKKSPackedPlaintext(values);
    auto encrypted = cc->Encrypt(keys.publicKey, input);
    std::vector<Sample> samples;
    double maximum_active_error = 0, maximum_inactive_error = 0;
    for (const int level : {21, 26, 34}) {
      auto operand = encrypted->Clone();
      operand->SetLevel(static_cast<uint32_t>(level));
      for (int block = 0; block < 4; ++block) {
        const int packing_slots = kPeriod;
        const bool subring = order[block] == 'b';
        for (int iteration = -2; iteration < 8; ++iteration) {
          // The same changing coefficient sequence is used for each mode.
          constexpr double scales[] = {1.0, 1e-4, 1e-8, 128.0};
          const double scale = scales[(iteration + 4) % 4];
          std::vector<double> coefficients(packing_slots, 0.0);
          std::vector<double> expected(kSlots, 0.0);
          for (int h = 0; h < kHeads; ++h) {
            const double coefficient = scale * std::sin((h + 1) * (iteration + 3.25));
            coefficients[(kOffset + h) % packing_slots] = coefficient;
            expected[kOffset + h] = values[kOffset + h] * coefficient;
          }
          synchronize_gpu();
          auto start = Clock::now();
          auto plain = subring ? encoder.encode(coefficients, level) : cc->MakeCKKSPackedPlaintext(
              coefficients, 1, static_cast<uint32_t>(level), nullptr,
              static_cast<uint32_t>(packing_slots));
          const double encode_ms = elapsed_ms(start);
          start = Clock::now();
          auto product = cc->EvalMult(operand, plain);
          synchronize_gpu();
          const double multiply_ms = elapsed_ms(start);

          // Decryption is outside both timers. Check all 32768 slots, including
          // the padding that would become nonzero for an unmasked operand.
          Plaintext decrypted;
          auto handle = product->Clone();
          cc->Decrypt(keys.secretKey, handle, &decrypted);
          decrypted->SetLength(kSlots);
          const auto actual = decrypted->GetRealPackedValue();
          double active_error = 0, inactive_error = 0;
          for (int slot = 0; slot < kSlots; ++slot) {
            if (!std::isfinite(actual[slot])) throw std::runtime_error("nonfinite decrypted output");
            const double error = std::abs(actual[slot] - expected[slot]);
            auto& maximum = slot >= kOffset && slot < kOffset + kHeads
                                ? active_error : inactive_error;
            maximum = std::max(maximum, error);
          }
          maximum_active_error = std::max(maximum_active_error, active_error);
          maximum_inactive_error = std::max(maximum_inactive_error, inactive_error);
          samples.push_back({level, block, iteration, packing_slots, subring, scale, encode_ms,
                             multiply_ms, active_error, inactive_error});
        }
        std::cerr << "level=" << level << " block=" << block
                  << " subring=" << subring << '\n';
      }
    }
    const bool passed = maximum_active_error < 1e-7 && maximum_inactive_error < 1e-7;
    std::ofstream out(argv[1]);
    if (!out) throw std::runtime_error("cannot write output");
    out << std::setprecision(15)
        << "{\"stage\":\"subring-encoding-probe\",\"version\":\"" << FHEMAMBA_VERSION
        << "\",\"repo_commit\":\"" << argv[2] << "\",\"binary_sha256\":\"" << argv[3]
        << "\",\"backend\":\"FIDESlib\",\"encrypted\":true,\"passed\":"
        << (passed ? "true" : "false") << ",\"status\":\"" << (passed ? "passed" : "failed")
        << "\",\"parameters\":{\"ring_dim\":65536,\"depth\":44,\"scale\":59,"
           "\"first_mod\":60,\"security\":\"not-set\",\"secret_key_dist\":\"sparse-ternary\","
           "\"head_count\":24,\"head_offset\":3328,\"order\":\"" << order << "\"}"
        << ",\"exact_encoding_cases\":" << exact_cases
        << ",\"exact_encoding_equal\":true"
        << ",\"max_active_error\":" << maximum_active_error
        << ",\"max_inactive_error\":" << maximum_inactive_error << ",\"samples\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const auto& s = samples[i];
      if (i) out << ',';
      out << "{\"level\":" << s.level << ",\"block\":" << s.block
          << ",\"iteration\":" << s.iteration << ",\"warmup\":"
          << (s.iteration < 0 ? "true" : "false") << ",\"packing_slots\":" << s.packing_slots
          << ",\"subring\":" << (s.subring ? "true" : "false")
          << ",\"coefficient_scale\":" << s.coefficient_scale << ",\"encode_ms\":" << s.encode_ms
          << ",\"upload_multiply_sync_ms\":" << s.multiply_ms
          << ",\"max_active_error\":" << s.active_error
          << ",\"max_inactive_error\":" << s.inactive_error << '}';
    }
    out << "],\"measurement_scope\":{\"artifact_level_report\":true,"
           "\"full_model_correctness_claimed\":false,\"claim\":"
           "\"Exact periodic plaintext encoding and masked encrypted multiplication; excludes "
           "masked polynomial basis construction, bootstrap, and model execution.\"}}\n";
    std::cout << "active_error=" << maximum_active_error
              << " inactive_error=" << maximum_inactive_error << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
