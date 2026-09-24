// Exact RNS and encrypted arithmetic gates for GPU plaintext NTT/upload.
#include "fideslib_plaintext_encoder.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace fideslib;
using Upload = fhemamba::PlaintextUploadMode;
using Clock = std::chrono::steady_clock;
static void synchronize_gpu() { if (cudaDeviceSynchronize()) throw std::runtime_error("CUDA sync failed"); }
static double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static auto cpu(const Plaintext& p) -> const lbcrypto::Plaintext& {
  return std::any_cast<const lbcrypto::Plaintext&>(p->cpu);
}
static void same(const Plaintext& a, const Plaintext& b) {
  auto poly = cpu(a)->GetElement<lbcrypto::DCRTPoly>();
  poly.SetFormat(::Format::EVALUATION);
  if (poly != cpu(b)->GetElement<lbcrypto::DCRTPoly>() ||
      cpu(a)->GetScalingFactor() != cpu(b)->GetScalingFactor() ||
      a->GetLevel() != b->GetLevel() || cpu(a)->GetNoiseScaleDeg() != cpu(b)->GetNoiseScaleDeg() ||
      cpu(a)->GetSlots() != cpu(b)->GetSlots()) throw std::runtime_error("CPU RNS/metadata mismatch");
}
struct Sample { int level, block, iteration, mode; double encode_ms, upload_ms, error; };

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--mamba2"))
      throw std::invalid_argument("usage: packed_plaintext_probe OUTPUT [--mamba2]");
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
    cc->LoadContext(keys.publicKey); synchronize_gpu();
    int exact_cases = 0, moved_cases = 0, borrowed_cases = 0;
    for (uint32_t slots : {1u, 32u, 1024u, 32768u}) {
      fhemamba::CoefficientPlaintextEncoder encoder(cc, slots);
      fhemamba::CoefficientPlaintextEncoder moving_encoder(cc, slots, true);
      for (uint32_t level : {0u, 21u, 34u, 44u}) for (std::size_t degree : {1u, 2u}) {
        for (int pattern = 0; pattern < 5; ++pattern) {
          std::vector<double> values(slots);
          for (uint32_t j = 0; j < slots; ++j)
            values[j] = pattern == 0 ? 0.0 : pattern == 1 ? 1.0 :
                pattern == 2 ? (j < slots / 2 + 1 ? -0.125 : 0) :
                pattern == 3 ? 128 * std::sin(j * 1.75 + 0.3) : 1e-8 * std::cos(j + 0.25);
          auto reference = cc->MakeCKKSPackedPlaintext(values, degree, level, nullptr, slots);
          auto coefficient = encoder.encode(values, level, degree);
          same(coefficient, reference);
          // Both the copy-only bridge and GPU NTT must exactly reproduce all
          // CPU evaluation residues, not merely decrypted approximations.
          fhemamba::load_plaintext(cc, reference);
          const auto expected = cpu(reference)->GetElement<lbcrypto::DCRTPoly>();
          auto moved = moving_encoder.encode(values, level, degree);
          same(moved, reference);
          if (fhemamba::load_plaintext(cc, moved, 16, Upload::Borrowed) != Upload::Borrowed ||
              fhemamba::readback_plaintext(moved) != expected)
            throw std::runtime_error("moved coefficient RNS mismatch");
          ++moved_cases;
          if (fhemamba::readback_plaintext(reference) != expected)
            throw std::runtime_error("GPU upload RNS mismatch");
          auto direct_reference = cc->MakeCKKSPackedPlaintext(values, degree, level, nullptr, slots);
          if (fhemamba::load_plaintext(cc, direct_reference, 16, Upload::Direct) != Upload::Direct ||
              fhemamba::readback_plaintext(direct_reference) != expected)
            throw std::runtime_error("direct evaluation-format upload mismatch");
          auto borrowed_reference = cc->MakeCKKSPackedPlaintext(values, degree, level, nullptr, slots);
          if (fhemamba::load_plaintext(cc, borrowed_reference, 16, Upload::Borrowed) != Upload::Borrowed ||
              fhemamba::readback_plaintext(borrowed_reference) != expected)
            throw std::runtime_error("borrowed evaluation-format upload mismatch");
          same(borrowed_reference, reference);  // Borrowing may not mutate the CPU polynomial.
          for (Upload mode : {Upload::Staged, Upload::Direct, Upload::Borrowed}) for (int batch : {1, 4, 16, 64}) {
            auto batched = encoder.encode(values, level, degree);
            if (fhemamba::load_plaintext(cc, batched, batch, mode) != mode)
              throw std::runtime_error("unexpected plaintext upload fallback");
            if (fhemamba::readback_plaintext(batched) != expected)
              throw std::runtime_error("GPU RNS mismatch: slots=" + std::to_string(slots) +
                  " level=" + std::to_string(level) + " degree=" + std::to_string(degree) +
                  " batch=" + std::to_string(batch));
            same(batched, reference);
          }
          borrowed_cases += 2;  // Both formats; coefficient input also checks all four NTT widths.
          ++exact_cases;
        }
      }
      std::cout << "exact_slots=" << slots << " cases=" << exact_cases << std::endl;
    }
    fhemamba::CoefficientPlaintextEncoder encoder(cc, 32768);
    fhemamba::CoefficientPlaintextEncoder moving_encoder(cc, 32768, true);
    std::vector<double> input(32768);
    for (int i = 0; i < 32768; ++i) input[i] = std::sin(i * 0.7 + 0.1) / 8;
    auto p = cc->MakeCKKSPackedPlaintext(input, 1, 0, nullptr, 32768);
    auto encrypted = cc->Encrypt(keys.publicKey, p); synchronize_gpu();
    std::vector<Sample> samples;
    double max_error = 0;
    int shared_cases = 0;
    for (int mode : {0, 1, 2, 3, 4, 5, 6, 7, 8}) {
      const bool gpu_ntt = mode == 2 || mode == 4;
      const bool move_coefficients = mode == 5 || mode == 6 || mode == 8;
      const bool direct_upload = mode == 3 || mode == 4 || mode == 6;
      const bool borrow_upload = mode >= 7;
      // Mode 5 checks that moving coefficients implies GPU NTT and fast upload
      // even without either explicit option. Periodic values still use CPU NTT.
      fhemamba::PlaintextPreparation preparation(
          cc, 32768, {.fast_upload = mode >= 1 && mode <= 6 && mode != 5,
                      .gpu_ntt = gpu_ntt, .direct_upload = direct_upload,
                      .move_coefficients = move_coefficients, .borrow_upload = borrow_upload});
      preparation.enable_periodic_encoding(32);
      for (uint32_t level : {0u, 21u, 26u, 34u, 44u}) {
        std::vector<double> periodic(32);
        for (int j = 0; j < 32; ++j) periodic[j] = std::sin(j + 0.25);
        for (std::size_t degree : {1u, 2u}) {
          auto reference = cc->MakeCKKSPackedPlaintext(periodic, degree, level, nullptr, 32);
          auto compact = preparation.encode(periodic, level, degree, 32);
          same(compact, reference);
          preparation.load(compact);
          cc->LoadPlaintext(compact);
          if (fhemamba::readback_plaintext(compact) != cpu(reference)->GetElement<lbcrypto::DCRTPoly>())
            throw std::runtime_error("shared periodic upload differs");
          ++shared_cases;
        }
        if (level == 44) continue;
        auto operand = encrypted->Clone(); operand->SetLevel(level);
        auto square = cc->EvalMult(operand, operand); synchronize_gpu();
        std::vector<double> values(32768, 0.125);
        long long reencodes = 0;
        // Exercise both newly encoded and previously cached addends. The
        // latter must repair degree 1 -> 2 through the same shared policy.
        for (bool cached : {false, true}) {
          auto addend = cached ? preparation.align_addend(square,
              preparation.encode(values, square->GetLevel()), values, reencodes)
              : preparation.encode_addend(square, values, reencodes);
          preparation.load(addend);
          auto sum = cc->EvalAdd(square, addend); synchronize_gpu();
          Plaintext output; cc->Decrypt(keys.secretKey, sum, &output);
          output->SetLength(32768);
          const auto actual = output->GetRealPackedValue();
          for (int i = 0; i < 32768; ++i) {
            if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite shared addend output");
            max_error = std::max(max_error, std::abs(actual[i] - input[i] * input[i] - values[i]));
          }
          ++shared_cases;
        }
      }
      if (preparation.subring_encodes != 10 ||
          (mode == 0 ? preparation.fast_uploads != 0 : preparation.fast_uploads == 0) ||
          ((gpu_ntt || move_coefficients) ? preparation.gpu_ntt_encodes == 0 : preparation.gpu_ntt_encodes != 0) ||
          ((direct_upload || borrow_upload) ? preparation.direct_uploads != preparation.fast_uploads : preparation.direct_uploads != 0) ||
          (borrow_upload ? preparation.borrowed_uploads != preparation.fast_uploads : preparation.borrowed_uploads != 0) ||
          (move_coefficients ? preparation.moved_coefficient_encodes != preparation.gpu_ntt_encodes :
                               preparation.moved_coefficient_encodes != 0))
        throw std::runtime_error("shared plaintext policy dispatch failed");
    }
    const std::vector<int> order{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    const std::vector<int> batches{1, 4, 16, 64};
    for (int level : {21, 26, 34}) {
      auto operand = encrypted->Clone(); operand->SetLevel(level); synchronize_gpu();
      for (int block = 0; block < static_cast<int>(order.size()); ++block) {
        const auto mode = order[block];
        for (int iteration = -1; iteration < 4; ++iteration) {
          std::vector<double> values(32768);
          for (int i = 0; i < 32768; ++i) values[i] = std::cos(i * 0.27 + iteration + 2);
          auto start = Clock::now();
          auto& selected_encoder = mode >= 8 ? moving_encoder : encoder;
          auto plain = (mode >= 2 && mode <= 5) || mode >= 7 ? selected_encoder.encode(values, level) :
              cc->MakeCKKSPackedPlaintext(values, 1, level, nullptr, 32768);
          const auto encode_ms = ms(start);
          start = Clock::now();
          if (mode == 0) cc->LoadPlaintext(plain);
          else fhemamba::load_plaintext(cc, plain, mode >= 7 ? 16 :
              mode >= 2 && mode <= 5 ? batches[mode - 2] : 1,
              mode == 9 ? Upload::Borrowed : mode >= 6 ? Upload::Direct : Upload::Staged);
          synchronize_gpu(); const auto upload_ms = ms(start);
          auto result = cc->EvalMult(operand, plain); synchronize_gpu();
          Plaintext output; cc->Decrypt(keys.secretKey, result, &output);
          output->SetLength(32768);
          auto actual = output->GetRealPackedValue();
          double error = 0;
          for (int i = 0; i < 32768; ++i) {
            if (!std::isfinite(actual[i])) throw std::runtime_error("nonfinite output");
            error = std::max(error, std::abs(actual[i] - input[i] * values[i]));
          }
          max_error = std::max(max_error, error);
          samples.push_back({level, block, iteration, mode, encode_ms, upload_ms, error});
        }
      }
    }
    const bool passed = exact_cases == 160 && moved_cases == 160 && borrowed_cases == 320 &&
                        shared_cases == 162 && max_error < 1e-6;
    std::ofstream report(argv[1]);
    if (!report) throw std::runtime_error("cannot open output report");
    report << std::setprecision(15) << "{\"schema\":\"fhemamba-plaintext-ntt-probe-v1\","
        << "\"passed\":" << (passed ? "true" : "false") << ",\"encrypted\":true,"
        << "\"exact_rns_cases\":" << exact_cases << ",\"shared_policy_cases\":" << shared_cases
        << ",\"moved_coefficient_cases\":" << moved_cases
        << ",\"borrowed_rns_cases\":" << borrowed_cases
        << ",\"secret_key_distribution\":\"" << (mamba2 ? "sparse-ternary" : "uniform-ternary") << '"'
        << ",\"ckks_data_type\":\"" << (mamba2 ? "complex" : "real") << '"'
        << ",\"max_abs_error\":" << max_error
        << ",\"tolerance\":1e-6,\"ring_dimension\":65536,\"depth\":44,\"scale_bits\":59,"
        << "\"security\":\"not-set\",\"ntt_batches\":[1,4,16,64],"
        << "\"modes\":[\"standard\",\"fast-upload\",\"gpu-ntt-1\",\"gpu-ntt-4\","
        << "\"gpu-ntt-16\",\"gpu-ntt-64\",\"direct-upload\",\"direct-gpu-ntt-16\","
        << "\"direct-gpu-ntt-16-move\",\"borrowed-gpu-ntt-16-move\"],\"samples\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const auto& s = samples[i]; if (i) report << ',';
      report << "{\"level\":" << s.level << ",\"block\":" << s.block
          << ",\"iteration\":" << s.iteration << ",\"mode\":" << s.mode
          << ",\"encode_ms\":" << s.encode_ms << ",\"upload_ms\":" << s.upload_ms
          << ",\"max_abs_error\":" << s.error << '}';
    }
    report << "]}\n";
    std::cout << "exact_cases=" << exact_cases << " max_error=" << max_error << std::endl;
    return passed ? 0 : 1;
  } catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 2; }
}
