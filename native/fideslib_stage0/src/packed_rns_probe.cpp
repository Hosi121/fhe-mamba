// Qualification of compact host coefficients: exact residues, fallback and
// encrypted arithmetic. Verification is outside the preparation timers.
#include "fideslib_plaintext_encoder.hpp"
#include "fideslib_security.hpp"
#include "plaintext_rns.hpp"
#include "bounded_prefetch.hpp"
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

static auto pattern_values(uint32_t slots, int pattern) -> std::vector<double> {
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
  return values;
}

int main(int argc, char** argv) {
  try {
    if (argc < 2)
      throw std::invalid_argument("usage: packed_rns_probe OUTPUT [--mamba2] [--ring N] [--classical128] [--security-digits N] [--strategy limb|batch|fused] [--prefetch] [--compact-addends] [--prefetch-workers 1|2]");
    bool mamba2 = false, prefetch = false, classical128 = false;
    bool compact_addends = false, gpu_fft = false;
    int security_digits = 4;
    int prefetch_workers = 1;
    uint32_t ring = 65536;
    std::string strategy_name = "limb";
    for (int i = 2; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--mamba2") mamba2 = true;
      else if (option == "--classical128") { classical128 = true; ring = 131072; }
      else if (option == "--security-digits" && i + 1 < argc) security_digits = std::stoi(argv[++i]);
      else if (option == "--prefetch") prefetch = true;
      else if (option == "--compact-addends") compact_addends = true;
      else if (option == "--gpu-fft") gpu_fft = true;
      else if (option == "--prefetch-workers" && i + 1 < argc) prefetch_workers = std::stoi(argv[++i]);
      else if (option == "--ring" && i + 1 < argc) ring = std::stoul(argv[++i]);
      else if (option == "--strategy" && i + 1 < argc) strategy_name = argv[++i];
      else throw std::invalid_argument("unknown probe option: " + option);
    }
    if ((!classical128 && ring != 32768 && ring != 65536) || (classical128 && ring != 131072))
      throw std::invalid_argument("unsupported probe ring");
    if (prefetch_workers < 1 || prefetch_workers > 2 || (prefetch_workers != 1 && !prefetch))
      throw std::invalid_argument("prefetch workers must be 1 or 2 and require prefetch");
    auto strategy = fhemamba::CompactRnsStrategy::PerLimb;
    if (strategy_name == "batch") strategy = fhemamba::CompactRnsStrategy::Batched;
    else if (strategy_name == "fused") strategy = fhemamba::CompactRnsStrategy::FusedNtt;
    else if (strategy_name != "limb") throw std::invalid_argument("unsupported probe strategy");
    if (gpu_fft && (prefetch || strategy_name != "limb"))
      throw std::invalid_argument("GPU FFT probe requires serial per-limb preparation");
    const uint32_t full_slots = ring / 2;
    const bool advanced = strategy != fhemamba::CompactRnsStrategy::PerLimb;
    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecurityLevel(classical128 ? HEStd_128_classic : HEStd_NotSet);
    params.SetSecretKeyDist(mamba2 && !classical128 ? SPARSE_TERNARY : UNIFORM_TERNARY);
    params.SetCKKSDataType(mamba2 ? COMPLEX : REAL);
    params.SetRingDim(ring); params.SetBatchSize(full_slots);
    params.SetMultiplicativeDepth(44); params.SetScalingModSize(59); params.SetFirstModSize(60);
    params.SetScalingTechnique(FLEXIBLEAUTO); params.SetKeySwitchTechnique(HYBRID);
    params.SetNumLargeDigits(classical128 ? security_digits : 3); params.SetDevices({0});
    params.SetPlaintextAutoload(false); params.SetCiphertextAutoload(true);
    auto cc = GenCryptoContext(params);
    const auto security_audit = fhemamba::audit_ckks_context(cc);
    if (classical128) security_audit.require_classical128();
    for (auto feature : {PKE, KEYSWITCH, LEVELEDSHE}) cc->Enable(feature);
    auto keys = cc->KeyGen(); cc->EvalMultKeyGen(keys.secretKey);
    cc->LoadContext(keys.publicKey); sync_gpu();
    auto cpu_context = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(cc->cpu);
    const auto cp = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cpu_context->GetCryptoParameters());
    const auto q0 = cp->GetElementParams()->GetParams()[0]->GetModulus().ConvertToInt();
    int exact_cases = 0, compact_cases = 0, fallback_cases = 0, compact_addend_cases = 0;
    long long batched_uploads = 0, fused_uploads = 0;
    long long gpu_fft_cases = 0;
    for (uint32_t slots : {32u, full_slots}) {
      fhemamba::PlaintextPreparation preparation(cc, slots,
          {.gpu_ntt = true, .move_coefficients = true, .borrow_upload = true, .gpu_rns = true,
           .gpu_addend_rns = compact_addends, .gpu_fft = gpu_fft,
           .rns_strategy = strategy});
      const std::vector<std::size_t> degrees = compact_addends ? std::vector<std::size_t>{1,2,3}
                                                             : std::vector<std::size_t>{1,2};
      for (uint32_t level : {0u, 18u, 22u, 29u, 34u, 44u}) for (std::size_t degree : degrees) {
        std::unique_ptr<fhemamba::BoundedPrefetch<lbcrypto::Plaintext>> queue;
        if (prefetch) queue = std::make_unique<fhemamba::BoundedPrefetch<lbcrypto::Plaintext>>(
            10, 2, [&](std::size_t pattern) {
          return preparation.cpu_encoder().encode_cpu(pattern_values(slots, pattern), level, degree);
        }, prefetch_workers);
        for (int pattern = 0; pattern < 10; ++pattern) {
          auto values = pattern_values(slots, pattern);
          auto reference = cc->MakeCKKSPackedPlaintext(values, degree, level, nullptr, slots);
          const bool expected_compact = slots == full_slots &&
              (degree == 1 || (compact_addends && degree == 2)) && level < 44 &&
              fhemamba::compact_plaintext_fits(values, slots, cp->GetScalingFactorReal(level), q0);
          const auto prior = preparation.compact_rns_encodes;
          auto result = queue ? preparation.adopt_cpu(queue->pop()) : preparation.encode(values, level, degree);
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
          if (expected_compact && degree == 2) ++compact_addend_cases;
        }
      }
      std::cout << "exact_slots=" << slots << " cases=" << exact_cases << std::endl;
      batched_uploads += preparation.batched_rns_uploads;
      fused_uploads += preparation.fused_rns_uploads;
      gpu_fft_cases += preparation.gpu_fft_encodes;
    }
    std::vector<double> input(full_slots);
    for (uint32_t i = 0; i < full_slots; ++i) input[i] = std::sin(i * 0.71 + 0.1) / 8;
    auto input_plaintext = cc->MakeCKKSPackedPlaintext(input, 1, 0, nullptr, full_slots);
    auto encrypted = cc->Encrypt(keys.publicKey, input_plaintext);
    double max_error = 0;
    int encrypted_cases = 0;
    fhemamba::PlaintextPreparation preparation(cc, full_slots,
        {.gpu_ntt = true, .move_coefficients = true, .borrow_upload = true, .gpu_rns = true,
         .gpu_addend_rns = compact_addends, .gpu_fft = gpu_fft,
         .rns_strategy = strategy});
    for (uint32_t level : {0u, 18u, 34u}) {
      auto x = encrypted->Clone(); x->SetLevel(level);
      for (int operation = 0; operation < 3; ++operation) {
        std::vector<double> v(full_slots, operation == 2 ? 0.75 : -0.125);
        auto operand = operation == 2 ? cc->EvalMult(x, x) : x;
        long long reencodes = 0;
        auto p = operation == 0 ? preparation.encode(v, operand->GetLevel())
                              : preparation.encode_addend(operand, v, reencodes);
        preparation.load(p);
        auto result = operation == 0 ? cc->EvalMult(operand, p) : cc->EvalAdd(operand, p);
        sync_gpu();
        Plaintext output; cc->Decrypt(keys.secretKey, result, &output); output->SetLength(full_slots);
        const auto actual = output->GetRealPackedValue();
        for (uint32_t i = 0; i < full_slots; ++i) {
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
      fhemamba::PlaintextPreparation timed(cc, full_slots,
          {.gpu_ntt = true, .move_coefficients = true, .borrow_upload = true,
           .gpu_rns = advanced || compact || gpu_fft, .gpu_fft = gpu_fft && compact,
           .rns_strategy = compact ? strategy : fhemamba::CompactRnsStrategy::PerLimb});
      for (int iteration = -1; iteration < 8; ++iteration) {
        std::vector<double> v(full_slots);
        for (uint32_t i = 0; i < full_slots; ++i) v[i] = 0.25 * std::cos(i * 0.27 + iteration + 2);
        const auto start = Clock::now();
        auto p = timed.encode(v, level);
        const double encode_ms = milliseconds(start);
        const auto upload = Clock::now(); timed.load(p); sync_gpu();
        const double upload_ms = milliseconds(upload);
        samples.push_back({static_cast<int>(level), block, iteration, compact, encode_ms, upload_ms});
      }
      if (compact && timed.compact_rns_uploads != 9) throw std::runtime_error("timed path not reached");
    }
    const bool passed = exact_cases == (compact_addends ? 360 : 240) &&
                        (!compact_addends || compact_addend_cases > 0) && compact_cases > 0 && fallback_cases > 0 &&
                        encrypted_cases == 9 && max_error < 1e-6 &&
                        (!gpu_fft || gpu_fft_cases == compact_cases) &&
                        (strategy != fhemamba::CompactRnsStrategy::Batched || batched_uploads > 0) &&
                        (strategy != fhemamba::CompactRnsStrategy::FusedNtt || fused_uploads > 0);
    std::ofstream out(argv[1]);
    if (!out) throw std::runtime_error("cannot write report");
    out << std::setprecision(15) << "{\"schema\":\"fhemamba-compact-rns-probe-v1\",\"passed\":"
        << (passed ? "true" : "false") << ",\"exact_rns_cases\":" << exact_cases
        << ",\"compact_cases\":" << compact_cases << ",\"fallback_cases\":" << fallback_cases
        << ",\"compact_addend_cases\":" << compact_addend_cases
        << ",\"gpu_fft\":" << (gpu_fft ? "true" : "false")
        << ",\"gpu_fft_cases\":" << gpu_fft_cases
        << ",\"encrypted_cases\":" << encrypted_cases << ",\"max_abs_error\":" << max_error
        << ",\"prefetch\":" << (prefetch ? "true" : "false")
        << ",\"prefetch_workers\":" << prefetch_workers
        << ",\"ring_dimension\":" << ring << ",\"rns_strategy\":\"" << strategy_name << "\""
        << ",\"batched_uploads\":" << batched_uploads << ",\"fused_uploads\":" << fused_uploads
        << ",\"timing_baseline\":\"" << (advanced || gpu_fft ? "compact-per-limb" : "full-rns-host") << "\""
        << ",\"tolerance\":1e-6,\"mamba2\":" << (mamba2 ? "true" : "false")
        << ",\"security\":\"" << (classical128 ? "128-classic" : "not-set") << "\",\"security_audit\":";
    if (classical128) security_audit.write_json(out); else out << "null";
    out << ",\"samples\":[";
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
