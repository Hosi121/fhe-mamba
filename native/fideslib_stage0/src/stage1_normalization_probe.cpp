// Isolated encrypted inverse-square-root probe; not the full model protocol.
#include <fideslib.hpp>

#include "stage1_normalization.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using namespace fideslib;
extern "C" int cudaDeviceSynchronize(void);

namespace {
using Clock = std::chrono::steady_clock;
using Ct = Ciphertext<DCRTPoly>;

void synchronize() {
  if (cudaDeviceSynchronize() != 0) throw std::runtime_error("CUDA synchronization failed");
}
auto elapsed(Clock::time_point start) -> double {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct EncryptedOps {
  CryptoContext<DCRTPoly> cc;
  int ct_ct = 0, ct_pt = 0, scalar_adds = 0, ct_adds = 0;
  std::vector<unsigned int> levels, scale_degrees;
  auto snapshot(const Ct& value) -> Ct { return value->Clone(); }
  auto scale(const Ct& x, double c) -> Ct {
    auto result = x->Clone();
    cc->EvalMultInPlace(result, c);
    ++ct_pt;
    return result;
  }
  auto scalar_add(const Ct& x, double c) -> Ct {
    ++scalar_adds;
    return cc->EvalAdd(x, c);
  }
  auto multiply(const Ct& a, const Ct& b) -> Ct {
    ++ct_ct;
    return cc->EvalMult(a, b);
  }
  auto add(const Ct& a, const Ct& b) -> Ct {
    ++ct_adds;
    return cc->EvalAdd(a, b);
  }
  void stage(const Ct& value) {
    levels.push_back(value->GetLevel());
    scale_degrees.push_back(value->GetNoiseScaleDeg());
    std::cout << "stage=" << levels.size() << " level=" << value->GetLevel()
              << " scale_degree=" << value->GetNoiseScaleDeg() << std::endl;
  }
};

struct Errors {
  double relative_exact = 0, absolute_polynomial = 0, relative_polynomial = 0;
  double worst_variance = 0;
  int non_finite = 0;
};

auto errors(const std::vector<double>& v, const std::vector<double>& polynomial,
            const std::vector<double>& actual) -> Errors {
  if (actual.size() < v.size()) throw std::runtime_error("short decrypted vector");
  Errors result;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (!std::isfinite(actual[i])) {
      ++result.non_finite;
      continue;
    }
    const double relative = std::abs(actual[i] * std::sqrt(v[i]) - 1);
    if (relative > result.relative_exact) {
      result.relative_exact = relative;
      result.worst_variance = v[i];
    }
    const double absolute = std::abs(actual[i] - polynomial[i]);
    result.absolute_polynomial = std::max(result.absolute_polynomial, absolute);
    result.relative_polynomial = std::max(
        result.relative_polynomial, absolute * std::sqrt(v[i]));
  }
  return result;
}

void write_errors(std::ostream& out, const Errors& e) {
  out << "{\"max_relative_error_vs_exact\":" << e.relative_exact
      << ",\"max_abs_error_vs_polynomial\":" << e.absolute_polynomial
      << ",\"max_normalized_error_vs_polynomial\":" << e.relative_polynomial
      << ",\"worst_variance\":" << e.worst_variance
      << ",\"non_finite\":" << e.non_finite << "}";
}

void write_list(std::ostream& out, const std::vector<unsigned int>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << values[i];
  out << ']';
}
}  // namespace

auto main(int argc, char* argv[]) -> int {
  try {
    if (argc != 12) {
      throw std::invalid_argument(
          "usage: stage1_normalization_probe OUTPUT RECIPE coupled|balanced|weighted DEPTH SCALE "
          "128-classic|not-set REPO_COMMIT BINARY_SHA256 RECIPE_SHA256 ERROR_TOLERANCE variance|normalize");
    }
    const std::string mode = argv[3], security = argv[6];
    const int depth = std::stoi(argv[4]), scale = std::stoi(argv[5]);
    const double tolerance = std::stod(argv[10]);
    const std::string input_mode = argv[11];
    const bool normalize = input_mode == "normalize";
    if ((mode != "coupled" && mode != "balanced" && mode != "weighted") ||
        (security != "128-classic" && security != "not-set") ||
        (input_mode != "variance" && !normalize) ||
        depth < 4 || depth > 44 || scale < 30 || scale > 59 ||
        !std::isfinite(tolerance) || !(0 < tolerance && tolerance < 1)) {
      throw std::invalid_argument(
          "unsupported probe parameters; pinned FIDESlib MAXP=64 bounds Q+P, "
          "so this HYBRID-3 probe conservatively caps depth at 44");
    }
    std::ifstream recipe_file(argv[2]);
    const auto recipe = fhemamba::stage1::read_normalization_schedule(recipe_file);
    constexpr int slots = 8192;
    const auto setup_start = Clock::now();
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(security == "128-classic" ? HEStd_128_classic : HEStd_NotSet);
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetCKKSDataType(REAL);
    // Let OpenFHE select a ring for the requested 128-bit parameters.
    if (security == "not-set") parameters.SetRingDim(65536);
    parameters.SetBatchSize(slots);
    parameters.SetMultiplicativeDepth(depth);
    parameters.SetScalingModSize(scale);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetNumLargeDigits(3);
    parameters.SetDevices({0});
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    auto cc = GenCryptoContext(parameters);
    std::cout << "ring=" << cc->GetRingDimension() << " depth=" << depth << std::endl;
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->LoadContext(keys.publicKey);
    synchronize();
    const double setup_seconds = elapsed(setup_start);
    std::cout << "context loaded" << std::endl;

    std::vector<double> values(slots), inputs(slots), expected(slots), expected_before(slots);
    fhemamba::stage1::ScalarNormalizationOps scalar_ops;
    for (int i = 0; i < slots; ++i) {
      values[i] = std::exp(std::log(recipe.lo) + (std::log(recipe.hi) - std::log(recipe.lo)) *
                          static_cast<double>(i) / (slots - 1));
    }
    values.front() = recipe.lo;
    values.back() = recipe.hi;
    for (int i = 0; i < slots; ++i) {
      inputs[i] = values[i];
      if (normalize) {
        const double root = std::sqrt(values[i] - recipe.lo);
        inputs[i] = std::nextafter(root, 0.0) * (i % 2 ? -1 : 1);
        values[i] = inputs[i] * inputs[i] + recipe.lo;
      }
      const auto result = fhemamba::stage1::evaluate_normalization(
          values[i], recipe, scalar_ops, mode == "balanced", mode == "weighted");
      expected[i] = result.output;
      expected_before[i] = result.before_final;
      if (!std::isfinite(expected[i]) || !std::isfinite(expected_before[i])) {
        throw std::runtime_error("nonfinite host reference");
      }
    }
    auto plain = cc->MakeCKKSPackedPlaintext(inputs);
    auto encrypted_source = cc->Encrypt(keys.publicKey, plain);
    synchronize();
    std::cout << "input encrypted" << std::endl;
    EncryptedOps ops{cc};
    const auto start = Clock::now();
    auto input = normalize
        ? ops.scalar_add(ops.multiply(encrypted_source, encrypted_source), recipe.lo)
        : encrypted_source;
    // No secret key is passed into this evaluator, and it never decrypts.
    const auto result = fhemamba::stage1::evaluate_normalization(
        input, recipe, ops, mode == "balanced", mode == "weighted");
    auto output = normalize ? ops.multiply(encrypted_source, result.output) : result.output;
    synchronize();
    const double evaluation_seconds = elapsed(start);
    auto decrypt = [&](const Ct& value) {
      Plaintext decoded;
      auto copy = value->Clone();
      cc->Decrypt(keys.secretKey, copy, &decoded);
      decoded->SetLength(slots);
      return decoded->GetRealPackedValue();
    };
    // Client-side diagnostics run only after encrypted evaluation completes.
    const auto actual_output = decrypt(output);
    const auto actual = normalize ? decrypt(result.output) : actual_output;
    const auto before = decrypt(result.before_final);
    const auto initial = decrypt(input);
    const auto final_errors = errors(values, expected, actual);
    const auto before_errors = errors(values, expected_before, before);
    const auto host_errors = errors(values, expected, expected);
    double normalized_error = 0, normalized_poly_error = 0;
    int normalized_non_finite = 0;
    if (normalize) {
      for (int i = 0; i < slots; ++i) {
        if (!std::isfinite(actual_output[i])) { ++normalized_non_finite; continue; }
        const double exact = inputs[i] / std::sqrt(values[i]);
        normalized_error = std::max(normalized_error, std::abs(actual_output[i] - exact));
        normalized_poly_error = std::max(
            normalized_poly_error, std::abs(actual_output[i] - inputs[i] * expected[i]));
      }
    }
    double input_error = 0;
    for (int i = 0; i < slots; ++i) {
      if (!std::isfinite(initial[i])) throw std::runtime_error("nonfinite decrypted input");
      input_error = std::max(input_error, std::abs(initial[i] - values[i]));
    }
    const bool passed = final_errors.non_finite == 0 && host_errors.non_finite == 0 &&
                        (normalize ? normalized_non_finite == 0 && normalized_error <= tolerance
                                   : final_errors.relative_exact <= tolerance);
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    std::ofstream out(argv[1]);
    if (!out) throw std::runtime_error("cannot write output artifact");
    out << std::setprecision(17)
        << "{\"stage\":\"normalization-ckks-probe\",\"version\":\"" << FHEMAMBA_VERSION
        << "\",\"repo_commit\":\"" << argv[7] << "\",\"binary_sha256\":\"" << argv[8]
        << "\",\"recipe_sha256\":\"" << argv[9]
        << "\",\"backend\":\"FIDESlib\",\"encrypted\":true,\"passed\":"
        << (passed ? "true" : "false") << ",\"status\":\"" << (passed ? "passed" : "failed")
        << "\",\"mode\":\"" << mode << "\",\"input_mode\":\"" << input_mode
        << "\",\"gate_metric\":\"" << (normalize ? "normalized-output-absolute-error" : "inverse-relative-error")
        << "\",\"parameters\":{\"ring_dim\":"
        << cc->GetRingDimension() << ",\"slots\":" << slots << ",\"depth\":" << depth
        << ",\"scale\":" << scale << ",\"security\":\"" << security
        << "\",\"secret_key_dist\":\"uniform-ternary\",\"scaling\":\"FLEXIBLEAUTO\","
           "\"key_switch\":\"HYBRID\",\"large_digits\":3,\"first_mod_size\":60}"
        << ",\"interval\":[" << recipe.lo << ',' << recipe.hi << ']'
        << ",\"error_threshold\":" << tolerance
        << ",\"measurements\":{\"final\":";
    write_errors(out, final_errors);
    out << ",\"before_final\":";
    write_errors(out, before_errors);
    out << ",\"host_reference\":";
    write_errors(out, host_errors);
    if (normalize) {
      out << ",\"normalized_output_max_abs_error\":" << normalized_error
          << ",\"normalized_output_polynomial_max_abs_error\":" << normalized_poly_error
          << ",\"normalized_output_non_finite\":" << normalized_non_finite;
    }
    out << ",\"input_max_abs_error\":" << input_error
        << ",\"smallest_input_abs_error\":" << std::abs(initial.front() - values.front())
        << ",\"ct_ct_products\":" << ops.ct_ct << ",\"ct_pt_products\":" << ops.ct_pt
        << ",\"scalar_additions\":" << ops.scalar_adds << ",\"ct_ct_additions\":" << ops.ct_adds
        << ",\"bootstraps\":0,\"rotations\":0,\"stage_levels\":";
    write_list(out, ops.levels);
    out << ",\"stage_scale_degrees\":";
    write_list(out, ops.scale_degrees);
    out << ",\"final_level\":" << output->GetLevel()
        << ",\"inverse_level\":" << result.output->GetLevel()
        << ",\"setup_seconds\":" << setup_seconds
        << ",\"evaluation_seconds\":" << evaluation_seconds
        << ",\"peak_rss_gib\":" << static_cast<double>(usage.ru_maxrss) / (1024 * 1024)
        << "},\"measurement_scope\":{\"artifact_level_report\":true,"
           "\"full_model_correctness_claimed\":false,\"geometric_grid_points\":8192,"
           "\"evaluation_decryptions\":0,\"post_evaluation_diagnostic_decryptions\":"
        << (normalize ? 4 : 3) << ",\"real_decoder_noise_enabled\":true,"
           "\"secret_key_in_benchmark_process\":true,\"bootstrap_error_measured\":false,"
           "\"full_vector_rms_reduction_measured\":false,"
           "\"claim\":\"Isolated encrypted normalization; sampled precision, not a model or protocol gate.\"}}\n";
    std::cout << "mode=" << mode << " gate_error="
              << (normalize ? normalized_error : final_errors.relative_exact)
              << " level=" << output->GetLevel() << " seconds=" << evaluation_seconds
              << std::endl;
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 2;
  }
}
