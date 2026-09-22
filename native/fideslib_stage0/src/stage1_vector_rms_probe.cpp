// Packed vector RMSNorm, with actual reductions, learned gamma and optional refresh.
#include <fideslib.hpp>

#include "stage1_normalization.hpp"
#include "stage1_vector_rms.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sys/resource.h>

using namespace fideslib;
extern "C" int cudaDeviceSynchronize(void);

namespace {
using Ct = Ciphertext<DCRTPoly>;
using Clock = std::chrono::steady_clock;
void synchronize_gpu() {
  if (cudaDeviceSynchronize() != 0) throw std::runtime_error("CUDA synchronization failed");
}
auto seconds(Clock::time_point start) -> double {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
struct Ops {
  CryptoContext<DCRTPoly> cc;
  int ct_ct = 0, ct_pt = 0, adds = 0, rotations = 0;
  auto snapshot(const Ct& x) -> Ct { return x->Clone(); }
  auto multiply(const Ct& a, const Ct& b) -> Ct { ++ct_ct; return cc->EvalMult(a, b); }
  auto add(const Ct& a, const Ct& b) -> Ct { ++adds; return cc->EvalAdd(a, b); }
  auto scalar_add(const Ct& a, double b) -> Ct { ++adds; return cc->EvalAdd(a, b); }
  auto scale(const Ct& x, double s) -> Ct {
    auto y = x->Clone(); ++ct_pt; cc->EvalMultInPlace(y, s); return y;
  }
  auto rotate(const Ct& x, int shift) -> Ct { ++rotations; return cc->EvalRotate(x, shift); }
  auto gamma(const Ct& x, const std::vector<double>& weights) -> Ct {
    auto p = cc->MakeCKKSPackedPlaintext(weights, 1, x->GetLevel(), nullptr, weights.size());
    ++ct_pt; return cc->EvalMult(x, p);
  }
  void stage(const Ct&) {}
};
struct Measurement {
  double absolute = 0, polynomial = 0;
  int non_finite = 0;
  void add(double actual, double exact, double reference) {
    if (!std::isfinite(actual)) { ++non_finite; return; }
    absolute = std::max(absolute, std::abs(actual - exact));
    polynomial = std::max(polynomial, std::abs(actual - reference));
  }
  void write(std::ostream& out) const {
    out << "{\"max_abs_error\":" << absolute
        << ",\"max_abs_error_vs_polynomial\":" << polynomial
        << ",\"non_finite\":" << non_finite << '}';
  }
};
struct RefreshEvent {
  int stage = -1, input_level = -1, output_level = -1;
  int ct_pt = 0, alignments = 0, bootstraps = 0;
  double elapsed = 0, bound = 0;
  double before_max = 0, after_max = 0, difference = 0;
  int diagnostic_non_finite = 0;
  Ct residual, before, after;
};

// Every alignment is a real arithmetic operation; never relabel ciphertext
// metadata. This helper is used for both internal and output refreshes.
auto refresh_value(Ops& ops, const Ct& value, double bound,
                   const std::vector<double>& scales, const std::vector<double>& inverse,
                   bool meta, int alpha, int iterations, int depth, RefreshEvent& event) -> Ct {
  const auto start = Clock::now();
  const int initial_products = ops.ct_pt;
  event.input_level = value->GetLevel();
  event.bound = bound;
  event.before = value->Clone();
  auto bounded = scales.empty() ? ops.scale(value, 1.0 / bound) : ops.gamma(value, inverse);
  while (bounded->GetNoiseScaleDeg() > 1) ops.cc->RescaleInPlace(bounded);
  if (meta && bounded->GetLevel() + 1 > static_cast<unsigned>(depth))
    throw std::invalid_argument("Meta-BTS needs a live level for residual amplification");
  synchronize_gpu(); auto refreshed = ops.cc->EvalBootstrap(bounded); synchronize_gpu();
  ++event.bootstraps;
  for (int k = 1; meta && k < iterations; ++k) {
    auto first = refreshed->Clone();
    while (first->GetNoiseScaleDeg() > 1) ops.cc->RescaleInPlace(first);
    auto reference = bounded->Clone();
    const auto target = std::max(first->GetLevel(), reference->GetLevel());
    for (auto* x : {&first, &reference}) {
      while ((*x)->GetLevel() < target) {
        *x = ops.scale(*x, 1.0); ++event.alignments;
        while ((*x)->GetNoiseScaleDeg() > 1) ops.cc->RescaleInPlace(*x);
      }
    }
    auto residual = ops.cc->EvalSub(reference, first);
    residual = ops.scale(residual, std::ldexp(1.0, alpha * k));
    while (residual->GetNoiseScaleDeg() > 1) ops.cc->RescaleInPlace(residual);
    event.residual = residual->Clone();
    synchronize_gpu(); auto second = ops.cc->EvalBootstrap(residual); synchronize_gpu();
    ++event.bootstraps;
    refreshed = ops.add(refreshed, ops.scale(second, std::ldexp(1.0, -alpha * k)));
  }
  refreshed = scales.empty() ? ops.scale(refreshed, bound) : ops.gamma(refreshed, scales);
  synchronize_gpu();
  event.output_level = refreshed->GetLevel();
  event.ct_pt = ops.ct_pt - initial_products;
  event.elapsed = seconds(start);
  event.after = refreshed->Clone();
  return refreshed;
}
struct Batch {
  int first_case, cases, level, scale_degree, refreshed_level = -1;
  int ct_ct, ct_pt, adds, rotations;
  int refresh_ct_pt_products = 0, alignment_products = 0, physical_bootstraps = 0;
  double evaluation_seconds, refresh_seconds = 0;
  double meta_residual_max_abs = 0;
  int meta_residual_non_finite = 0;
  int input_level = 0, input_conditioning_products = 0;
  double input_error = 0;
  std::vector<RefreshEvent> internal;
  Measurement before, after;
};
}  // namespace

auto main(int argc, char* argv[]) -> int {
  try {
    if (argc < 15 || argc > 22) throw std::invalid_argument(
        "usage: vector-probe OUTPUT RECIPE MODE DEPTH SCALE SECURITY COMMIT BINARY_SHA "
        "RECIPE_SHA TOLERANCE vector FIXTURE before|after none|output|output-meta "
        "[META_ALPHA] [global|channel] [none|meta] [INPUT_LEVEL] [REFRESH_TRIGGER] "
        "[global|stage] [META_ITERATIONS]");
    const std::string mode = argv[3], security = argv[6], placement = argv[13], refresh = argv[14];
    const int depth = std::stoi(argv[4]), scale = std::stoi(argv[5]);
    const double tolerance = std::stod(argv[10]);
    const int meta_alpha = argc >= 16 ? std::stoi(argv[15]) : 12;
    const std::string coordinates = argc >= 17 ? argv[16] : "global";
    const std::string internal_refresh = argc >= 18 ? argv[17] : "none";
    const int input_level = argc >= 19 ? std::stoi(argv[18]) : 0;
    const int refresh_trigger = argc >= 20 ? std::stoi(argv[19]) : depth - 6;
    const std::string inverse_coordinates = argc >= 21 ? argv[20] : "global";
    const int meta_iterations = argc >= 22 ? std::stoi(argv[21]) : 2;
    if ((mode != "balanced" && mode != "weighted") ||
        (security != "128-classic" && security != "not-set") ||
        (placement != "before" && placement != "after") ||
        (refresh != "none" && refresh != "output" && refresh != "output-meta") ||
        (coordinates != "global" && coordinates != "channel") ||
        (internal_refresh != "none" && internal_refresh != "meta") ||
        (internal_refresh != "none" && mode != "balanced") ||
        (inverse_coordinates != "global" && inverse_coordinates != "stage") ||
        meta_iterations < 2 || meta_iterations > 3 ||
        input_level < 0 || input_level > std::max(0, depth - 8) ||
        (internal_refresh != "none" && (refresh_trigger < 24 || refresh_trigger > depth - 4)) ||
        meta_alpha < 0 || meta_alpha > 20 || std::string(argv[11]) != "vector" ||
        depth < 4 || depth > 44 || scale < 54 || scale > 59 ||
        !std::isfinite(tolerance) || tolerance <= 0 || tolerance >= 1)
      throw std::invalid_argument("unsupported vector RMS probe parameters");
    std::ifstream recipe_stream(argv[2]), fixture_stream(argv[12]);
    const auto recipe = fhemamba::stage1::read_normalization_schedule(recipe_stream);
    const auto fixture = fhemamba::stage1::read_vector_rms_fixture(fixture_stream);
    const fhemamba::stage1::VectorRmsLayout layout(fixture.width);
    const int slots = layout.slots;
    std::vector<double> variances(fixture.cases), inverses(fixture.cases);
    fhemamba::stage1::ScalarNormalizationOps host_ops;
    for (int c = 0; c < fixture.cases; ++c) {
      long double sum = 0;
      for (int j = 0; j < fixture.width; ++j) {
        const long double x = fixture.inputs[c * fixture.width + j]; sum += x * x;
      }
      variances[c] = static_cast<double>(sum / fixture.width + fixture.epsilon);
      if (!(variances[c] >= recipe.lo && variances[c] <= recipe.hi))
        throw std::invalid_argument("vector variance outside certified recipe domain");
      inverses[c] = fhemamba::stage1::evaluate_normalization(
          variances[c], recipe, host_ops, mode == "balanced", mode == "weighted").output;
    }
    double gamma_max = 0;
    for (double g : fixture.gamma) gamma_max = std::max(gamma_max, std::abs(g));
    // Public worst-case component bound, with 10% numerical headroom.
    const double refresh_bound = std::max(1.0, 1.1 * gamma_max * std::sqrt(fixture.width));
    const auto feature_scales = fhemamba::stage1::vector_rms_refresh_scales(fixture.gamma);
    const double padding_scale = gamma_max == 0 ? 1.0 : 1.1 * std::sqrt(fixture.width) * gamma_max / 64;
    std::vector<double> channel_scales(slots, padding_scale), channel_inverse(slots);
    for (int j = 0; j < fixture.width; ++j)
      for (int lane = 0; lane < layout.lanes; ++lane)
        channel_scales[layout.slot(j, lane)] = feature_scales[j];
    for (int i = 0; i < slots; ++i) channel_inverse[i] = 1.0 / channel_scales[i];
    const auto setup_start = Clock::now();
    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecurityLevel(security == "128-classic" ? HEStd_128_classic : HEStd_NotSet);
    p.SetSecretKeyDist(UNIFORM_TERNARY); p.SetCKKSDataType(REAL);
    if (security == "not-set") p.SetRingDim(65536);
    p.SetBatchSize(slots); p.SetMultiplicativeDepth(depth);
    p.SetScalingModSize(scale); p.SetFirstModSize(60);
    p.SetScalingTechnique(FLEXIBLEAUTO); p.SetKeySwitchTechnique(HYBRID);
    p.SetNumLargeDigits(3); p.SetDevices({0});
    p.SetPlaintextAutoload(false); p.SetCiphertextAutoload(true);
    auto cc = GenCryptoContext(p);
    cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    const bool needs_bootstrap = refresh != "none" || internal_refresh != "none";
    if (needs_bootstrap) cc->Enable(FHE);
    auto keys = cc->KeyGen(); cc->EvalMultKeyGen(keys.secretKey);
    const auto rotation_indices = layout.rotations();
    cc->EvalRotateKeyGen(keys.secretKey, rotation_indices);
    if (needs_bootstrap) {
      cc->EvalBootstrapSetup({4, 4}, {0, 0}, slots, 0);
      cc->EvalBootstrapKeyGen(keys.secretKey, slots);
    }
    cc->LoadContext(keys.publicKey); synchronize_gpu();
    const double setup_seconds = seconds(setup_start);
    std::cout << "ring=" << cc->GetRingDimension() << " width=" << fixture.width
              << " lanes=" << layout.lanes << " setup_seconds=" << setup_seconds << std::endl;
    int decryption_failures = 0;
    auto decode = [&](const Ct& value) -> std::vector<double> {
      try {
        Plaintext decoded; auto copy = value->Clone();
        cc->Decrypt(keys.secretKey, copy, &decoded); decoded->SetLength(slots);
        return decoded->GetRealPackedValue();
      } catch (const std::exception& e) {
        ++decryption_failures;
        std::cerr << "post-evaluation decode failure: " << e.what() << std::endl;
        return std::vector<double>(slots, std::numeric_limits<double>::quiet_NaN());
      }
    };
    std::vector<double> packed_gamma(slots, 0);
    for (int j = 0; j < fixture.width; ++j)
      for (int lane = 0; lane < layout.lanes; ++lane)
        packed_gamma[layout.slot(j, lane)] = fixture.gamma[j];
    std::vector<Batch> batches;
    Measurement overall, overall_before, checkpoint, synthetic;
    double total_eval = 0, total_refresh = 0, max_padding = 0;
    int total_bootstraps = 0;
    for (int start = 0; start < fixture.cases; start += layout.lanes) {
      const int count = std::min(layout.lanes, fixture.cases - start);
      std::vector<double> inputs(slots, 0);
      for (int lane = 0; lane < count; ++lane)
        for (int j = 0; j < fixture.width; ++j)
          inputs[layout.slot(j, lane)] = fixture.inputs[(start + lane) * fixture.width + j];
      auto plaintext = cc->MakeCKKSPackedPlaintext(inputs);
      auto encrypted = cc->Encrypt(keys.publicKey, plaintext); synchronize_gpu();
      Ops ops{cc};
      Batch batch{}; batch.first_case = start; batch.cases = count;
      // Controlled depth pressure with real rescaling, not a prior-model layer.
      while (encrypted->GetLevel() < static_cast<unsigned>(input_level)) {
        encrypted = ops.scale(encrypted, 1.0); ++batch.input_conditioning_products;
        while (encrypted->GetNoiseScaleDeg() > 1) cc->RescaleInPlace(encrypted);
      }
      batch.input_level = encrypted->GetLevel();
      ops.ct_pt = 0;
      const auto begin = Clock::now();
      auto numerator = placement == "before" ? ops.gamma(encrypted, packed_gamma) : encrypted;
      auto sum = fhemamba::stage1::vector_rms_sum(ops.multiply(encrypted, encrypted), layout, ops);
      auto variance = ops.scalar_add(ops.scale(sum, 1.0 / fixture.width), fixture.epsilon);
      auto checkpoint_inverse = [&](const Ct& y, std::size_t stage) -> Ct {
        if (y->GetLevel() < static_cast<unsigned>(refresh_trigger)) return y;
        RefreshEvent event; event.stage = stage;
        const double bound = inverse_coordinates == "stage"
            ? fhemamba::stage1::normalization_inverse_bound(recipe, stage)
            : 1.0 / std::sqrt(recipe.lo);
        auto result = refresh_value(ops, y, 1.1 * bound, {}, {},
                                    true, meta_alpha, meta_iterations, depth, event);
        batch.internal.push_back(event);
        return result;
      };
      const auto inverse = internal_refresh == "meta"
          ? fhemamba::stage1::evaluate_balanced_normalization(variance, recipe, ops, checkpoint_inverse)
          : fhemamba::stage1::evaluate_normalization(
              variance, recipe, ops, mode == "balanced", mode == "weighted");
      auto output = ops.multiply(numerator, inverse.output);
      if (placement == "after") output = ops.gamma(output, packed_gamma);
      synchronize_gpu();
      batch.level = output->GetLevel(); batch.scale_degree = output->GetNoiseScaleDeg();
      batch.evaluation_seconds = seconds(begin);
      batch.ct_ct = ops.ct_ct; batch.ct_pt = ops.ct_pt; batch.adds = ops.adds;
      batch.rotations = ops.rotations;
      Ct refreshed, meta_residual;
      if (refresh != "none") {
        RefreshEvent event;
        refreshed = refresh_value(ops, output, refresh_bound,
            coordinates == "channel" ? channel_scales : std::vector<double>{},
            coordinates == "channel" ? channel_inverse : std::vector<double>{},
            refresh == "output-meta", meta_alpha, meta_iterations, depth, event);
        meta_residual = event.residual;
        batch.physical_bootstraps = event.bootstraps;
        batch.alignment_products = event.alignments;
        batch.refresh_ct_pt_products = event.ct_pt;
        batch.refreshed_level = refreshed->GetLevel();
        batch.refresh_seconds = event.elapsed;
      }
      // The evaluator above has no decryption. These are client diagnostics.
      const auto actual_before = decode(output);
      const auto actual = refresh != "none" ? decode(refreshed) : actual_before;
      for (auto& event : batch.internal) {
        const auto before = decode(event.before), after = decode(event.after);
        for (int j = 0; j < slots; ++j) {
          if (!std::isfinite(before[j]) || !std::isfinite(after[j])) {
            ++event.diagnostic_non_finite; continue;
          }
          event.before_max = std::max(event.before_max, std::abs(before[j]));
          event.after_max = std::max(event.after_max, std::abs(after[j]));
          event.difference = std::max(event.difference, std::abs(after[j] - before[j]));
        }
        event.before.reset(); event.after.reset(); event.residual.reset();
      }
      const auto carried_input = decode(encrypted);
      for (std::size_t j = 0; j < inputs.size(); ++j) {
        if (!std::isfinite(carried_input.at(j))) ++overall.non_finite;
        else batch.input_error = std::max(batch.input_error, std::abs(carried_input[j] - inputs[j]));
      }
      if (meta_residual) {
        for (double x : decode(meta_residual)) {
          if (!std::isfinite(x)) { ++batch.meta_residual_non_finite; ++overall.non_finite; }
          else batch.meta_residual_max_abs = std::max(batch.meta_residual_max_abs, std::abs(x));
        }
      }
      if (actual.size() < inputs.size() || actual_before.size() < inputs.size())
        throw std::runtime_error("short vector RMS decryption");
      for (int lane = 0; lane < count; ++lane) {
        const int c = start + lane;
        for (int j = 0; j < fixture.width; ++j) {
          const int slot = layout.slot(j, lane);
          const double base = inputs[slot] * fixture.gamma[j];
          const double exact = base / std::sqrt(variances[c]), poly = base * inverses[c];
          batch.before.add(actual_before[slot], exact, poly);
          batch.after.add(actual[slot], exact, poly);
          overall_before.add(actual_before[slot], exact, poly);
          overall.add(actual[slot], exact, poly);
          (c < 8 ? checkpoint : synthetic).add(actual[slot], exact, poly);
        }
      }
      for (int j = 0; j < layout.padded_width; ++j)
        for (int lane = 0; lane < layout.lanes; ++lane)
          if (j >= fixture.width || lane >= count) {
            const double x = actual[layout.slot(j, lane)];
            if (!std::isfinite(x)) ++overall.non_finite;
            else max_padding = std::max(max_padding, std::abs(x));
          }
      total_eval += batch.evaluation_seconds; total_refresh += batch.refresh_seconds;
      total_bootstraps += batch.physical_bootstraps;
      for (const auto& event : batch.internal) {
        total_bootstraps += event.bootstraps;
        total_refresh += event.elapsed; total_eval -= event.elapsed;
      }
      batches.push_back(batch);
      std::cout << "batch=" << start << " error=" << batch.after.absolute
                << " level=" << batch.level << " refreshed_level=" << batch.refreshed_level
                << " seconds=" << batch.evaluation_seconds + batch.refresh_seconds << std::endl;
    }
    const bool passed = decryption_failures == 0 && overall.non_finite == 0 && overall_before.non_finite == 0 &&
        overall.absolute <= tolerance && overall_before.absolute <= tolerance && max_padding <= tolerance;
    struct rusage usage {}; getrusage(RUSAGE_SELF, &usage);
    std::ofstream out(argv[1]); if (!out) throw std::runtime_error("cannot write vector RMS artifact");
    out << std::setprecision(17)
        << "{\"stage\":\"vector-rms-ckks-probe\",\"version\":\"" << FHEMAMBA_VERSION
        << "\",\"repo_commit\":\"" << argv[7] << "\",\"binary_sha256\":\"" << argv[8]
        << "\",\"recipe_sha256\":\"" << argv[9] << "\",\"backend\":\"FIDESlib\","
        << "\"encrypted\":true,\"passed\":" << (passed ? "true" : "false")
        << ",\"status\":\"" << (passed ? "passed" : "failed") << "\",\"mode\":\"" << mode
        << "\",\"gamma_placement\":\"" << placement << "\",\"refresh\":\"" << refresh
        << "\",\"refresh_coordinates\":\"" << coordinates << "\",\"meta_alpha\":" << meta_alpha
        << ",\"internal_refresh\":\"" << internal_refresh << "\",\"input_level\":" << input_level
        << ",\"refresh_trigger\":" << refresh_trigger
        << ",\"inverse_coordinates\":\"" << inverse_coordinates << '"'
        << ",\"meta_iterations\":" << meta_iterations
        << ",\"gate_metric\":\"vector-rms-output-absolute-error\",\"error_threshold\":" << tolerance
        << ",\"parameters\":{\"ring_dim\":" << cc->GetRingDimension() << ",\"slots\":" << slots
        << ",\"depth\":" << depth << ",\"scale\":" << scale << ",\"security\":\"" << security
        << "\",\"secret_key_dist\":\"uniform-ternary\",\"scaling\":\"FLEXIBLEAUTO\","
           "\"key_switch\":\"HYBRID\",\"large_digits\":3,\"first_mod_size\":60}"
        << ",\"layout\":{\"width\":" << fixture.width << ",\"padded_width\":" << layout.padded_width
        << ",\"lanes\":" << layout.lanes << ",\"cases\":" << fixture.cases
        << ",\"application_rotation_keys\":" << rotation_indices.size()
        << ",\"packing\":\"feature-strided-independent-vectors\"}"
        << ",\"refresh_bound\":" << refresh_bound
        << ",\"channel_scale_min\":" << *std::min_element(channel_scales.begin(), channel_scales.end())
        << ",\"channel_scale_max\":" << *std::max_element(channel_scales.begin(), channel_scales.end())
        << ",\"measurements\":{\"output\":";
    overall.write(out); out << ",\"before_refresh\":"; overall_before.write(out);
    out << ",\"checkpoint_inputs\":"; checkpoint.write(out);
    out << ",\"synthetic_inputs\":"; synthetic.write(out);
    out << ",\"padding_max_abs_error\":" << max_padding << ",\"setup_seconds\":" << setup_seconds
        << ",\"evaluation_seconds\":" << total_eval << ",\"refresh_seconds\":" << total_refresh
        << ",\"bootstraps\":" << total_bootstraps
        << ",\"decryption_failures\":" << decryption_failures
        << ",\"peak_rss_gib\":" << static_cast<double>(usage.ru_maxrss) / (1024 * 1024)
        << ",\"batches\":[";
    for (std::size_t i = 0; i < batches.size(); ++i) {
      const auto& b = batches[i];
      out << (i ? "," : "") << "{\"first_case\":" << b.first_case << ",\"cases\":" << b.cases
          << ",\"ct_ct_products\":" << b.ct_ct << ",\"ct_pt_products\":" << b.ct_pt
          << ",\"additions\":" << b.adds << ",\"rotations\":" << b.rotations
          << ",\"level\":" << b.level << ",\"scale_degree\":" << b.scale_degree
          << ",\"refreshed_level\":" << b.refreshed_level
          << ",\"evaluation_seconds\":" << b.evaluation_seconds
          << ",\"refresh_seconds\":" << b.refresh_seconds << ",\"before\":";
      b.before.write(out); out << ",\"after\":"; b.after.write(out);
      out << ",\"physical_bootstraps\":" << b.physical_bootstraps
          << ",\"refresh_ct_pt_products\":" << b.refresh_ct_pt_products
          << ",\"alignment_products\":" << b.alignment_products
          << ",\"meta_residual_max_abs\":" << b.meta_residual_max_abs
          << ",\"meta_residual_non_finite\":" << b.meta_residual_non_finite
          << ",\"input_level\":" << b.input_level
          << ",\"input_conditioning_products\":" << b.input_conditioning_products
          << ",\"input_max_abs_error\":" << b.input_error
          << ",\"internal_refreshes\":[";
      for (std::size_t j = 0; j < b.internal.size(); ++j) {
        const auto& e = b.internal[j];
        out << (j ? "," : "") << "{\"stage\":" << e.stage << ",\"input_level\":" << e.input_level
            << ",\"output_level\":" << e.output_level << ",\"ct_pt_products\":" << e.ct_pt
            << ",\"alignment_products\":" << e.alignments << ",\"physical_bootstraps\":" << e.bootstraps
            << ",\"seconds\":" << e.elapsed << ",\"before_max_abs\":" << e.before_max
            << ",\"bound\":" << e.bound
            << ",\"after_max_abs\":" << e.after_max << ",\"refresh_difference\":" << e.difference
            << ",\"diagnostic_non_finite\":" << e.diagnostic_non_finite << '}';
      }
      out << "]}";
    }
    out << "]},\"measurement_scope\":{\"artifact_level_report\":true,\"evaluation_decryptions\":0,"
        << "\"additional_post_evaluation_decryptions_per_internal_refresh\":2,"
        << "\"post_evaluation_diagnostic_decryptions_per_batch\":"
        << (refresh == "output-meta" ? 4 : refresh == "output" ? 3 : 2) << ','
        <<
           "\"real_decoder_noise_enabled\":true,\"full_vector_rms_reduction_measured\":true,"
           "\"learned_gamma_measured\":true,\"secret_key_in_benchmark_process\":true,"
           "\"prior_layer_ckks_error_measured\":false,\"full_model_correctness_claimed\":false,"
           "\"claim\":\"Packed RMSNorm with optional identity depth pressure and internal/output refresh; no preceding model layer or full-model claim.\"}}\n";
    return passed ? 0 : 1;
  } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 2; }
}
