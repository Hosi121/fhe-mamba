// Diagnostic fresh-encryption isolation, not replay of the original RNS noise.
// Reuse the executor implementation so level alignment/routing do not drift.
#define FHEMAMBA_PACKED_EVALUATOR_ONLY
#include "../../native/fideslib_stage0/src/packed_fideslib.cpp"
#include "../../native/fideslib_stage0/src/diagnostic_refresh_replay.hpp"

static auto read_vector(std::istream& in, int limit) -> std::vector<double> {
  int count = -1;
  if (!(in >> count) || count < 0 || count > limit) throw std::invalid_argument("invalid vector size");
  std::vector<double> values(count);
  for (auto& x : values)
    if (!(in >> x) || !std::isfinite(x)) throw std::invalid_argument("invalid case value");
  return values;
}

int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4)
      throw std::invalid_argument("usage: packed_operation_replay CASE OUTPUT [--in-context-controls]");
    const bool controls = argc == 4 && std::string(argv[3]) == "--in-context-controls";
    if (argc == 4 && !controls) throw std::invalid_argument("unknown replay option");
    std::ifstream in(argv[1]);
    std::string magic, operation;
    int slots, width, count;
    double bound;
    if (!(in >> magic >> operation >> slots >> width >> bound >> count) ||
        magic != "fhemamba-operation-case-v1" || slots != 65536 || width < 1 || width > slots ||
        !std::isfinite(bound) || bound <= 0 || count < 1 || count > 2)
      throw std::invalid_argument("invalid operation case");
    struct Input { int level, degree; std::vector<double> values; };
    std::vector<Input> inputs(count);
    for (auto& x : inputs) {
      if (!(in >> x.level >> x.degree) || x.level < 0 || x.level > 35 || x.degree < 1 || x.degree > 2)
        throw std::invalid_argument("invalid input level/degree");
      x.values = read_vector(in, slots);
      if (x.values.empty()) throw std::invalid_argument("empty input");
    }
    const auto data = read_vector(in, slots), expected = read_vector(in, slots);
    std::string trailing;
    if (expected.size() != static_cast<std::size_t>(width) || (in >> trailing))
      throw std::invalid_argument("invalid expected output/trailing data");
    const bool binary = operation == "add" || operation == "mul";
    const bool group = operation == "refresh_group";
    if (controls && !group) throw std::invalid_argument("in-context controls require refresh_group");
    if ((!group && (binary ? 2 : 1) != count) || (binary && inputs[0].values.size() != inputs[1].values.size()))
      throw std::invalid_argument("invalid operation input count/width");
    if (group && (count != 2 || data.size() != 2 || data[0] <= 0 || data[1] <= 0 ||
                  inputs[0].values.size() + inputs[1].values.size() != static_cast<std::size_t>(slots) || width != slots))
      throw std::invalid_argument("group isolation requires two fully observed full-packing inputs");
    if ((operation == "mulp" || operation == "addp") && data.size() != inputs[0].values.size())
      throw std::invalid_argument("invalid public operand");
    if (operation == "repeat" && (data.size() != 3 || data[0] * data[1] != inputs[0].values.size() ||
                                  data[0] * data[1] * data[2] != width))
      throw std::invalid_argument("invalid repeat shape");

    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecurityLevel(HEStd_128_classic); p.SetSecretKeyDist(UNIFORM_TERNARY);
    p.SetCKKSDataType(REAL); p.SetRingDim(131072); p.SetBatchSize(slots);
    p.SetMultiplicativeDepth(44); p.SetScalingModSize(59); p.SetFirstModSize(60);
    p.SetScalingTechnique(FLEXIBLEAUTO); p.SetKeySwitchTechnique(HYBRID);
    p.SetNumLargeDigits(4); p.SetDevices({0});
    p.SetPlaintextAutoload(false); p.SetCiphertextAutoload(true);
    auto cc = GenCryptoContext(p);
    const auto audit = fhemamba::audit_ckks_context(cc); audit.require_classical128();
    for (auto f : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE}) cc->Enable(f);
    auto keys = cc->KeyGen(); cc->EvalMultKeyGen(keys.secretKey);
    std::vector<int> rotations;
    for (int i = 1; i < slots; i *= 2) { rotations.push_back(i); rotations.push_back(-i); }
    cc->EvalRotateKeyGen(keys.secretKey, rotations);
    cc->EvalBootstrapSetup({4, 4}, {0, 0}, slots, 0, true);
    cc->EvalBootstrapKeyGen(keys.secretKey, slots);
    cc->LoadContext(keys.publicKey); sync_gpu();
    PackedEvaluator e{cc, keys.publicKey, slots, bound};
    e.planned_refresh = e.inplace_ops = e.naf_rotations = true;
    e.bsgs_routing_stages = e.hoist_rotations = e.merge_refresh_correction = true;
    e.refresh_ceiling = 35; e.refreshed_level = 18; e.current_bound = bound;
    e.cache_plaintexts = true;
    e.plaintext_cache = decltype(e.plaintext_cache)(2048, false);
    e.plaintexts = std::make_unique<fhemamba::PlaintextPreparation>(
        cc, slots, fhemamba::PlaintextPreparationOptions{
          .fast_upload = true, .gpu_ntt = true, .direct_upload = true,
          .move_coefficients = true, .borrow_upload = true,
          .gpu_rns = true, .gpu_addend_rns = true, .gpu_fft = true});
    auto decrypt = [&](const Ct& x, std::size_t size) {
      auto copy = x->Clone(); Plaintext plain;
      cc->Decrypt(keys.secretKey, copy, &plain); plain->SetLength(size);
      return plain->GetRealPackedValue();
    };
    auto error = [](const auto& a, const auto& b) {
      if (a.size() != b.size()) throw std::runtime_error("output size mismatch");
      double worst = 0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) throw std::runtime_error("nonfinite output");
        worst = std::max(worst, std::abs(a[i] - b[i]));
      }
      return worst;
    };
    std::vector<Ct> encrypted;
    double input_error = 0;
    for (const auto& x : inputs) {
      // Encoding directly at the target level produces fresh noise. It does
      // not pretend to reconstruct accumulated noise or original key material.
      auto plain = cc->MakeCKKSPackedPlaintext(x.values, 1, x.level, nullptr, slots);
      auto value = cc->Encrypt(keys.publicKey, plain);
      if (x.degree == 2) value = e.scale(value, 1.0);
      sync_gpu();
      if (value->GetLevel() != x.level || value->GetNoiseScaleDeg() != x.degree)
        throw std::runtime_error("fresh input level/degree mismatch");
      input_error = std::max(input_error, error(decrypt(value, x.values.size()), x.values));
      encrypted.push_back(value);
    }
    const auto start = Clock::now();
    const auto original = encrypted;
    Ct out;
    auto x = encrypted[0];
    if (binary) out = e.binary_owned(x->Clone(), encrypted[1]->Clone(), operation == "mul");
    else if (operation == "mulp" || operation == "addp") out = e.plain(x, data, operation == "mulp");
    else if (operation == "gather" || operation == "scatter") out = e.gather(x, data, width, operation == "scatter");
    else if (operation == "refresh") out = e.refresh(x, bound);
    else if (group) {
      fhemamba::PackedProgram program;
      program.slots = slots; program.bound = bound;
      program.nodes.resize(2);
      for (int i = 0; i < 2; ++i) {
        program.nodes[i].size = inputs[i].values.size(); program.nodes[i].bound = data[i];
      }
      e.refresh_group({0, 1}, encrypted, program);
      out = encrypted[0];
      if (controls) {
        std::size_t decryptions = 0;
        install_refresh_replay(e, program, {0, 1}, std::string(argv[2]) + ".refresh-replay.json",
            decrypt, decryptions);
        e.diagnostic_refresh_replay({0, 1}, original, encrypted);
      }
    }
    else if (operation == "repeat") {
      const int outer = data[0], inner = data[1], repeat = data[2];
      std::vector<double> destinations(outer * inner);
      for (int g = 0; g < outer; ++g) for (int j = 0; j < inner; ++j)
        destinations[g * inner + j] = g * inner * repeat + j;
      out = e.gather(x, destinations, width, true);
      out = fhemamba::stage1::rotation_sum(out, repeat, -inner, true,
          [&](const Ct& a, int shift) { return e.rotate(a, shift); },
          [&](const Ct& a, const Ct& b) { return e.add(a, b); });
    } else throw std::invalid_argument("unsupported operation");
    sync_gpu();
    const auto duration = elapsed(start);
    auto actual = decrypt(out, group ? inputs[0].values.size() : expected.size());
    if (group) {
      const auto second = decrypt(encrypted[1], inputs[1].values.size());
      actual.insert(actual.end(), second.begin(), second.end());
    }
    const double output_error = error(actual, expected);
    std::ofstream report(argv[2]); report << std::setprecision(17);
    report << "{\"diagnostic_only\":true,\"fresh_ciphertexts\":true,\"operation\":\"" << operation
           << "\",\"input_max_abs_error\":" << input_error << ",\"output_max_abs_error\":" << output_error
           << ",\"level\":" << out->GetLevel() << ",\"degree\":" << out->GetNoiseScaleDeg()
           << ",\"seconds\":" << duration << ",\"bootstraps\":" << e.bootstraps << ",\"security_audit\":";
    audit.write_json(report);
    report << ",\"scope\":\"Fresh encryption at recorded level/degree; original RNS noise/keys are not retained. refresh_group preserves the two full-packing companions and their public bounds.\"}\n";
    if (!report) throw std::runtime_error("could not write replay result");
    std::cout << "operation=" << operation << " input_error=" << input_error << " output_error=" << output_error << '\n';
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
