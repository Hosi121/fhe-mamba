#pragma once
// Included after PackedEvaluator, only by the diagnostic-capable client CLI.
// No key or ciphertext is serialized. Controls never replace DAG values.
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <optional>

template <class Decrypt>
void install_refresh_replay(PackedEvaluator& e, const fhemamba::PackedProgram& original_program,
                            std::pair<int, int> pair, const std::string& output,
                            Decrypt decrypt_client, std::size_t& decryptions,
                            std::function<const std::vector<double>&(int)> reference = {}) {
  const auto [a, b] = pair;
  if (a < 0 || b < 0 || a == b || static_cast<std::size_t>(std::max(a, b)) >= original_program.nodes.size() ||
      original_program.nodes[a].size + original_program.nodes[b].size != original_program.slots)
    throw std::invalid_argument("diagnostic refresh pair must fill the physical slots");
  e.diagnostic_refresh_pair = pair;
  e.diagnostic_refresh_replay = [&e, &original_program, output, decrypt_client, reference, &decryptions,
                                 done = false](const std::vector<int>& indices,
                                               const std::vector<Ct>& before,
                                               const std::vector<Ct>& actual) mutable {
    if (done) throw std::runtime_error("diagnostic refresh pair executed twice");
    done = true;
    namespace gpu = FIDESlib::CKKS;
    auto device = [](const Ct& x) {
      return std::static_pointer_cast<gpu::Ciphertext>(x->parent_context->GetDeviceCiphertext(x->gpu));
    };
    auto snapshot = [&](const Ct& x) {
      gpu::RawCipherText raw{}; sync_gpu(); device(x)->store(raw); sync_gpu(); return raw;
    };
    auto decrypt = [&](const Ct& x, int size) -> std::optional<std::vector<double>> {
      ++decryptions;
      try { return decrypt_client(x, size); }
      catch (const std::exception& error) {
        if (!fhemamba::diagnostic_decode_rejected(error)) throw;
        return std::nullopt;
      }
    };
    auto difference = [](const auto& x, const auto& y) {
      if (x.size() != y.size()) throw std::runtime_error("refresh replay width mismatch");
      double error = 0;
      for (std::size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
          throw std::runtime_error("refresh replay nonfinite output");
        error = std::max(error, std::abs(x[i] - y[i]));
      }
      return error;
    };
    fhemamba::PackedProgram program;
    program.slots = original_program.slots; program.bound = original_program.bound;
    program.nodes.resize(2);
    std::vector<std::vector<double>> expected;
    std::vector<bool> observed_input;
    std::vector<Ct> seeds;
    std::size_t exact_words = 0;
    std::ofstream report(output); report << std::setprecision(17);
    report << "{\"diagnostic_only\":true,\"fresh_ciphertexts\":false,\"input_nodes\":["
           << indices[0] << ',' << indices[1] << "],\"inputs\":[";
    for (int i = 0; i < 2; ++i) {
      program.nodes[i].size = original_program.nodes[indices[i]].size;
      program.nodes[i].bound = original_program.nodes[indices[i]].bound;
      auto copy = before[i]->Clone(); sync_gpu();
      const auto x = snapshot(before[i]), y = snapshot(copy);
      if (x.sub_0 != y.sub_0 || x.sub_1 != y.sub_1 || x.numRes != y.numRes ||
          x.Noise != y.Noise || x.NoiseLevel != y.NoiseLevel || x.slots != y.slots || x.keyid != y.keyid)
        throw std::runtime_error("refresh replay clone changed residues or metadata");
      for (const auto& limb : x.sub_0) exact_words += limb.size();
      for (const auto& limb : x.sub_1) exact_words += limb.size();
      seeds.push_back(copy);
      auto input = decrypt(before[i], program.nodes[i].size);
      observed_input.push_back(input.has_value());
      if (!input && !reference) throw std::runtime_error("refresh replay input decode rejected and no CPU reference available");
      expected.push_back(input ? *input : reference(indices[i]));
      auto initial = decrypt(actual[i], program.nodes[i].size);
      if (i) report << ',';
      report << "{\"node\":" << indices[i] << ",\"size\":" << program.nodes[i].size
             << ",\"bound\":" << program.nodes[i].bound << ",\"level\":" << before[i]->GetLevel()
             << ",\"degree\":" << before[i]->GetNoiseScaleDeg()
             << ",\"scaling_factor\":" << device(before[i])->NoiseFactor
             << ",\"retained_q_towers\":" << x.numRes
             << ",\"input_decryption_rejected\":" << (input ? "false" : "true")
             << ",\"reference\":\"" << (input ? "observed_input" : "cpu_reference") << "\""
             << ",\"original_refresh_decryption_rejected\":" << (initial ? "false" : "true")
             << ",\"original_refresh_max_abs_error_vs_reference\":";
      if (initial) report << difference(*initial, expected.back()); else report << "null";
      report << '}';
    }
    report << "],\"exact_clone_words\":" << exact_words << ",\"controls\":[";
    const bool original_merge = e.merge_refresh_correction;
    struct Restore { PackedEvaluator& e; bool merge; ~Restore() { e.merge_refresh_correction = merge; } } restore{e, original_merge};
    const std::vector<std::string> modes{"identical_group", "double_bound_group", "separate_groups", "unmerged_group"};
    for (std::size_t mode = 0; mode < modes.size(); ++mode) {
      std::vector<Ct> values{seeds[0]->Clone(), seeds[1]->Clone()}; sync_gpu();
      for (int i = 0; i < 2; ++i)
        program.nodes[i].bound = original_program.nodes[indices[i]].bound * (mode == 1 ? 2 : 1);
      e.merge_refresh_correction = mode == 3 ? false : original_merge;
      const auto start = Clock::now(); const auto boots = e.bootstraps;
      if (mode == 2) {
        e.refresh_group({0}, values, program); e.refresh_group({1}, values, program);
      } else e.refresh_group({0, 1}, values, program);
      const auto seconds = elapsed(start);
      if (mode) report << ',';
      report << "{\"mode\":\"" << modes[mode] << "\",\"seconds\":" << seconds
             << ",\"bootstraps\":" << e.bootstraps - boots << ",\"outputs\":[";
      for (int i = 0; i < 2; ++i) {
        const auto value = decrypt(values[i], program.nodes[i].size);
        if (i) report << ',';
        report << "{\"reference\":\"" << (observed_input[i] ? "observed_input" : "cpu_reference") << "\""
               << ",\"decryption_rejected\":" << (value ? "false" : "true")
               << ",\"max_abs_error_vs_reference\":";
        if (value) report << difference(*value, expected[i]); else report << "null";
        report << ",\"level\":" << values[i]->GetLevel()
               << ",\"degree\":" << values[i]->GetNoiseScaleDeg() << '}';
      }
      report << "]}";
      report.flush();
    }
    report << "],\"scope\":\"Diagnostic controls on exact clones of the original encrypted inputs; no control output replaces a model value and no timing qualification is claimed\"}\n";
    if (!report) throw std::runtime_error("could not write refresh replay report");
    std::cout << "diagnostic_refresh_replay_complete=true exact_clone_words=" << exact_words << std::endl;
  };
}
