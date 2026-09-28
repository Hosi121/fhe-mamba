#pragma once

#include "packed_program.hpp"
#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <ostream>
#include <string_view>

namespace fhemamba {
inline bool diagnostic_decode_rejected(const std::exception& error) {
  return std::string_view(error.what()).find("approximation error is too high") != std::string_view::npos;
}
// Test-fixture diagnostics only. This object owns no ciphertext or secret key;
// the explicitly enabled client observer supplies decrypted copies. It never
// feeds reference values or observed plaintext back into the encrypted graph.
class PackedDiagnostics {
 public:
  PackedDiagnostics(std::istream& input, const PackedProgram& program) {
    static_assert(std::endian::native == std::endian::little);
    char magic[8];
    input.read(magic, 8);
    if (!input || std::string(magic, 8) != "FHEMDG01")
      throw std::invalid_argument("invalid packed diagnostic magic");
    const auto count = read<uint64_t>(input);
    if (!count || count > program.nodes.size())
      throw std::invalid_argument("invalid packed diagnostic count");
    std::size_t total = 0;
    for (uint64_t k = 0; k < count; ++k) {
      const auto node = read<uint32_t>(input), size = read<uint32_t>(input);
      if (node >= program.nodes.size() || size != program.nodes[node].size ||
          references_.contains(node) || (total += size) > (1u << 27))
        throw std::invalid_argument("invalid packed diagnostic node/size");
      auto& values = references_[node];
      values.resize(size);
      input.read(reinterpret_cast<char*>(values.data()), size * sizeof(double));
      if (!input || !std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("invalid packed diagnostic reference");
    }
    if (input.peek() != std::char_traits<char>::eof())
      throw std::invalid_argument("trailing packed diagnostic data");
  }
  bool contains(int node) const { return references_.contains(node); }
  const std::vector<double>& reference(int node) const { return references_.at(node); }
  std::size_t decryptions = 0;
  void record_decode_rejection(std::ostream& output, int node, const char* event, int level, int degree) {
    ++decryptions;
    output << "{\"node\":" << node << ",\"event\":\"" << event
           << "\",\"level\":" << level << ",\"degree\":" << degree
           << ",\"max_abs_error\":null,\"max_abs_value\":null,\"non_finite\":null,\"worst_index\":null"
           << ",\"decryption_error\":\"ckks_approximation_error_too_high\",\"values\":[";
    for (std::size_t i = 0; i < references_.at(node).size(); ++i) {
      if (i) output << ',';
      output << "null";
    }
    output << "]}\n"; output.flush();
    if (!output) throw std::runtime_error("could not write packed diagnostic observation");
  }
  void record(std::ostream& output, int node, const char* event, int level, int degree,
              const std::vector<double>& values) {
    const auto& reference = references_.at(node);
    if (values.size() != reference.size())
      throw std::invalid_argument("diagnostic observation size mismatch");
    double error = 0, maximum = 0;
    int non_finite = 0, worst = -1;
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (!std::isfinite(values[i])) { ++non_finite; continue; }
      const auto delta = std::abs(values[i] - reference[i]);
      if (delta > error) { error = delta; worst = i; }
      maximum = std::max(maximum, std::abs(values[i]));
    }
    ++decryptions;
    output << std::setprecision(17) << "{\"node\":" << node << ",\"event\":\"" << event
           << "\",\"level\":" << level << ",\"degree\":" << degree
           << ",\"max_abs_error\":" << error << ",\"max_abs_value\":" << maximum
           << ",\"non_finite\":" << non_finite << ",\"worst_index\":" << worst << ",\"values\":[";
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i) output << ',';
      if (std::isfinite(values[i])) output << values[i]; else output << "null";
    }
    output << "]}\n";
    output.flush();
    if (!output) throw std::runtime_error("could not write packed diagnostic observation");
  }
 private:
  template <class T> static T read(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::invalid_argument("truncated packed diagnostic input");
    return value;
  }
  std::map<int, std::vector<double>> references_;
};
}  // namespace fhemamba
