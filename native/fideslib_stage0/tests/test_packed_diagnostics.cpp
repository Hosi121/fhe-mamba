#include "packed_diagnostics.hpp"
#include <limits>
#include <sstream>

template <class T> void put(std::ostream& out, T value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
static std::string fixture(uint32_t node = 0, uint32_t size = 2) {
  std::ostringstream out;
  out.write("FHEMDG01", 8); put<uint64_t>(out, 1);
  put(out, node); put(out, size); put(out, 1.0); put(out, -2.0);
  return out.str();
}
static void require(bool value) { if (!value) throw std::runtime_error("diagnostic contract failed"); }
int main() {
  fhemamba::PackedProgram program{8, 8, {{"input", 2, {}, {1, -2}, 8}}, {}};
  std::istringstream input(fixture());
  fhemamba::PackedDiagnostics diagnostic(input, program);
  require(diagnostic.contains(0) && !diagnostic.contains(1));
  std::ostringstream out;
  diagnostic.record(out, 0, "node_output", 5, 1, {1.0, -1.75});
  require(diagnostic.decryptions == 1 && out.str().find("\"max_abs_error\":0.25") != std::string::npos);
  diagnostic.record(out, 0, "after_refresh", 3, 1, {1, std::numeric_limits<double>::quiet_NaN()});
  require(diagnostic.decryptions == 2 && out.str().find("\"non_finite\":1") != std::string::npos);
  require(out.str().find("null") != std::string::npos && out.str().find("nan") == std::string::npos);
  require(diagnostic.reference(0) == std::vector<double>({1, -2}));
  std::ostringstream rejected;
  diagnostic.record_decode_rejection(rejected, 0, "before_refresh", 35, 2);
  require(diagnostic.decryptions == 3);
  require(rejected.str().find("\"max_abs_error\":null") != std::string::npos);
  require(rejected.str().find("\"values\":[null,null]") != std::string::npos);
  require(fhemamba::diagnostic_decode_rejected(std::runtime_error("Decode: approximation error is too high")));
  require(!fhemamba::diagnostic_decode_rejected(std::runtime_error("CUDA allocation failed")));
  for (const auto& bad : {fixture(1), fixture(0, 3), fixture().substr(0, 20), fixture() + "x"}) {
    bool rejected = false;
    try { std::istringstream in(bad); fhemamba::PackedDiagnostics d(in, program); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
  }
}
