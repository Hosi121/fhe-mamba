#include "packed_program.hpp"
#include <cassert>
#include <sstream>
#include <limits>

int main() {
  const std::string valid = "fhemamba-packed-v1 8 64 2 1\n"
      "input 2 0 2 1 2\ngather 2 1 0 2 1 0\n1 2 2 1 2 1\n";
  std::istringstream stream(valid);
  auto p = fhemamba::read_packed_program(stream);
  assert(p.slots == 8 && p.nodes.size() == 2 && p.outputs[0].exact[0] == 2);
  const auto errors = fhemamba::packed_output_errors({2.0005, 1}, p.outputs[0]);
  assert(errors.passes(.001, .001) && !errors.passes(.0001, .001));
  assert(!errors.passes(.001, .0001));
  assert(!errors.passes(std::numeric_limits<double>::infinity(), .001));
  const auto nonfinite = fhemamba::packed_output_errors(
      {std::numeric_limits<double>::quiet_NaN(), 1}, p.outputs[0]);
  assert(nonfinite.non_finite == 1 && !nonfinite.passes(.001, .001));
  bool width_failed = false;
  try { fhemamba::packed_output_errors({2}, p.outputs[0]); }
  catch (const std::invalid_argument&) { width_failed = true; }
  assert(width_failed);
  const std::string extended = "fhemamba-packed-v2 8 64 4 1\n"
      "input 2 8 0 2 1 2\nlinear 2 16 1 0 4 1 0 0 1\n"
      "linear_ref 2 16 1 0 1 1\nfeedback 2 8 1 2 0\n2 2 1 2 1 2\n";
  std::istringstream v2_stream(extended);
  auto v2 = fhemamba::read_packed_program(v2_stream);
  assert(v2.nodes[2].operation == "linear_ref" && v2.nodes[2].bound == 16);
  assert(v2.nodes[3].operation == "feedback" && v2.nodes[3].data.empty());
  // A generation-sized dependency chain must parse beyond the old probe cap.
  const int long_nodes = 197330;
  std::ostringstream long_text;
  long_text << "fhemamba-packed-v2 8 64 " << long_nodes << " 1\ninput 1 8 0 1 1\n";
  for (int i = 1; i < long_nodes; ++i)
    long_text << "addp 1 8 1 " << i - 1 << " 1 0\n";
  long_text << long_nodes - 1 << " 1 1 1\n";
  std::istringstream long_stream(long_text.str());
  auto long_program = fhemamba::read_packed_program(long_stream);
  assert(long_program.nodes.size() == long_nodes);
  assert(long_program.outputs[0].node == long_nodes - 1);
  std::istringstream compact_stream(extended);
  auto compact = fhemamba::read_packed_program(compact_stream, true);
  assert(compact.nodes[1].data.empty() && compact.nodes[1].bf16_weights.size() == 4);
  for (std::size_t i = 0; i < v2.nodes[1].data.size(); ++i)
    assert(v2.nodes[1].weights()[i] == compact.nodes[1].weights()[i]);
  // Enumerate all finite BF16 encodings, including subnormals and signed zero.
  std::vector<double> exact;
  for (uint32_t bits = 0; bits <= 0xffff; ++bits) {
    const float value = std::bit_cast<float>(bits << 16);
    if (std::isfinite(value)) exact.push_back(value);
  }
  const auto original = exact;
  std::vector<uint16_t> packed;
  assert(fhemamba::compact_exact_bf16(exact, packed) && exact.empty());
  const fhemamba::PublicWeightView view(packed);
  assert(view.size() == original.size());
  for (std::size_t i = 0; i < original.size(); ++i)
    assert(std::bit_cast<uint64_t>(view[i]) == std::bit_cast<uint64_t>(original[i]));
  for (double inexact : {0.1, 1.0000000001, 1e-50, std::numeric_limits<double>::max(),
                        std::numeric_limits<double>::infinity()}) {
    std::vector<double> values{1, -0.0, inexact};
    std::vector<uint16_t> empty;
    assert(!fhemamba::compact_exact_bf16(values, empty));
    assert(values.size() == 3 && values.back() == inexact && empty.empty());
  }
  for (const std::string invalid : std::vector<std::string>{
      "fhemamba-packed-v1 3 64 1 1", // non-power-of-two slots
      "fhemamba-packed-v2 8 64 1000001 1", // bounded node allocation
      "fhemamba-packed-v1 8 64 1 1 input 2 0 2 1 nan", // invalid constants
      "fhemamba-packed-v1 8 64 1 1 add 2 2 0 0 0", // future dependency
      "fhemamba-packed-v1 8 64 2 1 input 2 0 2 1 2 gather 1 1 0 1 2", // index overflow
      valid + "trailing",
      "fhemamba-packed-v2 8 64 1 1 input 2 65 0 2 1 2", // out-of-policy bound
      "fhemamba-packed-v2 8 64 2 1 input 2 8 0 2 1 2 linear_ref 2 8 1 0 1 0", // reference is not weights
      "fhemamba-packed-v2 8 64 2 1 input 2 8 0 2 1 2 feedback 3 8 1 0 0"}) {
    bool failed = false;
    try { std::istringstream s(invalid); fhemamba::read_packed_program(s); }
    catch (const std::invalid_argument&) { failed = true; }
    assert(failed);
  }
}
