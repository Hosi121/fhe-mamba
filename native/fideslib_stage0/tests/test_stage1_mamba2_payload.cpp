#include "stage1_mamba2_payload.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

void require_invalid(const std::function<void()>& operation) {
  try {
    operation();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error("expected runtime_error");
}

}  // namespace

auto main() -> int {
  using fhemamba::stage1::M1Payload;
  using fhemamba::stage1::read_chain_payload;
  using fhemamba::stage1::require_same_layer_dims;
  using fhemamba::stage1::parse_poly_spec;
  using fhemamba::stage1::parse_joint_gate;
  const std::string joint = R"({"kind":"shared-dissipation-factor-v1",)"
      R"("coefficient_layout":"degree-major-head-minor","lo":[-1,-2],"hi":[1,2],)"
      R"("p":[0.1,0.2,0.3,0.4],"q":[0.5,0.6],"rates":[1,2]})";
  if (parse_joint_gate(joint, 2).p.size() != 4)
    throw std::runtime_error("joint matrix parsing failed");
  require_invalid([&] { parse_joint_gate(joint,3); });
  for (const auto& replacement : {std::pair{"[0.1,0.2,0.3,0.4]", "[0.1,0.2,0.3]"},
          {"[0.1,0.2,0.3,0.4]", "[]"}, {"[0.1,0.2,0.3,0.4]", "[NaN,0.2]"},
          {"[-1,-2]", "[1,-2]"}, {"\"rates\":[1,2]", "\"rates\":[0,2]"}}) {
    auto invalid=joint;
    invalid.replace(invalid.find(replacement.first),std::string(replacement.first).size(),replacement.second);
    require_invalid([&] { parse_joint_gate(invalid,2); });
  }

  const std::string recipe = R"({"kind":"scaled-goldschmidt-invsqrt-v1",)"
      R"("lo":0.01,"hi":10,"seed":0.3,"tolerance":0.0000001,)"
      R"("coefficients":[[1.5,0.5],[1.6,0.6]],)"
      R"("final_recomputed_newton":true,"oracle_workspace_dtype":"float64"})";
  const auto parsed = parse_poly_spec(recipe);
  if (!parsed.normalization || parsed.normalization->coefficients.size() != 2 ||
      parsed.normalization->coefficients[1].second != 0.6 || !parsed.coeffs.empty())
    throw std::runtime_error("scheduled normalization was not parsed independently of Chebyshev");
  for (const auto& replacement : {std::pair{"[[1.5,0.5],[1.6,0.6]]", "[]"},
          {"[[1.5,0.5],[1.6,0.6]]", "[[1.5],[0.5,1.6,0.6]]"},
          {"[[1.5,0.5],[1.6,0.6]]", "[[1.5,0.5],]"},
          {"\"lo\":0.01", "\"lo\":0"},
          {"\"seed\":0.3", "\"seed\":-0.3"},
          {"true", "false"}, {"float64", "float32"}}) {
    auto invalid = recipe;
    invalid.replace(invalid.find(replacement.first), std::string(replacement.first).size(), replacement.second);
    require_invalid([&] { parse_poly_spec(invalid); });
  }

  M1Payload expected;
  expected.d_model = 768;
  expected.d_inner = 1536;
  expected.num_heads = 24;
  expected.head_dim = 64;
  expected.state_size = 64;
  expected.n_groups = 1;
  expected.conv_kernel = 4;
  expected.conv_dim = 1664;
  expected.proj_dim = 3352;
  require_same_layer_dims(expected, expected, 0);

  auto mismatch = expected;
  mismatch.state_size = 32;
  require_invalid([&] { require_same_layer_dims(expected, mismatch, 7); });

  const auto root = fs::temp_directory_path() / "fhemamba-stage1-payload-test";
  fs::remove_all(root);
  fs::create_directories(root);
  {
    std::ofstream meta(root / "chain.json");
    meta << R"({"format":"wrong"})";
  }
  require_invalid([&] { read_chain_payload(root.string(), false); });
  {
    std::ofstream meta(root / "chain.json");
    meta << R"({"format":"fhemamba-m2-chain-v1","n_layers":2,)"
            R"("n_test_tokens":1,"final_norm_eps":0.00001,)"
            R"("layer_dirs":["layer_00"],"tensors":{}})";
  }
  require_invalid([&] { read_chain_payload(root.string(), false); });
  fs::remove_all(root);
  return 0;
}
