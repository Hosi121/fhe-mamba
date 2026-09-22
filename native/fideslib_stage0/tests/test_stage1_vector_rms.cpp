#include "stage1_vector_rms.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

using namespace fhemamba::stage1;
namespace {
void require(bool b) { if (!b) throw std::runtime_error("vector RMS contract failed"); }
struct HostOps {
  auto rotate(const std::vector<double>& x, int shift) -> std::vector<double> {
    auto y = x;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) y[i] = x[(i + shift) % x.size()];
    return y;
  }
  auto add(const std::vector<double>& a, const std::vector<double>& b) -> std::vector<double> {
    auto c = a; for (std::size_t i = 0; i < a.size(); ++i) c[i] += b[i]; return c;
  }
};
}
auto main() -> int {
  const std::vector<double> gamma = {0, -53, 0.001, 2};
  const auto scales = vector_rms_refresh_scales(gamma);
  for (std::size_t i = 0; i < gamma.size(); ++i)
    require(scales[i] >= std::sqrt(gamma.size()) * std::abs(gamma[i]));
  require(*std::max_element(scales.begin(), scales.end()) /
          *std::min_element(scales.begin(), scales.end()) <= 64);
  require(vector_rms_refresh_scales({0, 0}) == std::vector<double>({1, 1}));
  for (int width : {1, 3, 768, 1536}) {
    VectorRmsLayout layout(width);
    std::vector<double> squares(layout.slots, 0), sums(layout.lanes, 0);
    for (int lane = 0; lane < layout.lanes; ++lane)
      for (int j = 0; j < width; ++j) {
        // Large cross-lane differences detect accidental mixing or broadcasts.
        const double x = lane * 3 + (j % 7) - 4;
        squares[layout.slot(j, lane)] = x * x; sums[lane] += x * x;
      }
    HostOps ops;
    const auto actual = vector_rms_sum(squares, layout, ops);
    for (int j = 0; j < layout.padded_width; ++j)
      for (int lane = 0; lane < layout.lanes; ++lane)
        require(actual[layout.slot(j, lane)] == sums[lane]);
  }
  std::istringstream good("fhemamba-vector-rms-v1 2 1 0.01 1 -2 3 4");
  auto f = read_vector_rms_fixture(good);
  require(f.width == 2 && f.gamma[1] == -2 && f.inputs[1] == 4);
  for (const auto* text : {"bad 2 1 0.01 1 2 3 4", "fhemamba-vector-rms-v1 2 1 -1 1 2 3 4",
                          "fhemamba-vector-rms-v1 2 1 0.01 1 2 3", "fhemamba-vector-rms-v1 1 1 0.01 1 2 extra"}) {
    bool rejected = false;
    try { std::istringstream in(text); read_vector_rms_fixture(in); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
  }
}
