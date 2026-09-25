#include "plaintext_rns.hpp"
#include <stdexcept>
#include <limits>

int main() {
  using fhemamba::compact_plaintext_fits;
  const auto check = [](bool value) { if (!value) throw std::runtime_error("compact RNS bound"); };
  const uint64_t q = (uint64_t{1} << 60) - 131071;
  const double scale = std::ldexp(1.0, 59);
  check(compact_plaintext_fits({0.0, -0.0}, 2, scale, q));
  check(compact_plaintext_fits({0.49, -0.49}, 2, scale, q));
  check(!compact_plaintext_fits({0.51, -0.51}, 2, scale, q));
  check(!compact_plaintext_fits({1.0, -1.0}, 2, scale, q)); // cancellation cannot relax the bound
  check(compact_plaintext_fits({1.0}, 4, scale, q)); // zero padding contributes to the mean
  check(!compact_plaintext_fits({1.0}, 0, scale, q));
  check(!compact_plaintext_fits({0.0, 0.0}, 1, scale, q));
  check(!compact_plaintext_fits({std::numeric_limits<double>::infinity()}, 1, scale, q));
  check(!compact_plaintext_fits({std::numeric_limits<double>::quiet_NaN()}, 1, scale, q));
  check(!compact_plaintext_fits({0.0}, 1, -1, q));
  check(!compact_plaintext_fits({0.0}, 1, scale, uint64_t{1} << 61));
}
