// Extracted finite-input Encode range scan, not an OpenFHE/FIDESlib change.
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using Values = std::vector<std::complex<double>>;
__attribute__((noinline)) int original(Values& inverse, double scale) {
  int logc = std::numeric_limits<int>::min();
  for (auto& value : inverse) {
    value *= scale;
    if (value.real() != 0.0) logc = std::max(logc, static_cast<int>(std::ceil(std::log2(std::abs(value.real())))));
    if (value.imag() != 0.0) logc = std::max(logc, static_cast<int>(std::ceil(std::log2(std::abs(value.imag())))));
  }
  return logc == std::numeric_limits<int>::min() ? 0 : logc;
}
__attribute__((noinline)) int reduced(Values& inverse, double scale) {
  double largest = 0;
  for (auto& value : inverse) {
    value *= scale;
    largest = std::max(largest, std::abs(value.real()));
    largest = std::max(largest, std::abs(value.imag()));
  }
  return largest == 0 ? 0 : static_cast<int>(std::ceil(std::log2(largest)));
}

int main() {
  int cases = 0;
  auto check = [&](const Values& values, double scale) {
    auto a = values, b = values;
    const int x = original(a, scale), y = reduced(b, scale);
    if (x != y || std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) != 0)
      throw std::runtime_error("range scan mismatch");
    for (const auto& v : a)
      if (!std::isfinite(v.real()) || !std::isfinite(v.imag()))
        throw std::runtime_error("probe left its finite scaled-input contract");
    ++cases;
  };
  check({}, 1); check(Values(32), std::ldexp(1.0, 59));
  // libm log2 rounding immediately around powers matters: replacing log2 with
  // ilogb + a mantissa test is not assumed equivalent to the current encoder.
  for (int exponent = -1074; exponent <= 1023; ++exponent) {
    const double power = std::ldexp(1.0, exponent);
    for (double v : {std::nextafter(power, 0.0), power,
                    std::nextafter(power, std::numeric_limits<double>::infinity())}) {
      if (!std::isfinite(v)) continue;
      check({{v, -0.0}, {0.0, -v}, {-v, v}}, 1);
    }
  }
  std::mt19937_64 random(0x594b434b5352414eULL);
  for (int n : {1, 2, 3, 31, 32, 33, 511, 512, 513, 32767, 32768, 32769}) {
    Values values(n);
    for (auto& v : values) {
      auto component = [&] {
        const auto fraction = static_cast<double>(random() >> 11) * 0x1p-53;
        return std::ldexp((fraction - .5) * 2, static_cast<int>(random() % 121) - 60);
      };
      v = {component(), component()};
    }
    const auto scale = std::ldexp(1.0, 59);
    for (double factor : {0.25, 1.0, scale, std::nextafter(scale, 0.0),
                          std::nextafter(scale, std::numeric_limits<double>::infinity())})
      check(values, factor);
  }
  Values input(32768), work;
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = {std::sin(i * 0.013 + 0.1) * .001, std::cos(i * 0.027 + 0.2) * .001};
  for (int i = 0; i < 4; ++i) { work = input; original(work, 0x1p59); work = input; reduced(work, 0x1p59); }
  std::cout << std::setprecision(17) << "{\"passed\":true,\"exact_cases\":" << cases
            << ",\"complex_slots\":32768,\"repetitions_per_sample\":31,\"samples\":[";
  int checksum = 0;
  for (int i = 0; i < 8; ++i) {
    const bool candidate = (i % 4 == 1 || i % 4 == 2);
    double seconds = 0;
    for (int repeat = 0; repeat < 31; ++repeat) {
      work = input; // Restore outside timing in both modes.
      const auto start = std::chrono::steady_clock::now();
      checksum += candidate ? reduced(work, 0x1p59) : original(work, 0x1p59);
      seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      if (std::bit_cast<uint64_t>(work[31].real()) != std::bit_cast<uint64_t>(input[31].real() * 0x1p59))
        throw std::runtime_error("scaled input was not restored");
    }
    if (i) std::cout << ',';
    std::cout << "{\"candidate\":" << (candidate ? "true" : "false") << ",\"seconds\":" << seconds / 31 << '}';
  }
  std::cout << "],\"checksum\":" << checksum
            << ",\"scope\":\"Extracted CPU range scan on finite scaled inputs only. Neither full encoding nor encrypted/model performance; no production change.\"}\n";
}
