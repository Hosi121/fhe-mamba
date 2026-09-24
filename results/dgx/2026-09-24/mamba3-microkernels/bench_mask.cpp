#include "plaintext_mask.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using Clock = std::chrono::steady_clock;
extern "C" __attribute__((noinline))
std::vector<double> mask_scalar(const std::vector<double>& input, int shift) {
  const int n = input.size(); std::vector<double> out(n);
  if (!n) return out;
  shift = (shift % n + n) % n;
  for (int i = 0; i < n; ++i) out[(i + shift) % n] = input[i];
  return out;
}
extern "C" __attribute__((noinline))
std::vector<double> mask_spans(const std::vector<double>& input, int shift) {
  return fhemamba::rotate_plaintext_mask(input, shift);
}
int main(int argc, char** argv) {
  const std::string order = argc > 1 ? argv[1] : "abba";
  if (order != "abba" && order != "baab") throw std::invalid_argument("use abba or baab");
  double checksum = 0; bool comma = false;
  std::cout << std::setprecision(12) << "{\"compiler\":\"" << __VERSION__ << "\",\"order\":\"" << order
            << "\",\"allocation_included\":true,\"samples\":[";
  for (int n : {24, 768, 1024, 32768}) {
    std::vector<double> input(n);
    for (int i = 0; i < n; ++i) input[i] = std::sin(i * .7 + .3);
    for (int shift : {0, 1, -1, n - 1, n + 1, 127, -233}) {
      auto a = mask_scalar(input, shift), b = mask_spans(input, shift);
      if (std::memcmp(a.data(), b.data(), n * sizeof(double))) throw std::runtime_error("coefficient bits differ");
    }
    for (int block = 0; block < 4; ++block) {
      auto fn = order[block] == 'a' ? mask_scalar : mask_spans;
      for (int j = -10; j < 100; ++j) {
        const int shift = ((j + 12) * 127) % n;
        auto start = Clock::now();
        auto out = fn(input, shift);
        const auto ns = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
        checksum += out[(j + 10) % n];
        if (j < 0) continue;
        if (comma) std::cout << ',';
        comma = true;
        std::cout << "{\"size\":" << n << ",\"block\":" << block << ",\"iteration\":" << j
                  << ",\"mode\":\"" << (order[block] == 'a' ? "scalar" : "spans")
                  << "\",\"ns\":" << ns << '}';
      }
    }
  }
  std::cout << "],\"bitwise_equal\":true,\"checksum\":" << checksum << "}\n";
}
