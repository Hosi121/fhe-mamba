#pragma once
#include <complex>
#include <cstddef>
#include <vector>

// A qualification prototype for the full-packing public inverse transform.
// The butterfly order matches OpenFHE, using explicit round-to-nearest double
// operations, so CPU/GPU output bits can be checked before integrating encoding.
class GpuSpecialInverseFFT {
 public:
  explicit GpuSpecialInverseFFT(unsigned slots);
  ~GpuSpecialInverseFFT();
  GpuSpecialInverseFFT(const GpuSpecialInverseFFT&) = delete;
  GpuSpecialInverseFFT& operator=(const GpuSpecialInverseFFT&) = delete;
  std::vector<std::complex<double>> transform(const std::vector<std::complex<double>>& values);
 private:
  unsigned slots_, log_slots_;
  void* data_ = nullptr;
  void* output_ = nullptr;
  void* twiddles_ = nullptr;
};
