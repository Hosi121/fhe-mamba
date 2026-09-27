#pragma once
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

// Exact-order transform for full-packing public CKKS operands.
// The butterfly order matches OpenFHE, using explicit round-to-nearest double
// operations, so CPU/GPU output bits can be checked before integrating encoding.
namespace fhemamba {
class GpuSpecialInverseFFT {
 public:
  explicit GpuSpecialInverseFFT(unsigned slots, bool tiled = false);
  ~GpuSpecialInverseFFT();
  GpuSpecialInverseFFT(const GpuSpecialInverseFFT&) = delete;
  GpuSpecialInverseFFT& operator=(const GpuSpecialInverseFFT&) = delete;
  std::vector<std::complex<double>> transform(const std::vector<std::complex<double>>& values);
  // Return degree-one residues modulo q0 in device memory, valid until the next
  // call. No coefficient array crosses back to the CPU. Return null for inputs
  // requiring the stock encoder's small-scale or approximate-scaling branches.
  // The caller must first prove compact_plaintext_fits and consume before reuse.
  const uint64_t* coefficients(const std::vector<double>& values, double scale, uint64_t q0);
 private:
  void execute();
  void select_device() const;
  unsigned slots_, log_slots_;
  bool tiled_;
  int device_ = -1;
  void* data_ = nullptr;
  void* output_ = nullptr;
  void* twiddles_ = nullptr;
  void* real_ = nullptr;
  void* coefficients_ = nullptr;
  void* status_ = nullptr;
};

}  // namespace fhemamba
