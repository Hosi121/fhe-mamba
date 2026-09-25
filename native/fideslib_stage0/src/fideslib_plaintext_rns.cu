#include "fideslib_plaintext_rns.hpp"
#include <stdexcept>

namespace fhemamba {
namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__global__ void expand(const uint64_t* source, uint64_t* output, std::size_t n,
                       uint64_t source_modulus, uint64_t modulus, uint64_t reciprocal) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const auto value = source[i];
  const bool negative = value > source_modulus / 2;
  const auto magnitude = negative ? source_modulus - value : value;
  // floor(2^64 / q) underestimates floor(magnitude / q) by at most one.
  const auto quotient = __umul64hi(magnitude, reciprocal);
  auto residue = magnitude - quotient * modulus;
  if (residue >= modulus) residue -= modulus;
  output[i] = negative && residue ? modulus - residue : residue;
}
}  // namespace

PlaintextRnsWorkspace::~PlaintextRnsWorkspace() {
  if (!data_) return;
  int previous = 0;
  cudaGetDevice(&previous);
  cudaSetDevice(device_);
  cudaFree(data_);
  cudaSetDevice(previous);
}

auto PlaintextRnsWorkspace::upload(const void* coefficients, std::size_t bytes, int device)
    -> const uint64_t* {
  check(cudaSetDevice(device));
  if (data_ && device_ != device) throw std::invalid_argument("RNS workspace device changed");
  if (bytes > bytes_) {
    if (data_) check(cudaFree(data_));
    data_ = nullptr;
    check(cudaMalloc(&data_, bytes));
    bytes_ = bytes; device_ = device;
  }
  // The shared staging vector must be visible to every limb stream.
  check(cudaMemcpy(data_, coefficients, bytes, cudaMemcpyHostToDevice));
  check(cudaStreamSynchronize(nullptr));
  return static_cast<const uint64_t*>(data_);
}

void expand_plaintext_rns(const uint64_t* source, uint64_t* output, std::size_t n,
                          uint64_t source_modulus, uint64_t modulus, cudaStream_t stream) {
  if (modulus < 2 || modulus >= (uint64_t{1} << 61))
    throw std::invalid_argument("unsupported plaintext RNS modulus");
  const auto reciprocal = static_cast<uint64_t>((static_cast<__uint128_t>(1) << 64) / modulus);
  expand<<<(n + 255) / 256, 256, 0, stream>>>(source, output, n, source_modulus, modulus, reciprocal);
  check(cudaGetLastError());
}
}  // namespace fhemamba
