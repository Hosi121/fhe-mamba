#include "fideslib_plaintext_rns.hpp"
#include <NTT.cuh>
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

__global__ void expand_batch(const uint64_t* source, PlaintextRnsBatch batch,
                             std::size_t n, uint64_t source_modulus) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const auto j = blockIdx.y;
  const auto modulus = batch.modulus[j];
  const auto value = source[i];
  const bool negative = value > source_modulus / 2;
  const auto magnitude = negative ? source_modulus - value : value;
  auto residue = magnitude - __umul64hi(magnitude, batch.reciprocal[j]) * modulus;
  if (residue >= modulus) residue -= modulus;
  batch.output[j][i] = negative && residue ? modulus - residue : residue;
}

__global__ void expand_scaled(const uint64_t* source, uint64_t* output, std::size_t n,
                              uint64_t source_modulus, uint64_t modulus, uint64_t reciprocal,
                              uint64_t factor, uint64_t factor_shoup) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const auto value = source[i];
  const bool negative = value > source_modulus / 2;
  const auto magnitude = negative ? source_modulus - value : value;
  auto residue = magnitude - __umul64hi(magnitude, reciprocal) * modulus;
  if (residue >= modulus) residue -= modulus;
  if (negative && residue) residue = modulus - residue;
  // Shoup reduction: both operands < q < 2^61, so the difference is < 2q.
  residue = residue * factor - __umul64hi(residue, factor_shoup) * modulus;
  if (residue >= modulus) residue -= modulus;
  output[i] = residue;
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
    check(cudaMalloc(&data_, bytes + sizeof(void*)));
    bytes_ = bytes; device_ = device;
    // The allocation address is stable until the next growth. Set this table
    // only on growth, not with an extra transfer for every plaintext.
    check(cudaMemcpy(source_pointer(), &data_, sizeof(data_), cudaMemcpyHostToDevice));
  }
  // The shared staging vector must be visible to every limb stream.
  check(cudaMemcpy(data_, coefficients, bytes, cudaMemcpyHostToDevice));
  check(cudaStreamSynchronize(nullptr));
  return static_cast<const uint64_t*>(data_);
}

auto PlaintextRnsWorkspace::source_pointer() const -> void** {
  return reinterpret_cast<void**>(static_cast<unsigned char*>(data_) + bytes_);
}

void expand_plaintext_rns(const uint64_t* source, uint64_t* output, std::size_t n,
                          uint64_t source_modulus, uint64_t modulus, cudaStream_t stream) {
  if (modulus < 2 || modulus >= (uint64_t{1} << 61))
    throw std::invalid_argument("unsupported plaintext RNS modulus");
  const auto reciprocal = static_cast<uint64_t>((static_cast<__uint128_t>(1) << 64) / modulus);
  expand<<<(n + 255) / 256, 256, 0, stream>>>(source, output, n, source_modulus, modulus, reciprocal);
  check(cudaGetLastError());
}

void expand_plaintext_rns_batch(const uint64_t* source, const PlaintextRnsBatch& batch,
                                std::size_t n, uint64_t source_modulus, int limbs,
                                cudaStream_t stream) {
  if (limbs < 1 || limbs > kMaxRnsBatch) throw std::invalid_argument("invalid RNS batch size");
  expand_batch<<<dim3((n + 255) / 256, limbs), 256, 0, stream>>>(source, batch, n, source_modulus);
  check(cudaGetLastError());
}

void expand_scaled_plaintext_rns(const uint64_t* source, uint64_t* output, std::size_t n,
                                 uint64_t source_modulus, uint64_t modulus,
                                 uint64_t factor, cudaStream_t stream) {
  if (modulus < 2 || modulus >= (uint64_t{1} << 61))
    throw std::invalid_argument("unsupported scaled plaintext RNS modulus");
  const auto reciprocal = uint64_t((__uint128_t{1} << 64) / modulus);
  factor %= modulus;
  const auto shoup = uint64_t((__uint128_t{factor} << 64) / modulus);
  expand_scaled<<<(n + 255) / 256, 256, 0, stream>>>(source, output, n, source_modulus,
                                                  modulus, reciprocal, factor, shoup);
  check(cudaGetLastError());
}

void fused_plaintext_rns_ntt(const void* opaque_globals, void** source,
                             void** scratch, void** output, int log_n, int limbs,
                             int primeid_init, int source_primeid, cudaStream_t stream) {
  using namespace FIDESlib;
  const auto* globals = static_cast<const Global::Globals*>(opaque_globals);
  if (log_n < 10 || log_n > 20 || limbs < 1)
    throw std::invalid_argument("unsupported fused plaintext NTT geometry");
  const unsigned n = 1u << log_n;
  const unsigned first = 1u << ((log_n + 1) / 2 - 1);
  const unsigned second = 1u << (log_n / 2 - 1);
  // Match LimbPartition::ApplyNTT's uint64/Shoup geometry (M=4). Only the
  // first half uses NTT_RESCALE: its modulus switch is the signed RNS lift.
  NTT_<false, ALGO_SHOUP, NTT_RESCALE><<<dim3(n / (first * 8), limbs), first,
      8 * first * 10, stream>>>(globals, source, primeid_init, scratch, nullptr,
                               source_primeid, nullptr, nullptr);
  check(cudaGetLastError());
  NTT_<true, ALGO_SHOUP, NTT_NONE><<<dim3(n / (second * 8), limbs), second,
      8 * second * 10, stream>>>(globals, scratch, primeid_init, output, nullptr,
                                -1, nullptr, nullptr);
  check(cudaGetLastError());
}
}  // namespace fhemamba
