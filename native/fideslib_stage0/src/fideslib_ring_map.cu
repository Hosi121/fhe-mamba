#include "fideslib_ring_map.hpp"
#include <stdexcept>

namespace {
__global__ void map_ntt(const uint64_t* in, uint64_t* out, std::size_t n,
                        uint64_t q, uint64_t reciprocal, bool expand) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (expand) {
    out[2*i] = out[2*i+1] = in[i];
  } else {
    // Canonicalize each input even if a backend operation left lazy residues.
    auto a = in[2*i], b = in[2*i+1];
    a -= __umul64hi(a, reciprocal) * q;
    b -= __umul64hi(b, reciprocal) * q;
    if (a >= q) a -= q;
    if (b >= q) b -= q;
    auto sum = a + b;
    if (sum >= q) sum -= q;
    out[i] = (sum + (sum & 1) * q) >> 1;
  }
}
}

void ring_map(const uint64_t* source, uint64_t* target, std::size_t small_n,
              uint64_t modulus, bool expand, cudaStream_t stream) {
  if (!small_n || modulus < 3 || !(modulus & 1) || modulus >= (uint64_t{1} << 61))
    throw std::invalid_argument("unsupported ring-map modulus or dimension");
  const auto reciprocal = uint64_t((__uint128_t{1} << 64) / modulus);
  map_ntt<<<(small_n + 255)/256, 256, 0, stream>>>(source, target, small_n,
                                               modulus, reciprocal, expand);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
