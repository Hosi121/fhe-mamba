#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstddef>

// Bit-reversed negacyclic NTT, with root_small = root_large^2.
// Down: even coefficients, i.e. (f(X) + f(-X))/2, evaluated at X^2.
// Up: f(X^2). These are public polynomial maps, not key switches.
void ring_map(const uint64_t* source, uint64_t* target, std::size_t small_n,
              uint64_t modulus, bool expand, cudaStream_t stream);
