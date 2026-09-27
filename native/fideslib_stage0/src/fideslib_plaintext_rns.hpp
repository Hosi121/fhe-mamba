#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>


namespace fhemamba {

// One serial evaluator owns one staging allocation. Each load synchronizes
// before returning, so neither the next plaintext nor its destructor can race.
class PlaintextRnsWorkspace {
 public:
  ~PlaintextRnsWorkspace();
  PlaintextRnsWorkspace() = default;
  PlaintextRnsWorkspace(const PlaintextRnsWorkspace&) = delete;
  PlaintextRnsWorkspace& operator=(const PlaintextRnsWorkspace&) = delete;
  auto upload(const void* coefficients, std::size_t bytes, int device) -> const uint64_t*;
  // Device pointer table for kernels that broadcast one compact source.
  auto source_pointer() const -> void**;

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
  int device_ = -1;
};

void expand_plaintext_rns(const uint64_t* source, uint64_t* output, std::size_t n,
                          uint64_t source_modulus, uint64_t modulus,
                          cudaStream_t stream);

// Expand signed degree-one coefficients, then apply OpenFHE's exact degree-two
// residue multiplier. This is integer modular arithmetic, not a new rounding.
void expand_scaled_plaintext_rns(const uint64_t* source, uint64_t* output, std::size_t n,
                                 uint64_t source_modulus, uint64_t modulus,
                                 uint64_t factor, cudaStream_t stream);

inline constexpr int kMaxRnsBatch = 64;
struct PlaintextRnsBatch {
  uint64_t* output[kMaxRnsBatch]{};
  uint64_t modulus[kMaxRnsBatch]{};
  uint64_t reciprocal[kMaxRnsBatch]{};
};
void expand_plaintext_rns_batch(const uint64_t* source, const PlaintextRnsBatch& batch,
                                std::size_t n, uint64_t source_modulus, int limbs,
                                cudaStream_t stream);

// Reuse the pinned input-switching NTT first half, followed by the ordinary
// second half. No rescale epilogue runs and no RNS input array is materialized.
void fused_plaintext_rns_ntt(const void* globals, void** source,
                             void** scratch, void** output, int log_n, int limbs,
                             int primeid_init, int source_primeid, cudaStream_t stream);

}  // namespace fhemamba
