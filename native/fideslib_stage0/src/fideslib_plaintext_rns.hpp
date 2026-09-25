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

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0;
  int device_ = -1;
};

void expand_plaintext_rns(const uint64_t* source, uint64_t* output, std::size_t n,
                          uint64_t source_modulus, uint64_t modulus,
                          cudaStream_t stream);

}  // namespace fhemamba
