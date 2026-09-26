#pragma once
#include "fideslib_ring_map.hpp"
#include <fideslib.hpp>
#include <CKKS/Ciphertext.cuh>
#include <CKKS/Context.cuh>
#include <CKKS/KeySwitchingKey.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <openfhe.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <type_traits>


namespace fhemamba::dual_ring {
namespace gpu = FIDESlib::CKKS;
using namespace fideslib;
using Ct = Ciphertext<DCRTPoly>;
using Context = CryptoContext<DCRTPoly>;
using CpuContext = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using Clock = std::chrono::steady_clock;

inline void sync_gpu() {
  auto e = cudaDeviceSynchronize();
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
inline double seconds(Clock::time_point t) { sync_gpu(); return std::chrono::duration<double>(Clock::now()-t).count(); }
inline CpuContext cpu(const Context& c) { return std::any_cast<CpuContext>(c->cpu); }
inline std::shared_ptr<gpu::Ciphertext> device(const Ct& c) {
  return std::static_pointer_cast<gpu::Ciphertext>(c->parent_context->GetDeviceCiphertext(c->gpu));
}
inline auto cp(const Context& c) {
  return std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cpu(c)->GetCryptoParameters());
}

// OpenFHE's host NTT cache is keyed by modulus, not ring degree. Only setup,
// public reference checks and final client decoding use this serialized helper.
inline void select_cpu(const Context& c) {
  intnat::ChineseRemainderTransformFTTNat<lbcrypto::NativeVector> ntt;
  ntt.Reset();
  for (const auto& params : {cp(c)->GetElementParams(), cp(c)->GetParamsP()})
    for (const auto& limb : params->GetParams())
      ntt.PreCompute(limb->GetRootOfUnity(), limb->GetCyclotomicOrder(), limb->GetModulus());
}

inline Context context(int n) {
  CCParams<CryptoContextCKKSRNS> p;
  p.SetSecurityLevel(HEStd_NotSet); p.SetSecretKeyDist(UNIFORM_TERNARY);
  p.SetCKKSDataType(REAL); p.SetRingDim(n); p.SetBatchSize(n/2);
  p.SetMultiplicativeDepth(44); p.SetScalingModSize(59); p.SetFirstModSize(60);
  p.SetScalingTechnique(FLEXIBLEAUTO); p.SetKeySwitchTechnique(HYBRID);
  p.SetNumLargeDigits(3); p.SetDevices({0});
  p.SetPlaintextAutoload(false); p.SetCiphertextAutoload(true);
  auto c = GenCryptoContext(p);
  for (auto f : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE}) c->Enable(f);
  return c;
}

// Preserve every Q modulus and flexible scale of the full model context.
inline void match_small_parameters(const Context& small, const Context& large) {
  std::vector<lbcrypto::NativeInteger> q, roots;
  for (const auto& p : cp(large)->GetElementParams()->GetParams()) {
    q.push_back(p->GetModulus());
    roots.push_back(p->GetRootOfUnity().ModMul(p->GetRootOfUnity(), p->GetModulus()));
  }
  auto p = cp(small);
  p->SetElementParams(std::make_shared<lbcrypto::DCRTPoly::Params>(65536, q, roots));
  p->PrecomputeCRTTables(p->GetKeySwitchTechnique(), p->GetScalingTechnique(),
      p->GetEncryptionTechnique(), p->GetMultiplicationTechnique(), p->GetNumPartQ(),
      p->GetAuxBits(), p->GetExtraBits());
  for (int i=0; i<=44; ++i)
    if (p->GetScalingFactorReal(i) != cp(large)->GetScalingFactorReal(i))
      throw std::runtime_error("different flexible scale");
}

// Every transfer includes an encrypted key switch in the large ring. The small
// secret is embedded as s(X^2), never substituted for the dense refresh secret.
class RingBridge {
  Context large_, small_;
  Ct large_template_, small_template_;
  std::string large_id_, small_id_;
  std::unique_ptr<gpu::KeySwitchingKey> down_, up_;
 public:
  RingBridge(Context large, Context small, const KeyPair<DCRTPoly>& lk,
             const KeyPair<DCRTPoly>& sk, Ct lt, Ct st)
      : large_(large), small_(small), large_template_(lt), small_template_(st) {
    if (cpu(large)->GetRingDimension()!=65536 || cpu(small)->GetRingDimension()!=32768 ||
        large->devices != std::vector<int>{0} || small->devices != std::vector<int>{0})
      throw std::invalid_argument("dual ring requires N=65536/32768 on GPU 0");
    auto lsk = std::any_cast<lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>(lk.secretKey->pimpl);
    auto ssk = std::any_cast<lbcrypto::PrivateKey<lbcrypto::DCRTPoly>>(sk.secretKey->pimpl);
    large_id_ = lsk->GetKeyTag(); small_id_ = ssk->GetKeyTag();
    // NTT embedding is exact because both contexts use squared compatible roots.
    lbcrypto::DCRTPoly lifted(cp(large)->GetElementParams(), Format::EVALUATION, true);
    const auto& s = ssk->GetPrivateElement();
    for (std::size_t j=0; j<lifted.GetNumOfElements(); ++j)
      for (std::size_t i=0; i<32768; ++i) {
        lifted.GetAllElements().at(j)[2*i] = s.GetElementAtIndex(j)[i];
        lifted.GetAllElements().at(j)[2*i+1] = s.GetElementAtIndex(j)[i];
      }
    auto embedded = std::make_shared<lbcrypto::PrivateKeyImpl<lbcrypto::DCRTPoly>>(cpu(large));
    embedded->SetPrivateElement(std::move(lifted)); embedded->SetKeyTag(small_id_);
    select_cpu(large);
    auto& gc = std::any_cast<gpu::Context&>(large->gpu);
    auto make_key = [&](auto from, auto to, const std::string& source_id) {
      auto key = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
          cpu(large)->GetScheme()->KeySwitchGen(from, to));
      auto raw = gpu::GetKeySwitchKey(key);
      raw.keyid = source_id;
      auto out = std::make_unique<gpu::KeySwitchingKey>(gc);
      out->Initialize(raw); sync_gpu(); return out;
    };
    down_ = make_key(lsk, embedded, large_id_);
    up_ = make_key(embedded, lsk, small_id_);
  }

  Ct map(const Ct& input, bool expand) const {
    const auto& dest = expand ? large_ : small_;
    const auto& prototype = expand ? large_template_ : small_template_;
    auto source = device(input);
    auto& gc = std::any_cast<gpu::Context&>(dest->gpu);
    sync_gpu(); gpu::SetCurrentContext(gc); sync_gpu();
    auto target = std::make_shared<gpu::Ciphertext>(gc);
    const int level = source->getLevel();
    target->c0.grow(level); target->c1.grow(level);
    for (auto pair : {std::pair{&source->c0, &target->c0}, std::pair{&source->c1, &target->c1}}) {
      for (int j=0; j<=level; ++j) {
        auto a = source->cc.limbGPUid.at(j), b = target->cc.limbGPUid.at(j);
        const uint64_t* from = nullptr; uint64_t* to = nullptr;
        std::visit([&](auto& limb) {
          if constexpr(std::is_same_v<decltype(limb.v.data), uint64_t*>) from = limb.v.data;
        }, pair.first->GPU.at(a.x).limb.at(a.y));
        cudaStream_t stream{};
        std::visit([&](auto& limb) {
          if constexpr(std::is_same_v<decltype(limb.v.data), uint64_t*>) to = limb.v.data;
          stream = limb.stream.ptr();
        }, pair.second->GPU.at(b.x).limb.at(b.y));
        if (!from || !to || source->cc.prime.at(j).p != target->cc.prime.at(j).p)
          throw std::runtime_error("incompatible GPU ring-map limb");
        ring_map(from, to, 32768, source->cc.prime.at(j).p, expand, stream);
      }
    }
    target->NoiseFactor = source->NoiseFactor; target->NoiseLevel = source->NoiseLevel;
    target->slots = expand ? 32768 : 16384; target->keyID = small_id_;
    sync_gpu();
    auto out = std::make_shared<CiphertextImpl<DCRTPoly>>(Context(dest));
    out->cpu = prototype->cpu; out->need_lazy_copy = true;
    out->gpu = dest->RegisterDeviceCiphertext(std::move(target)); out->loaded = true;
    return out;
  }
  long long down_calls=0, up_calls=0;
  Ct down(const Ct& input) {
    ++down_calls;
    auto temp = input->Clone();
    device(temp)->keySwitch(*down_); sync_gpu();
    return map(temp, false);
  }
  Ct up(const Ct& input) {
    ++up_calls;
    auto out = map(input, true);
    device(out)->keySwitch(*up_); sync_gpu(); device(out)->keyID = large_id_;
    return out;
  }
};

} // namespace fhemamba::dual_ring
