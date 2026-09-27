#pragma once

#include <fideslib.hpp>
#include <openfhe.h>
#include <any>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <stdexcept>

namespace fhemamba {

// Published 2024 Security Guidelines, Table 5.2, uniform ternary / sigma 3.19.
// https://cic.iacr.org/p/1/4/26/pdf
// The bound applies to Q*P for HYBRID, including public evaluation keys.
// Do not extrapolate to unlisted dimensions or apply it to sparse secrets.
inline uint32_t classical128_modulus_bound(uint32_t ring) {
  switch (ring) {
    case 1024: return 26;
    case 2048: return 53;
    case 4096: return 106;
    case 8192: return 214;
    case 16384: return 430;
    case 32768: return 868;
    case 65536: return 1747;
    case 131072: return 3523;
    default: return 0;
  }
}

struct CkksSecurityAudit {
  uint32_t ring = 0, q_bits = 0, p_bits = 0, qp_bits = 0;
  uint32_t q_towers = 0, p_towers = 0, digits = 0, guideline_bound = 0;
  double sigma = 0;
  bool library_classical128 = false, uniform_ternary = false, hybrid = false;
  bool satisfies_classical128() const {
    return library_classical128 && uniform_ternary && hybrid &&
        std::isfinite(sigma) && sigma >= 3.19 && guideline_bound &&
        q_towers && p_towers && qp_bits && qp_bits <= guideline_bound;
  }
  void require_classical128() const {
    if (!satisfies_classical128())
      throw std::invalid_argument("classical-128 CKKS audit failed: require HEStd_128_classic, uniform ternary, sigma >= 3.19 and actual HYBRID Q*P within the published dimension bound");
  }
  void write_json(std::ostream& out) const {
    out << "{\"ring_dimension\":" << ring << ",\"q_bits\":" << q_bits
        << ",\"p_bits\":" << p_bits << ",\"qp_bits\":" << qp_bits
        << ",\"q_towers\":" << q_towers << ",\"p_towers\":" << p_towers
        << ",\"hybrid_digits\":" << digits << ",\"error_sigma\":" << sigma
        << ",\"guideline_max_qp_bits\":" << guideline_bound
        << ",\"library_classical128\":" << (library_classical128 ? "true" : "false")
        << ",\"uniform_ternary\":" << (uniform_ternary ? "true" : "false")
        << ",\"hybrid\":" << (hybrid ? "true" : "false")
        << ",\"passed\":" << (satisfies_classical128() ? "true" : "false")
        << ",\"basis\":\"2024 Security Guidelines Table 5.2; classical RLWE parameter bound\"}";
  }
};

inline auto audit_ckks_context(const fideslib::CryptoContext<fideslib::DCRTPoly>& context)
    -> CkksSecurityAudit {
  const auto cpu = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(context->cpu);
  const auto p = std::dynamic_pointer_cast<lbcrypto::CryptoParametersCKKSRNS>(cpu->GetCryptoParameters());
  if (!p || !p->GetParamsP()) throw std::invalid_argument("CKKS audit requires HYBRID parameters");
  CkksSecurityAudit a;
  a.ring = cpu->GetRingDimension();
  const auto& q = p->GetElementParams()->GetModulus();
  const auto& special = p->GetParamsP()->GetModulus();
  a.q_bits = q.GetMSB(); a.p_bits = special.GetMSB();
  // Exact integer bit length: a conservative ceil(log2(Q*P)), not a sum
  // of rounded per-prime bit lengths or a ciphertext's current level.
  a.qp_bits = (q * special).GetMSB();
  a.q_towers = p->GetElementParams()->GetParams().size();
  a.p_towers = p->GetParamsP()->GetParams().size();
  a.digits = p->GetNumPartQ(); a.sigma = p->GetDistributionParameter();
  a.guideline_bound = classical128_modulus_bound(a.ring);
  a.library_classical128 = p->GetStdLevel() == lbcrypto::HEStd_128_classic;
  a.uniform_ternary = p->GetSecretKeyDist() == lbcrypto::UNIFORM_TERNARY;
  a.hybrid = p->GetKeySwitchTechnique() == lbcrypto::HYBRID;
  return a;
}

}  // namespace fhemamba
