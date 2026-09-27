// Client-side qualification only. The production correction helper has no key
// or decryption interface; its integer operation is checked against raw residues.
#include "fideslib_refresh_correction.hpp"
#include "fideslib_security.hpp"
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <complex>
#include <fstream>
#include <iostream>
#include <vector>

using namespace fideslib;
using Ct = Ciphertext<DCRTPoly>;
namespace gpu = FIDESlib::CKKS;
using Clock = std::chrono::steady_clock;
static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
static void sync_gpu() {
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
static auto device(const Ct& x) {
  return std::static_pointer_cast<gpu::Ciphertext>(x->parent_context->GetDeviceCiphertext(x->gpu));
}
static auto snapshot(const Ct& x) {
  gpu::RawCipherText raw{};
  sync_gpu(); device(x)->store(raw); sync_gpu();
  // FIDESlib store() writes residues and scale metadata, but does not fill
  // RawCipherText::moduli. Read the public context's retained Q prefix.
  const auto cpu = std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>>(x->parent_context->cpu);
  const auto& parameters = cpu->GetCryptoParameters()->GetElementParams()->GetParams();
  require(raw.numRes > 0 && static_cast<std::size_t>(raw.numRes) <= parameters.size(), "invalid residue count");
  for (int j=0;j<raw.numRes;++j)raw.moduli.push_back(parameters[j]->GetModulus().ConvertToInt());
  return raw;
}
static void same_metadata(const Ct& a, const Ct& b) {
  auto x = device(a), y = device(b);
  require(a->GetLevel() == b->GetLevel() && x->NoiseLevel == y->NoiseLevel &&
      x->NoiseFactor == y->NoiseFactor && x->slots == y->slots && x->keyID == y->keyID,
      "correction changed level/scale metadata");
}
static void exact_integer_sum(const gpu::RawCipherText& a, const gpu::RawCipherText& b,
                              const gpu::RawCipherText& result) {
  require(a.numRes == b.numRes && a.numRes == result.numRes && a.moduli == result.moduli &&
      a.N == result.N && a.Noise == result.Noise && a.NoiseLevel == result.NoiseLevel &&
      a.slots == result.slots && a.keyid == result.keyid && a.format == result.format,
      "raw metadata mismatch");
  for (int component = 0; component < 2; ++component) {
    const auto& x = component ? a.sub_1 : a.sub_0;
    const auto& y = component ? b.sub_1 : b.sub_0;
    const auto& z = component ? result.sub_1 : result.sub_0;
    for (std::size_t j = 0; j < x.size(); ++j)
      for (std::size_t k = 0; k < x[j].size(); ++k)
        require(z[j][k] == (static_cast<__uint128_t>(x[j][k]) * 4096 + y[j][k]) % a.moduli[j],
                "integer merge differs from exact modular oracle");
  }
}

int main(int argc, char** argv) {
  try {
    require(argc == 3, "usage: refresh_correction_probe OUTPUT real|complex");
    const std::string type = argv[2];
    require(type == "real" || type == "complex", "invalid data type");
    const bool complex = type == "complex";
    const int slots = 65536;
    const auto setup = Clock::now();
    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecurityLevel(HEStd_128_classic); p.SetSecretKeyDist(UNIFORM_TERNARY);
    p.SetCKKSDataType(complex ? COMPLEX : REAL); p.SetRingDim(131072); p.SetBatchSize(slots);
    p.SetMultiplicativeDepth(44); p.SetScalingModSize(59); p.SetFirstModSize(60);
    p.SetScalingTechnique(FLEXIBLEAUTO); p.SetKeySwitchTechnique(HYBRID);
    p.SetNumLargeDigits(4); p.SetDevices({0});
    p.SetPlaintextAutoload(false); p.SetCiphertextAutoload(true);
    auto cc = GenCryptoContext(p);
    const auto audit = fhemamba::audit_ckks_context(cc); audit.require_classical128();
    for (auto f : {PKE, KEYSWITCH, LEVELEDSHE, ADVANCEDSHE, FHE}) cc->Enable(f);
    auto keys = cc->KeyGen(); cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalRotateKeyGen(keys.secretKey, {1, 32, 1024, 4096});
    // The existing S2C-first backend accepts only full real packing. Complex
    // arithmetic still qualifies the shared integer primitive, without asking
    // this probe to add a new bootstrap mode or alter its configuration.
    if (!complex) {
      cc->EvalBootstrapSetup({4,4}, {0,0}, slots, 0, true);
      cc->EvalBootstrapKeyGen(keys.secretKey, slots);
    }
    cc->LoadContext(keys.publicKey); sync_gpu();
    const auto setup_seconds = std::chrono::duration<double>(Clock::now() - setup).count();
    auto encrypt = [&](const std::vector<std::complex<double>>& values) {
      auto plain = cc->MakeCKKSPackedPlaintext(values);
      return cc->Encrypt(keys.publicKey, plain);
    };
    auto scaled = [&](const Ct& x, double scale) {
      auto result = x->Clone(); cc->EvalMultInPlace(result, scale); sync_gpu(); return result;
    };
    auto decrypt = [&](const Ct& x) {
      auto copy = x->Clone(); Plaintext plain; cc->Decrypt(keys.secretKey, copy, &plain);
      plain->SetLength(slots); return plain->GetCKKSPackedValue();
    };
    double maximum_normalized_error = 0, maximum_absolute_error = 0, maximum_difference = 0;
    auto check = [&](const Ct& x, const std::vector<std::complex<double>>& expected, double bound) {
      auto actual = decrypt(x); double worst = 0;
      for (int j = 0; j < slots; ++j) {
        require(std::isfinite(actual[j].real()) && std::isfinite(actual[j].imag()), "nonfinite output");
        worst = std::max(worst, std::abs(actual[j] - expected[j]));
      }
      maximum_absolute_error = std::max(maximum_absolute_error, worst);
      maximum_normalized_error = std::max(maximum_normalized_error, worst / std::max(1.0, bound));
      require(worst / std::max(1.0, bound) <= 1e-6, "normalized error gate failed");
      return actual;
    };
    std::vector<std::complex<double>> av(slots), bv(slots);
    for (int j = 0; j < slots; ++j) {
      av[j] = {.8 * std::sin(.37*j + .2), complex ? .2 * std::cos(.31*j) : 0};
      bv[j] = {.5 * std::cos(.17*j), complex ? .1 * std::sin(.11*j) : 0};
    }
    auto original_a = encrypt(av), original_b = encrypt(bv);
    int exact_cases = 0, rejection_cases = 0, refresh_cases = 0, extraction_cases = 0, following_products = 0;
    for (int level : {0,18,35}) for (int degree : {1,2}) {
      auto a = original_a->Clone(), b = original_b->Clone();
      a->SetLevel(level); b->SetLevel(level);
      if (degree == 2) { a = scaled(a, 1.0); b = scaled(b, 1.0); }
      auto before_a = snapshot(a), before_b = snapshot(b);
      auto result = a;
      require(fhemamba::merge_refresh_correction(result, b, sync_gpu), "compatible integer merge declined");
      exact_integer_sum(before_a, before_b, snapshot(result));
      same_metadata(result, a);
      auto after_a = snapshot(a), after_b = snapshot(b);
      require(before_a.sub_0 == after_a.sub_0 && before_a.sub_1 == after_a.sub_1 &&
          before_b.sub_0 == after_b.sub_0 && before_b.sub_1 == after_b.sub_1, "shared input changed");
      std::vector<std::complex<double>> expected(slots);
      for (int j=0;j<slots;++j) expected[j]=av[j]+bv[j]/4096.0;
      check(scaled(result,1.0/4096.0), expected, 1.0); ++exact_cases;
      std::cout<<"exact_case="<<exact_cases<<" level="<<level<<" degree="<<degree<<std::endl;
    }
    auto same = original_a;
    require(!fhemamba::merge_refresh_correction(same, same, sync_gpu), "same handle was accepted"); ++rejection_cases;
    auto wrong_level = original_b->Clone(); wrong_level->SetLevel(1);
    require(!fhemamba::merge_refresh_correction(same, wrong_level, sync_gpu), "level mismatch accepted"); ++rejection_cases;
    auto wrong_scale = original_b->Clone(); device(wrong_scale)->NoiseFactor *= 2;
    require(!fhemamba::merge_refresh_correction(same, wrong_scale, sync_gpu), "scale mismatch accepted"); ++rejection_cases;
    auto wrong_degree = scaled(original_b,1.0);
    require(!fhemamba::merge_refresh_correction(same, wrong_degree, sync_gpu), "degree mismatch accepted"); ++rejection_cases;
    auto wrong_slots = original_b->Clone(); wrong_slots->SetSlots(slots/2);
    require(!fhemamba::merge_refresh_correction(same, wrong_slots, sync_gpu), "slot mismatch accepted"); ++rejection_cases;
    auto components = [&](Ct input) {
      while (input->GetNoiseScaleDeg()>1) cc->RescaleInPlace(input);
      auto first=cc->EvalBootstrap(input); sync_gpu();
      auto a=input->Clone(), b=first->Clone();
      while(b->GetNoiseScaleDeg()>1) cc->RescaleInPlace(b);
      const auto target=std::max(a->GetLevel(),b->GetLevel());
      for(auto* x:{&a,&b}) while((*x)->GetLevel()<target) {
        *x=scaled(*x,1.0);
        while((*x)->GetNoiseScaleDeg()>1) cc->RescaleInPlace(*x);
      }
      auto residual=scaled(cc->EvalSub(a,b),4096.0);
      while(residual->GetNoiseScaleDeg()>1) cc->RescaleInPlace(residual);
      auto second=cc->EvalBootstrap(residual); sync_gpu();
      return std::pair{first,second};
    };
    struct Block { int offset,width; double bound; };
    const std::vector<std::vector<Block>> layouts{
      {{0,1,.5},{1,24,8},{32,32,64},{1024,768,256},{4096,1536,1024}},
      {{0,slots,1}},
    };
    if (!complex) for (const auto& layout : layouts) for (int level : {0,18,35}) {
      std::vector<std::complex<double>> values(slots);
      for (const auto& block : layout) for (int j=0;j<block.width;++j) values[block.offset+j]=av[block.offset+j];
      auto input=encrypt(values); input->SetLevel(level);
      auto [first,second]=components(input);
      auto merged=first;
      require(fhemamba::merge_refresh_correction(merged,second,sync_gpu), "bootstrap metadata does not permit correction merging");
      same_metadata(first,merged); ++refresh_cases;
      for (const auto& block : layout) {
        auto rotated=[&](const Ct& x) { auto r=block.offset?cc->EvalRotate(x,block.offset):x->Clone();sync_gpu();return r; };
        std::vector<double> mask(slots);
        std::fill_n(mask.begin(),block.width,block.bound);
        auto apply=[&](const Ct& x,const std::vector<double>& v) {
          auto plain=cc->MakeCKKSPackedPlaintext(v,1,x->GetLevel(),nullptr,slots);
          cc->LoadPlaintext(plain);auto out=cc->EvalMult(x,plain);sync_gpu();return out;
        };
        auto reference_a=apply(rotated(first),mask);
        for(auto& v:mask) v/=4096.0;
        auto reference_b=apply(rotated(second),mask);
        auto reference=cc->EvalAdd(reference_a,reference_b);sync_gpu();
        auto candidate=apply(rotated(merged),mask);
        same_metadata(reference,candidate);
        require(candidate->GetLevel()==18 && candidate->GetNoiseScaleDeg()==2,"changed refreshed depth");
        std::vector<std::complex<double>> expected(slots);
        for(int j=0;j<block.width;++j)expected[j]=values[block.offset+j]*block.bound;
        const auto ref=check(reference,expected,block.bound),actual=check(candidate,expected,block.bound);
        for(int j=0;j<slots;++j)maximum_difference=std::max(maximum_difference,std::abs(ref[j]-actual[j])/std::max(1.0,block.bound));
        require(maximum_difference<=1e-6,"merged/reference difference gate failed");
        auto product=cc->EvalMult(candidate,candidate);sync_gpu();
        for(auto& v:expected)v*=v;
        check(product,expected,block.bound*block.bound);
        ++extraction_cases; ++following_products;
      }
      std::cout<<"refresh_case="<<refresh_cases<<" level="<<level<<" max_normalized_error="<<maximum_normalized_error<<std::endl;
    }
    fhemamba::audit_ckks_context(cc).require_classical128();
    std::ofstream out(argv[1]); out<<std::setprecision(15);
    out<<"{\"passed\":true,\"type\":\""<<type<<"\",\"scope\":\""
       <<(complex?"complex integer primitive only; existing S2C-first requires REAL":"real integer primitive and S2C-first correction circuit")
       <<"\",\"security_audit\":";audit.write_json(out);
    out<<",\"setup_seconds\":"<<setup_seconds<<",\"exact_rns_cases\":"<<exact_cases
       <<",\"metadata_rejection_cases\":"<<rejection_cases<<",\"refresh_cases\":"<<refresh_cases
       <<",\"extraction_cases\":"<<extraction_cases<<",\"following_products\":"<<following_products
       <<",\"max_normalized_error\":"<<maximum_normalized_error<<",\"max_abs_error_including_scaled_products\":"<<maximum_absolute_error
       <<",\"max_normalized_difference\":"<<maximum_difference<<",\"normalized_tolerance\":1e-6}\n";
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
