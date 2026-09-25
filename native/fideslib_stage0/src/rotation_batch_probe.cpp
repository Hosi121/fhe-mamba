#include "fideslib_rotation_batch.hpp"
#include <CKKS/Ciphertext.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <cuda_runtime_api.h>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace fideslib;
using Ct = Ciphertext<DCRTPoly>;
using Clock = std::chrono::steady_clock;
static void sync_gpu() { if (cudaDeviceSynchronize()) throw std::runtime_error("CUDA sync"); }
static bool same(const FIDESlib::CKKS::RawCipherText& a, const FIDESlib::CKKS::RawCipherText& b) {
  return a.sub_0 == b.sub_0 && a.sub_1 == b.sub_1 && a.moduli == b.moduli &&
      a.numRes == b.numRes && a.N == b.N && a.format == b.format &&
      a.Noise == b.Noise && a.NoiseLevel == b.NoiseLevel && a.slots == b.slots;
}
int main(int argc, char** argv) {
 try {
  if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--mamba2"))
    throw std::invalid_argument("usage: rotation_batch_probe OUTPUT [--mamba2]");
  bool mamba2 = argc == 3;
  CCParams<CryptoContextCKKSRNS> p;
  p.SetSecurityLevel(HEStd_NotSet); p.SetSecretKeyDist(mamba2 ? SPARSE_TERNARY : UNIFORM_TERNARY);
  p.SetCKKSDataType(mamba2 ? COMPLEX : REAL);
  p.SetRingDim(65536); p.SetBatchSize(32768); p.SetMultiplicativeDepth(44);
  p.SetScalingModSize(59); p.SetFirstModSize(60); p.SetScalingTechnique(FLEXIBLEAUTO);
  p.SetKeySwitchTechnique(HYBRID); p.SetNumLargeDigits(3); p.SetDevices({0});
  p.SetPlaintextAutoload(false); p.SetCiphertextAutoload(true);
  auto cc = GenCryptoContext(p);
  for (auto feature : {PKE, KEYSWITCH, LEVELEDSHE}) cc->Enable(feature);
  auto keys = cc->KeyGen(); cc->EvalMultKeyGen(keys.secretKey);
  std::vector<int32_t> key_steps;
  for (int step = 1; step < 32768; step *= 2) { key_steps.push_back(step); key_steps.push_back(-step); }
  cc->EvalRotateKeyGen(keys.secretKey, key_steps); cc->LoadContext(keys.publicKey); sync_gpu();
  auto snapshot = [&](const Ct& value) {
    FIDESlib::CKKS::RawCipherText raw{};
    auto gpu = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(cc->GetDeviceCiphertext(value->gpu));
    sync_gpu(); gpu->store(raw); sync_gpu(); return raw;
  };
  auto scalar = [&](const Ct& input, const std::vector<int>& offsets, bool naf) {
    std::vector<Ct> output;
    for (int offset : offsets) {
      if (fhemamba::rotation_steps(offset, 32768, naf).empty()) {
        output.push_back(input);
        continue;
      }
      auto value = input->Clone();
      for (int step : fhemamba::rotation_steps(offset, 32768, naf)) {
        cc->EvalRotateInPlace(value, step); sync_gpu();
      }
      output.push_back(value);
    }
    return output;
  };
  std::vector<std::complex<double>> values(32768);
  for (int i=0;i<32768;++i) values[i]={std::sin(i*.71+.1)/8,mamba2 ? std::cos(i*.3)/16 : 0};
  auto plain=cc->MakeCKKSPackedPlaintext(values);
  auto encrypted=cc->Encrypt(keys.publicKey,plain);
  const std::vector<std::vector<int>> families{
    {}, {0,0,32768}, {3,7,15,31,-3,-7,-15,-31,3}, {0,24,48,72,96,120,144,168,192,216,240}};
  int exact_cases=0;
  fhemamba::RotationBatchStats stats;
  for (int level : {0,18,34}) for (bool scaled : {false,true}) for (bool naf : {false,true}) {
    auto input=encrypted->Clone(); input->SetLevel(level);
    if (scaled) cc->EvalMultInPlace(input,.75);
    const auto original=snapshot(input);
    for (const auto& offsets : families) {
      auto expected=scalar(input,offsets,naf);
      auto actual=fhemamba::hoisted_rotation_batch(cc,input,offsets,32768,naf,sync_gpu,stats);
      for (std::size_t i=0;i<actual.size();++i) {
        if (!same(snapshot(expected[i]),snapshot(actual[i])) ||
            expected[i]->GetLevel()!=actual[i]->GetLevel() ||
            expected[i]->GetNoiseScaleDeg()!=actual[i]->GetNoiseScaleDeg())
          throw std::runtime_error("rotation RNS mismatch case="+std::to_string(exact_cases));
        ++exact_cases;
      }
      if (!same(original,snapshot(input))) throw std::runtime_error("live source mutated");
    }
  }
  struct Sample { int level,block,iteration; bool hoisted; double ms; };
  std::vector<Sample> samples;
  std::vector<int> offsets; for (int i=0;i<32;++i) offsets.push_back(24*i);
  for (int level : {18,34}) {
    auto input=encrypted->Clone(); input->SetLevel(level);
    for (int block=0;block<4;++block) for (int iteration=-1;iteration<4;++iteration) {
      bool hoisted=block==1||block==2;
      auto start=Clock::now();
      auto out=hoisted ? fhemamba::hoisted_rotation_batch(cc,input,offsets,32768,true,sync_gpu,stats)
                       : scalar(input,offsets,true);
      sync_gpu();
      double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
      samples.push_back({level,block,iteration,hoisted,ms});
    }
  }
  std::ofstream report(argv[1]);
  report << std::setprecision(15) << "{\"passed\":true,\"exact_rns_cases\":" << exact_cases
    << ",\"live_inputs_unchanged\":true,\"mamba2\":" << (mamba2?"true":"false")
    << ",\"sibling_batches\":" << stats.sibling_batches << ",\"samples\":[";
  for (std::size_t i=0;i<samples.size();++i) {
    const auto& s=samples[i]; if(i)report<<',';
    report << "{\"level\":"<<s.level<<",\"block\":"<<s.block<<",\"iteration\":"<<s.iteration
      <<",\"hoisted\":"<<(s.hoisted?"true":"false")<<",\"ms\":"<<s.ms<<'}';
  }
  report << "]}\n";
  std::cout << "passed exact_rns_cases=" << exact_cases << std::endl;
 } catch(const std::exception& e) { std::cerr << e.what() << std::endl; return 2; }
}
