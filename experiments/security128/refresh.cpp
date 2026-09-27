// Encrypted qualification at the secure geometry, with client-only verification.
#include "fideslib_security.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace fideslib;
using Ct=Ciphertext<DCRTPoly>;
using Clock=std::chrono::steady_clock;

static void sync_gpu() {
  const auto e=cudaDeviceSynchronize();
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
static double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now()-start).count();
}

int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::invalid_argument("usage: security_refresh OUTPUT HYBRID_DIGITS");
    const int digits=std::stoi(argv[2]);
    const auto setup=Clock::now();
    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecurityLevel(HEStd_128_classic);p.SetSecretKeyDist(UNIFORM_TERNARY);
    p.SetCKKSDataType(REAL);p.SetRingDim(131072);p.SetBatchSize(65536);
    p.SetMultiplicativeDepth(44);p.SetScalingModSize(59);p.SetFirstModSize(60);
    p.SetScalingTechnique(FLEXIBLEAUTO);p.SetKeySwitchTechnique(HYBRID);
    p.SetNumLargeDigits(digits);p.SetDevices({0});
    p.SetPlaintextAutoload(false);p.SetCiphertextAutoload(true);
    auto c=GenCryptoContext(p);
    auto audit=fhemamba::audit_ckks_context(c);audit.require_classical128();
    for(auto feature:{PKE,KEYSWITCH,LEVELEDSHE,ADVANCEDSHE,FHE})c->Enable(feature);
    auto keys=c->KeyGen();c->EvalMultKeyGen(keys.secretKey);
    c->EvalRotateKeyGen(keys.secretKey,{1,-1,32,-32});
    c->EvalBootstrapSetup({4,4},{0,0},65536,0,true);
    c->EvalBootstrapKeyGen(keys.secretKey,65536);
    c->LoadContext(keys.publicKey);sync_gpu();
    const double setup_seconds=elapsed(setup);
    auto refresh=[&](Ct input) {
      while(input->GetNoiseScaleDeg()>1)c->RescaleInPlace(input);
      auto first=c->EvalBootstrap(input);sync_gpu();
      auto a=input->Clone(),b=first->Clone();
      while(b->GetNoiseScaleDeg()>1)c->RescaleInPlace(b);
      const auto level=std::max(a->GetLevel(),b->GetLevel());
      for(auto* x:{&a,&b})while((*x)->GetLevel()<level) {
        c->EvalMultInPlace(*x,1.0);
        while((*x)->GetNoiseScaleDeg()>1)c->RescaleInPlace(*x);
      }
      auto residual=c->EvalSub(a,b);c->EvalMultInPlace(residual,4096.0);
      while(residual->GetNoiseScaleDeg()>1)c->RescaleInPlace(residual);
      auto second=c->EvalBootstrap(residual);sync_gpu();
      c->EvalMultInPlace(second,1.0/4096.0);c->EvalMultInPlace(first,1.0);
      auto result=c->EvalAdd(first,second);sync_gpu();return result;
    };
    auto error=[&](const Ct& input,const std::vector<double>& expected) {
      Plaintext plain;auto copy=input->Clone();c->Decrypt(keys.secretKey,copy,&plain);
      plain->SetLength(expected.size());const auto actual=plain->GetRealPackedValue();
      double worst=0;
      for(std::size_t i=0;i<expected.size();++i) {
        if(!std::isfinite(actual[i]))throw std::runtime_error("nonfinite secure result");
        worst=std::max(worst,std::abs(actual[i]-expected[i]));
      }
      return worst;
    };
    struct Sample {int width,level,output_level;double seconds,error,following_error;};
    std::vector<Sample> samples;
    double max_error=0;
    int rotations=0;
    for(int width:{1,32,768,1536,65536})for(int level:{0,18,35}) {
      std::vector<double> values(65536);
      for(int i=0;i<width;++i)values[i]=std::sin(.37*i+.2)/8;
      auto plain=c->MakeCKKSPackedPlaintext(values);
      auto input=c->Encrypt(keys.publicKey,plain);input->SetLevel(level);
      const auto begin=Clock::now();auto output=refresh(input);const auto seconds=elapsed(begin);
      const auto e=error(output,values);max_error=std::max(max_error,e);
      auto product=c->EvalMult(output,output);sync_gpu();
      auto squared=values;for(auto& x:squared)x*=x;
      const auto next=error(product,squared);max_error=std::max(max_error,next);
      for(int shift:{1,-1,32,-32}) {
        auto rotated=c->EvalRotate(output,shift);sync_gpu();
        auto expected=values;
        for(int i=0;i<65536;++i)expected[i]=values[(i+shift+65536)%65536];
        max_error=std::max(max_error,error(rotated,expected));++rotations;
      }
      samples.push_back({width,level,int(output->GetLevel()),seconds,e,next});
      std::cout<<"width="<<width<<" level="<<level<<" refresh="<<seconds<<" error="<<e<<std::endl;
      if(max_error>1e-6)throw std::runtime_error("secure refresh numerical gate failed");
    }
    fhemamba::audit_ckks_context(c).require_classical128();
    std::ofstream out(argv[1]);
    out<<std::setprecision(15)<<"{\"passed\":true,\"security\":\"128-classic\",\"security_audit\":";
    audit.write_json(out);
    out<<",\"setup_seconds\":"<<setup_seconds<<",\"refresh_cases\":"<<samples.size()
       <<",\"following_products\":"<<samples.size()<<",\"rotation_cases\":"<<rotations
       <<",\"max_abs_error\":"<<max_error<<",\"tolerance\":1e-6,\"bootstrap_passes\":2,\"samples\":[";
    for(std::size_t i=0;i<samples.size();++i) {
      const auto& s=samples[i];if(i)out<<',';
      out<<"{\"width\":"<<s.width<<",\"input_level\":"<<s.level<<",\"output_level\":"<<s.output_level
         <<",\"seconds\":"<<s.seconds<<",\"error\":"<<s.error<<",\"following_error\":"<<s.following_error<<'}';
    }
    out<<"]}\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
