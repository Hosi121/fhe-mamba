// Research adapter: composite 59-bit arithmetic and encrypted two-pass refresh.
// The upstream test client is not a production client or a security claim.
#include "UserInterface.h"
#include "extension/BootContext.h"
#include "composite_params.hpp"
#include "stock32_params.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
using namespace cheddar;
using Clock = std::chrono::steady_clock;
void sync_gpu() { auto e=cudaDeviceSynchronize(); if(e) throw std::runtime_error(cudaGetErrorString(e)); }
double seconds(Clock::time_point start) { sync_gpu(); return std::chrono::duration<double>(Clock::now()-start).count(); }
template<class W> int run(const std::string& output, bool diagnose, bool stock) {
 using Ct=Ciphertext<W>;
 auto start=Clock::now();
 std::vector<std::pair<int,int>> levels;
 for(int i=0;i<=33;++i) levels.emplace_back((i+1)*CompositePrimes<W>::limbs,0);
 int initial_level=21, base_scale_bits=57;
 auto main_primes=CompositePrimes<W>::q(),aux_primes=CompositePrimes<W>::p();
 std::vector<W> terminal_primes;
 if constexpr(sizeof(W)==4) {if(stock) {
  levels=Stock32Primes::levels();main_primes=Stock32Primes::q();
  aux_primes=Stock32Primes::p();terminal_primes=Stock32Primes::t();
  initial_level=16;base_scale_bits=40;
 }}
 // q0 <= 62 bits is an upstream requirement. Scale at level zero is 57 bits;
 // the ordinary rescale primes are 59 bits, converging to a 59-bit scale.
 Parameter<W> param(16,std::ldexp(1.,base_scale_bits),initial_level,levels,main_primes,aux_primes,terminal_primes);
 param.SetDenseHammingWeight(43690); param.SetSparseHammingWeight(32);
 auto cc=BootContext<W>::Create(param,BootParameter(param.max_level_,4,3));
 UserInterface<W> client(cc);
 cc->PrepareEvalMod(); cc->PrepareEvalSpecialFFT(32768,BootVariant::kImaginaryRemoving);
 EvkRequest req; cc->AddRequiredRotations(req,32768); client.PrepareRotationKey(req);
 // Explicit level: upstream's default -1 denotes its short SSE base here.
 for(int rotation : {1,2,3,4,8,12}) client.PrepareRotationKey(rotation,initial_level-2);
 double setup=seconds(start);
 std::ofstream report(output);
 report << std::setprecision(15) << "{\"word_bits\":"<<sizeof(W)*8
   <<",\"setup_seconds\":"<<setup<<",\"base_scale_bits\":"<<base_scale_bits<<",\"ordinary_scale_bits\":"<<std::log2(param.GetScale(initial_level))
   <<",\"ordinary_rescale_bits\":"<<(stock?40:59)<<",\"stock_profile_control\":"<<(stock?"true":"false")
   <<",\"diagnostic_client_validations\":"<<(diagnose?13:0)
   <<",\"ring_dimension\":65536,\"slots\":32768,\"bootstrap_passes\":2"
   <<",\"residual_multiplier\":4096,\"intermediate_evaluator_decryptions\":0,\"security\":\"not-set\",\"upstream_client\":\"test-only\",\"samples\":[";
 bool passed=true;
 for(int iteration=-1;iteration<(diagnose?0:3);++iteration) {
  std::vector<Complex> input(32768),expected(32768);
  for(int i=0;i<32768;++i) input[i]=std::sin(i*.71+.1)/2;
  Plaintext<W> plain;cc->encoder_.Encode(plain,initial_level,param.GetScale(initial_level),input);
  Ct encrypted;client.Encrypt(encrypted,plain);sync_gpu();
  start=Clock::now();
  Ct square,biased,multiplied,product,rotated;
  cc->HMult(square,encrypted,encrypted,client.GetMultiplicationKey(),true);
  Constant<W> bias,weight;
  cc->encoder_.EncodeConstant(bias,initial_level-1,square.GetScale(),.125);
  cc->Add(biased,square,bias);
  cc->encoder_.EncodeConstant(weight,initial_level-1,square.GetScale(),.75);
  cc->Mult(multiplied,biased,weight);cc->Rescale(product,multiplied);
  // Sixteen public, nonconstant diagonals in a 4 x 4 BSGS transform.
  // Rotate plaintext masks backwards so the giant rotation restores them.
  std::vector<Ct> babies(4);
  cc->Copy(babies[0],product);
  for(int b=1;b<4;++b)cc->HRot(babies[b],product,client.GetRotationKey(b),b);
  auto weight_at=[](int diagonal,int slot){return .03+.005*((slot+diagonal)%7);};
  for(int g=0;g<4;++g) {
   Ct inner;
   for(int b=0;b<4;++b) {
    std::vector<Complex> mask(32768);
    for(int k=0;k<32768;++k)mask[k]=weight_at(4*g+b,(k-4*g+32768)%32768);
    Plaintext<W> pt;cc->encoder_.Encode(pt,initial_level-2,product.GetScale(),mask);
    Ct raw,term;cc->Mult(raw,babies[b],pt);cc->Rescale(term,raw);
    if(b==0)inner=std::move(term);else{Ct sum;cc->Add(sum,inner,term);inner=std::move(sum);}
   }
   Ct giant;
   if(g)cc->HRot(giant,inner,client.GetRotationKey(4*g),4*g);else giant=std::move(inner);
   if(g==0)rotated=std::move(giant);else{Ct sum;cc->Add(sum,rotated,giant);rotated=std::move(sum);}
  }
  double ordinary_before=seconds(start);
  start=Clock::now();
  Ct low;cc->LevelDown(low,rotated,0);
  Ct first,first_low,error,scaled_error,correction,divided,correction_scaled,first_aligned,refreshed;
  cc->Boot(first,low,client.GetEvkMap());
  cc->LevelDown(first_low,first,0);cc->Sub(error,low,first_low);
  Constant<W> up;cc->encoder_.EncodeConstant(up,0,1.,4096.);
  cc->Mult(scaled_error,error,up);
  cc->Boot(correction,scaled_error,client.GetEvkMap());
  const int level=param.NPToLevel(correction.GetNP());
  Constant<W> down;cc->encoder_.EncodeConstant(down,level,correction.GetScale(),1./4096.);
  cc->Mult(correction_scaled,correction,down);cc->Rescale(divided,correction_scaled);
  cc->LevelDown(first_aligned,first,level-1);cc->Add(refreshed,first_aligned,divided);
  double refresh=seconds(start);
  start=Clock::now();Ct result;cc->HMult(result,refreshed,refreshed,client.GetMultiplicationKey(),true);
  double ordinary_after=seconds(start);
  if(diagnose) {
   // Client-only validation of retained subcircuit outputs after the complete
   // encrypted computation. These values never feed back into evaluation.
   for(const auto& item:std::vector<std::pair<const char*,const Ct*>>{
       {"input",&encrypted},{"square",&square},{"product",&product},{"bsgs",&rotated},
       {"low",&low},{"first",&first},{"first_low",&first_low},{"error",&error},
       {"scaled_error",&scaled_error},{"correction",&correction},{"divided",&divided},
       {"refreshed",&refreshed},{"result",&result}}) {
    Plaintext<W> pt;client.Decrypt(pt,*item.second);std::vector<Complex> v;cc->encoder_.Decode(v,pt);
    double maximum=0;for(auto x:v)maximum=std::max(maximum,std::abs(x));
    std::cout<<"diagnostic "<<item.first<<" level="<<param.NPToLevel(item.second->GetNP())
      <<" scale="<<std::log2(item.second->GetScale())<<" max="<<maximum<<" first="<<v[0]<<std::endl;
   }
  }
  Plaintext<W> result_plain;client.Decrypt(result_plain,result);
  std::vector<Complex> actual;cc->encoder_.Decode(actual,result_plain);
  double worst=0;
  for(int i=0;i<32768;++i) {
   Complex y=0;
   for(int d=0;d<16;++d){auto x=input[(i+d)%32768];y+=(x*x+.125)*.75*weight_at(d,i);}
   expected[i]=y*y;
   double error=std::abs(actual[i]-expected[i]);
   if(!std::isfinite(error))throw std::runtime_error("non-finite output");
   worst=std::max(worst,error);
  }
  passed &= worst<=1e-6;
  if(iteration!=-1)report<<',';
  report<<"{\"iteration\":"<<iteration<<",\"ordinary_seconds\":"<<ordinary_before+ordinary_after
    <<",\"refresh_boundary_seconds\":"<<refresh<<",\"refreshed_level\":"<<param.NPToLevel(refreshed.GetNP())
    <<",\"output_level\":"<<param.NPToLevel(result.GetNP())<<",\"max_abs_error\":"<<worst<<'}'<<std::flush;
  std::cout<<"iteration="<<iteration<<" ordinary="<<ordinary_before+ordinary_after<<" refresh="<<refresh<<" error="<<worst<<std::endl;
 }
 report<<"],\"passed\":"<<(passed?"true":"false")<<",\"tolerance\":1e-6}\n";
 return passed?0:3;
}
int main(int argc,char**argv) {
 try { if(argc<3||argc>5)throw std::invalid_argument("usage: composite_probe 32|64 OUTPUT [--diagnose] [--stock32]");
  bool diagnose=false,stock=false;
  for(int i=3;i<argc;++i){std::string flag=argv[i];if(flag=="--diagnose")diagnose=true;else if(flag=="--stock32")stock=true;else throw std::invalid_argument(flag);}
  if(std::string(argv[1])=="32")return run<uint32_t>(argv[2],diagnose,stock);
  if(std::string(argv[1])=="64"&&!stock)return run<uint64_t>(argv[2],diagnose,false);
  throw std::invalid_argument("word size");
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 2;}
}
