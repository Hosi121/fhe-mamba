#include "gpu_fft.hpp"
#include <math/dftransform.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using Clock=std::chrono::steady_clock;
using FFT=lbcrypto::DiscreteFourierTransform;
int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::invalid_argument("gpu_fft_probe OUTPUT");
    std::ofstream out(argv[1]);out<<std::setprecision(15)<<"{\"samples\":[";
    int cases=0;bool first=true;
    for(unsigned n:{1u,16u,1024u,16384u,32768u,65536u}) {
      FFT::Initialize(4*n,n);GpuSpecialInverseFFT gpu(n);
      for(unsigned pattern=0;pattern<6;++pattern) {
        std::vector<std::complex<double>> v(n);
        for(unsigned j=0;j<n;++j){
          switch(pattern){
            case 0:v[j]={j%2?-0.0:0.0,-0.0};break;
            case 1:v[j]={std::sin(.71*j),std::cos(.31*j)};break;
            case 2:v[j]={j==n/2?.5:0,0};break;
            case 3:v[j]={j%2?-.25:.25,.125};break;
            case 4:v[j]={std::ldexp(double(j%7)-3,-1068),-0.0};break;
            case 5:v[j]={std::ldexp(double(j%7)-3,900),0};break;
          }
        }
        auto reference=v;FFT::FFTSpecialInv(reference,4*n);
        auto actual=gpu.transform(v);
        if(std::memcmp(reference.data(),actual.data(),n*sizeof(v[0])))
          throw std::runtime_error("FFT bits differ at n="+std::to_string(n)+" pattern="+std::to_string(pattern));
        ++cases;
      }
      if(n<1024)continue;
      std::vector<std::complex<double>> values(n);
      for(unsigned j=0;j<n;++j)values[j]={std::sin(.71*j),0};
      double cpu_seconds=0,gpu_seconds=0;
      for(int repeat=0;repeat<40;++repeat){
        auto reference=values;auto begin=Clock::now();FFT::FFTSpecialInv(reference,4*n);
        cpu_seconds+=std::chrono::duration<double>(Clock::now()-begin).count();
        begin=Clock::now();auto result=gpu.transform(values);
        gpu_seconds+=std::chrono::duration<double>(Clock::now()-begin).count();
        if(std::memcmp(reference.data(),result.data(),n*sizeof(values[0])))throw std::runtime_error("timed FFT mismatch");
      }
      if(!first) out<<',';
      first=false;
      out<<"{\"slots\":"<<n<<",\"cpu_seconds\":"<<cpu_seconds/40<<",\"gpu_roundtrip_seconds\":"<<gpu_seconds/40<<'}';
      std::cout<<"slots="<<n<<" cpu_seconds="<<cpu_seconds/40<<" gpu_roundtrip_seconds="<<gpu_seconds/40<<std::endl;
    }
    out<<"],\"passed\":true,\"bitwise_cases\":"<<cases<<",\"scope\":\"FFT only; transfer/allocation included for GPU, no encrypted-encoding or whole-model qualification\"}\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
