#include "gpu_special_fft.hpp"
#include <math/dftransform.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using fhemamba::GpuSpecialInverseFFT;
using Clock=std::chrono::steady_clock;
using FFT=lbcrypto::DiscreteFourierTransform;
int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::invalid_argument("gpu_fft_probe OUTPUT");
    std::ofstream out(argv[1]);out<<std::setprecision(15)<<"{\"samples\":[";
    int cases=0, coefficient_cases=0, small_scale_cases=0;bool first=true;
    for(unsigned n=1; n<=65536; n*=2) {
      FFT::Initialize(4*n,n);GpuSpecialInverseFFT gpu(n), tiled(n, true);
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
        actual=tiled.transform(v);
        if(std::memcmp(reference.data(),actual.data(),n*sizeof(v[0])))
          throw std::runtime_error("tiled FFT bits differ at n="+std::to_string(n)+" pattern="+std::to_string(pattern));
        cases+=2;
      }
      const uint64_t q0 = (uint64_t{1} << 60) - 93;
      for (int pattern=0; pattern<12; ++pattern) {
        const double scale = std::ldexp(1.0, 59);
        std::vector<double> values(pattern == 11 ? 1 : n);
        for (unsigned j=0; j<values.size(); ++j) {
          if (pattern < 8) values[j] = std::ldexp((pattern - 3) * 0.5, -59);
          else if (pattern == 8) values[j] = 0.125 * std::sin(j * 0.3);
          else if (pattern == 9) values[j] = j%2 ? -0.125 : 0.125;
          else if (pattern == 10) values[j] = 1e-100;
          else values[j] = -0.125;
        }
        std::vector<std::complex<double>> reference(values.begin(), values.end());
        reference.resize(n); FFT::FFTSpecialInv(reference, 4*n);
        double maximum = 0;
        std::vector<uint64_t> expected(2*n), actual(2*n);
        for (unsigned j=0; j<n; ++j) {
          reference[j] *= scale;
          maximum = std::max({maximum, std::abs(reference[j].real()), std::abs(reference[j].imag())});
          const auto re=std::llround(reference[j].real()), im=std::llround(reference[j].imag());
          expected[j] = re<0 ? q0-static_cast<uint64_t>(-re) : re;
          expected[j+n] = im<0 ? q0-static_cast<uint64_t>(-im) : im;
        }
        const auto* result=gpu.coefficients(values, scale, q0);
        if (maximum != 0 && maximum <= 0.5) {
          if (result) throw std::runtime_error("GPU FFT accepted a stock small-scale rejection");
          ++small_scale_cases;
        } else {
          if (!result || cudaMemcpy(actual.data(),result,2*n*sizeof(uint64_t),cudaMemcpyDeviceToHost) != cudaSuccess || actual!=expected)
            throw std::runtime_error("GPU rounded coefficients differ at n="+std::to_string(n)+" pattern="+std::to_string(pattern));
          ++coefficient_cases;
        }
      }
      if(n!=1024 && n!=16384 && n!=32768 && n!=65536)continue;
      std::vector<std::complex<double>> values(n);
      for(unsigned j=0;j<n;++j)values[j]={std::sin(.71*j),0};
      double cpu_seconds=0,gpu_seconds=0,tiled_seconds=0;
      for(int repeat=0;repeat<40;++repeat){
        auto reference=values;auto begin=Clock::now();FFT::FFTSpecialInv(reference,4*n);
        cpu_seconds+=std::chrono::duration<double>(Clock::now()-begin).count();
        begin=Clock::now();auto result=gpu.transform(values);
        gpu_seconds+=std::chrono::duration<double>(Clock::now()-begin).count();
        if(std::memcmp(reference.data(),result.data(),n*sizeof(values[0])))throw std::runtime_error("timed FFT mismatch");
        begin=Clock::now();result=tiled.transform(values);
        tiled_seconds+=std::chrono::duration<double>(Clock::now()-begin).count();
        if(std::memcmp(reference.data(),result.data(),n*sizeof(values[0])))throw std::runtime_error("timed tiled FFT mismatch");
      }
      if(!first) out<<',';
      first=false;
      out<<"{\"slots\":"<<n<<",\"cpu_seconds\":"<<cpu_seconds/40<<",\"gpu_roundtrip_seconds\":"<<gpu_seconds/40<<",\"tiled_roundtrip_seconds\":"<<tiled_seconds/40<<'}';
      std::cout<<"slots="<<n<<" cpu_seconds="<<cpu_seconds/40<<" gpu_roundtrip_seconds="<<gpu_seconds/40<<" tiled_roundtrip_seconds="<<tiled_seconds/40<<std::endl;
    }
    out<<"],\"passed\":true,\"bitwise_cases\":"<<cases
       <<",\"coefficient_cases\":"<<coefficient_cases<<",\"small_scale_cases\":"<<small_scale_cases
       <<",\"scope\":\"FFT and compact coefficient rounding; transfer/allocation included for GPU, no encrypted-arithmetic or whole-model qualification\"}\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
