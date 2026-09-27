#include "gpu_fft.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
__global__ void butterfly(double2* values, const double2* twiddles, unsigned n, unsigned len) {
  const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n/2) return;
  const unsigned half = len/2, j = t % half, i = (t/half)*len+j;
  const auto a=values[i], b=values[i+half], w=twiddles[half-1+j];
  const double vr=__dsub_rn(a.x,b.x), vi=__dsub_rn(a.y,b.y);
  values[i]=make_double2(__dadd_rn(a.x,b.x),__dadd_rn(a.y,b.y));
  values[i+half]=make_double2(
      __dsub_rn(__dmul_rn(vr,w.x),__dmul_rn(vi,w.y)),
      __dadd_rn(__dmul_rn(vi,w.x),__dmul_rn(vr,w.y)));
}
__global__ void permute(const double2* values,double2* output,unsigned n,unsigned bits) {
  const unsigned t=blockIdx.x*blockDim.x+threadIdx.x;
  if(t>=n)return;
  const auto value=values[bits ? __brev(t)>>(32-bits) : 0];
  output[t]=make_double2(__ddiv_rn(value.x,static_cast<double>(n)),
                         __ddiv_rn(value.y,static_cast<double>(n)));
}
}

GpuSpecialInverseFFT::GpuSpecialInverseFFT(unsigned slots):slots_(slots),log_slots_(0) {
  if(!slots || (slots&(slots-1)) || slots>65536)throw std::invalid_argument("unsupported FFT size");
  while((1u<<log_slots_)<slots)++log_slots_;
  std::vector<unsigned> powers(slots);
  const unsigned m=4*slots;
  unsigned power=1;
  for(unsigned j=0;j<slots;++j){powers[j]=power;power=power*5%m;}
  std::vector<std::complex<double>> twiddles(slots);
  for(unsigned len=2;len<=slots;len*=2)for(unsigned j=0;j<len/2;++j){
    const auto index=(4*len-powers[j]%(4*len))*(m/(4*len));
    const double angle=2.0*M_PI*index/m;
    twiddles[len/2-1+j]={std::cos(angle),std::sin(angle)};
  }
  static_assert(sizeof(std::complex<double>)==sizeof(double2));
  try {
    check(cudaMalloc(&data_,slots*sizeof(double2)));
    check(cudaMalloc(&output_,slots*sizeof(double2)));
    check(cudaMalloc(&twiddles_,slots*sizeof(double2)));
    check(cudaMemcpy(twiddles_,twiddles.data(),slots*sizeof(double2),cudaMemcpyHostToDevice));
  }catch(...){cudaFree(data_);cudaFree(output_);cudaFree(twiddles_);throw;}
}
GpuSpecialInverseFFT::~GpuSpecialInverseFFT(){cudaFree(data_);cudaFree(output_);cudaFree(twiddles_);}

auto GpuSpecialInverseFFT::transform(const std::vector<std::complex<double>>& values)
    -> std::vector<std::complex<double>> {
  if(values.size()!=slots_)throw std::invalid_argument("FFT input size differs");
  for(auto v:values)if(!std::isfinite(v.real())||!std::isfinite(v.imag()))
    throw std::invalid_argument("nonfinite FFT input");
  check(cudaMemcpy(data_,values.data(),slots_*sizeof(double2),cudaMemcpyHostToDevice));
  for(unsigned len=slots_;len>=2;len/=2)
    butterfly<<<(slots_/2+255)/256,256>>>(static_cast<double2*>(data_),static_cast<const double2*>(twiddles_),slots_,len);
  permute<<<(slots_+255)/256,256>>>(static_cast<const double2*>(data_),static_cast<double2*>(output_),slots_,log_slots_);
  check(cudaGetLastError());
  std::vector<std::complex<double>> result(slots_);
  check(cudaMemcpy(result.data(),output_,slots_*sizeof(double2),cudaMemcpyDeviceToHost));
  return result;
}
