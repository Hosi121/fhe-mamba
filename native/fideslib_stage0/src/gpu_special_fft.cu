#include "gpu_special_fft.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fhemamba {
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

__global__ void real_input(const double* values, double2* output, unsigned count, unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = make_double2(i < count ? values[i] : 0.0, 0.0);
}

// Match OpenFHE's division, scale multiplication and llround, including ties
// away from zero. Compact range checks exclude its approximate-scaling branch.
// Only the status word comes back to the host. Three block reductions preserve
// the stock encoder's rejection of nonzero coefficients with max <= 0.5.
__global__ void round_coefficients(const double2* values, uint64_t* output,
                                  unsigned n, unsigned bits, double scale,
                                  uint64_t q0, unsigned* status) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  bool nonzero = false, sufficient_scale = false, invalid = false;
  if (i < n) {
    const auto v = values[bits ? __brev(i) >> (32 - bits) : 0];
    const double re = __dmul_rn(__ddiv_rn(v.x, static_cast<double>(n)), scale);
    const double im = __dmul_rn(__ddiv_rn(v.y, static_cast<double>(n)), scale);
    nonzero = re != 0.0 || im != 0.0;
    sufficient_scale = fabs(re) > 0.5 || fabs(im) > 0.5;
    invalid = !isfinite(re) || !isfinite(im) ||
        fabs(re) >= static_cast<double>(q0 / 2) || fabs(im) >= static_cast<double>(q0 / 2);
    if (!invalid) {
      const auto r = llround(re), s = llround(im);
      output[i] = r < 0 ? q0 - static_cast<uint64_t>(-r) : static_cast<uint64_t>(r);
      output[i + n] = s < 0 ? q0 - static_cast<uint64_t>(-s) : static_cast<uint64_t>(s);
    }
  }
  unsigned flags = __syncthreads_or(nonzero);
  flags |= static_cast<unsigned>(__syncthreads_or(sufficient_scale)) << 1;
  flags |= static_cast<unsigned>(__syncthreads_or(invalid)) << 2;
  if (threadIdx.x == 0) atomicOr(status, flags);
}

// A tile owns all inputs of the remaining butterflies. Preserve the original
// operation order, but keep ten stages in shared memory and launch them once.
// All threads reach every barrier, including for sizes smaller than a tile.
__global__ void butterfly_tail(double2* values, const double2* twiddles, unsigned size) {
  __shared__ double2 tile[1024];
  const unsigned t = threadIdx.x, base = blockIdx.x * size;
  for (unsigned j = t; j < size; j += blockDim.x) tile[j] = values[base + j];
  __syncthreads();
  for (unsigned len = size; len >= 2; len >>= 1) {
    if (t < size / 2) {
      const unsigned half = len / 2, j = t & (half - 1), i = 2 * (t - j) + j;
      const auto a = tile[i], b = tile[i + half], w = twiddles[half - 1 + j];
      const double vr = __dsub_rn(a.x, b.x), vi = __dsub_rn(a.y, b.y);
      tile[i] = make_double2(__dadd_rn(a.x, b.x), __dadd_rn(a.y, b.y));
      tile[i + half] = make_double2(
          __dsub_rn(__dmul_rn(vr, w.x), __dmul_rn(vi, w.y)),
          __dadd_rn(__dmul_rn(vi, w.x), __dmul_rn(vr, w.y)));
    }
    __syncthreads();
  }
  for (unsigned j = t; j < size; j += blockDim.x) values[base + j] = tile[j];
}
}

GpuSpecialInverseFFT::GpuSpecialInverseFFT(unsigned slots, bool tiled)
    : slots_(slots), log_slots_(0), tiled_(tiled) {
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
  check(cudaGetDevice(&device_));
  try {
    check(cudaMalloc(&data_,slots*sizeof(double2)));
    check(cudaMalloc(&output_,slots*sizeof(double2)));
    check(cudaMalloc(&twiddles_,slots*sizeof(double2)));
    check(cudaMalloc(&real_,slots*sizeof(double)));
    check(cudaMalloc(&coefficients_,2*slots*sizeof(uint64_t)));
    check(cudaMalloc(&status_,sizeof(unsigned)));
    check(cudaMemcpy(twiddles_,twiddles.data(),slots*sizeof(double2),cudaMemcpyHostToDevice));
  }catch(...){
    cudaFree(data_);cudaFree(output_);cudaFree(twiddles_);
    cudaFree(real_);cudaFree(coefficients_);cudaFree(status_);throw;
  }
}
GpuSpecialInverseFFT::~GpuSpecialInverseFFT(){
  int previous = 0;
  cudaGetDevice(&previous); cudaSetDevice(device_);
  cudaFree(data_);cudaFree(output_);cudaFree(twiddles_);
  cudaFree(real_);cudaFree(coefficients_);cudaFree(status_);
  cudaSetDevice(previous);
}

void GpuSpecialInverseFFT::select_device() const {
  int current = -1;
  check(cudaGetDevice(&current));
  if (current != device_) throw std::invalid_argument("GPU FFT device changed");
}

auto GpuSpecialInverseFFT::transform(const std::vector<std::complex<double>>& values)
    -> std::vector<std::complex<double>> {
  if(values.size()!=slots_)throw std::invalid_argument("FFT input size differs");
  for(auto v:values)if(!std::isfinite(v.real())||!std::isfinite(v.imag()))
    throw std::invalid_argument("nonfinite FFT input");
  select_device();
  check(cudaMemcpy(data_,values.data(),slots_*sizeof(double2),cudaMemcpyHostToDevice));
  execute();
  permute<<<(slots_+255)/256,256>>>(static_cast<const double2*>(data_),static_cast<double2*>(output_),slots_,log_slots_);
  check(cudaGetLastError());
  std::vector<std::complex<double>> result(slots_);
  check(cudaMemcpy(result.data(),output_,slots_*sizeof(double2),cudaMemcpyDeviceToHost));
  return result;
}

void GpuSpecialInverseFFT::execute() {
  const unsigned tail = tiled_ ? std::min(slots_, 1024u) : 1u;
  for(unsigned len=slots_;len>tail;len/=2)
    butterfly<<<(slots_/2+255)/256,256>>>(static_cast<double2*>(data_),static_cast<const double2*>(twiddles_),slots_,len);
  if (tail > 1)
    butterfly_tail<<<slots_ / tail, 512>>>(static_cast<double2*>(data_),
                                         static_cast<const double2*>(twiddles_), tail);
  check(cudaGetLastError());
}

const uint64_t* GpuSpecialInverseFFT::coefficients(const std::vector<double>& values,
                                                 double scale, uint64_t q0) {
  if (values.empty() || values.size() > slots_ || !std::isfinite(scale) || scale <= 0 ||
      q0 < 4 || q0 >= (uint64_t{1} << 61))
    throw std::invalid_argument("unsupported GPU coefficient parameters");
  select_device();
  check(cudaMemcpy(real_, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
  real_input<<<(slots_+255)/256,256>>>(static_cast<const double*>(real_),
      static_cast<double2*>(data_), values.size(), slots_);
  execute();
  check(cudaMemset(status_, 0, sizeof(unsigned)));
  round_coefficients<<<(slots_+255)/256,256>>>(static_cast<const double2*>(data_),
      static_cast<uint64_t*>(coefficients_), slots_, log_slots_, scale, q0,
      static_cast<unsigned*>(status_));
  check(cudaGetLastError());
  unsigned status = 0;
  check(cudaMemcpy(&status, status_, sizeof(status), cudaMemcpyDeviceToHost));
  if ((status & 4) || status == 1) return nullptr;
  return static_cast<const uint64_t*>(coefficients_);
}

}  // namespace fhemamba
