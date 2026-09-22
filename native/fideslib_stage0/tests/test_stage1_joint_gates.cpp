#include "stage1_joint_gates.hpp"
#include "stage1_mamba2_depth.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using Value = std::vector<double>;
using namespace fhemamba::stage1;

struct Ops {
  bool periodic = false;
  auto multiply(const Value& a, const Value& b) { auto c=a; for (std::size_t i=0;i<c.size();++i)c[i]*=b[i]; return c; }
  auto add(const Value& a, const Value& b) { auto c=a; for (std::size_t i=0;i<c.size();++i)c[i]+=b[i]; return c; }
  auto subtract(const Value& a, const Value& b) { auto c=a; for (std::size_t i=0;i<c.size();++i)c[i]-=b[i]; return c; }
  auto scalar_add(const Value& a, double b) {
    auto c=a;
    for (std::size_t i=0;i<c.size();++i)
      if (!periodic || (i%8>=2 && i%8<5)) c[i]+=b;
    return c;
  }
  auto scale(const Value& a, double b) { auto c=a; for (auto& x:c)x*=b; return c; }
  auto coefficient_product(const Value& a, const Value& row) {
    if (!periodic) return multiply(a,joint_coefficient_slots(row,16,8,2,2));
    const auto packed=joint_periodic_coefficient_slots(row,16,8,2,2);
    auto result=a;
    for (std::size_t i=0;i<result.size();++i)result[i]*=packed[i%packed.size()];
    return result;
  }
  auto constant(const Value& row) {
    if (!periodic) return joint_coefficient_slots(row,16,8,2,2);
    return coefficient_product(joint_coefficient_slots(Value(3,1.0),16,8,2,2),row);
  }
};

auto main() -> int {
  Ops ops;
  Ops periodic_ops{true};
  for (int degree : {0,1,2,5,31,64,191,384,768,1024}) {
    JointGateSpec spec{{-1,-2,-3},{1,2,3},{},{},{1,2,3}};
    for (int k=0;k<=degree;++k)
      for (int h=0;h<3;++h) spec.p.push_back(std::sin(k*7.1+h+0.2)/(1+k*k)*(k%11==7?1e-14:1));
    spec.q.assign(spec.p.begin(),spec.p.begin()+3*(degree/2+1));
    for (double& x:spec.q)x*=0.3;
    for (int sample=0;sample<=20;++sample) {
      Value input(16,0.9);
      for (int stream=0;stream<2;++stream)
        for (int h=0;h<3;++h)input[stream*8+2+h]=-1+2.0*((sample+h+stream)%21)/20;
      auto [p,q]=evaluate_joint_roots(input,spec,ops);
      auto masked_input=input;
      for (int slot=0;slot<16;++slot)
        if (slot%8<2 || slot%8>=5) masked_input[slot]=0.0;
      auto [periodic_p,periodic_q]=evaluate_joint_roots(masked_input,spec,periodic_ops);
      for (const auto& pair : {std::pair{spec.p,p},std::pair{spec.q,q},
                              std::pair{spec.p,periodic_p},std::pair{spec.q,periodic_q}}) {
        for (int slot=0;slot<16;++slot) {
          const int h=slot%8-2;
          double expected=0;
          if(h>=0&&h<3) {
            Value coefficients;
            for(std::size_t k=h;k<pair.first.size();k+=3)coefficients.push_back(pair.first[k]);
            expected=cheb_clenshaw_host(coefficients,input[slot]);
          }
          if(std::abs(pair.second[slot]-expected)>1e-10)
            throw std::runtime_error("vector PS differs from independent Clenshaw or leaks lanes");
        }
      }
    }
  }
  // The model's 24 heads have period 32, including offsets that wrap around
  // that period and streams whose inactive padding must stay zero.
  for (int heads : {1,3,24,32,33}) {
    Value row(heads);
    for (int h=0;h<heads;++h)row[h]=h+0.125;
    for (int offset : {0,17,63}) {
      const auto full=joint_coefficient_slots(row,512,128,offset,3);
      const auto packed=joint_periodic_coefficient_slots(row,512,128,offset,3);
      for (int stream=0;stream<3;++stream)
        for (int h=0;h<heads;++h) {
          const int slot=stream*128+offset+h;
          if (full[slot]!=packed[slot%packed.size()])
            throw std::runtime_error("periodic coefficient differs on a head lane");
        }
    }
  }
  bool rejected=false;
  try { (void)joint_periodic_coefficient_slots(Value(3,1.0),24,6,1,2); }
  catch (const std::runtime_error&) { rejected=true; }
  if (!rejected)throw std::runtime_error("misaligned stream period was accepted");
  return 0;
}
