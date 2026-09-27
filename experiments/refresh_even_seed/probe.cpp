// Standalone encrypted GPU feasibility probe; no trained-model speed claim.
#include "fideslib_dual_ring.hpp"
#include "EvenBootstrapSeed.cuh"
using namespace fhemamba::dual_ring;

int exact_maps(const Context& large, const Context& small) {
  int cases=0;
  for(int tower : {0,20,44}) {
    auto lp=cp(large)->GetElementParams()->GetParams().at(tower);
    auto sp=cp(small)->GetElementParams()->GetParams().at(tower);
    auto q=lp->GetModulus().ConvertToInt();
    lbcrypto::NativePoly full(lp,Format::COEFFICIENT,true), half(sp,Format::COEFFICIENT,true);
    for(int i=0;i<65536;++i) full[i] = i%3 ? uint64_t(i)*7919%q : q-1;
    for(int i=0;i<32768;++i) half[i] = full[2*i];
    select_cpu(large); full.SetFormat(Format::EVALUATION);
    select_cpu(small); half.SetFormat(Format::EVALUATION);
    std::vector<uint64_t> source(65536),actual(65536);
    for(int i=0;i<65536;++i)source[i]=full[i].ConvertToInt();
    uint64_t *from=nullptr,*to=nullptr;
    auto check=[](cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); };
    check(cudaMalloc(&from,65536*sizeof(uint64_t))); check(cudaMalloc(&to,65536*sizeof(uint64_t)));
    check(cudaMemcpy(from,source.data(),source.size()*sizeof(uint64_t),cudaMemcpyHostToDevice));
    ring_map(from,to,32768,q,false,nullptr); sync_gpu();
    check(cudaMemcpy(actual.data(),to,32768*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    for(int i=0;i<32768;++i)
      if(actual[i]!=half[i].ConvertToInt())throw std::runtime_error("NTT projection differs from coefficient oracle");
    ++cases;
    // Independently form f(X^2) in the coefficient domain and transform it.
    select_cpu(small); half.SetFormat(Format::COEFFICIENT);
    full=lbcrypto::NativePoly(lp,Format::COEFFICIENT,true);
    for(int i=0;i<32768;++i) full[2*i]=half[i];
    select_cpu(large); full.SetFormat(Format::EVALUATION);
    ring_map(to,from,32768,q,true,nullptr); sync_gpu();
    check(cudaMemcpy(actual.data(),from,65536*sizeof(uint64_t),cudaMemcpyDeviceToHost));
    for(int i=0;i<65536;++i)
      if(actual[i]!=full[i].ConvertToInt())throw std::runtime_error("NTT embedding differs from coefficient oracle");
    ++cases;
    check(cudaFree(from)); check(cudaFree(to));
  }
  return cases;
}
Ct encrypt(const Context& c, const KeyPair<DCRTPoly>& keys, const std::vector<double>& v) {
  select_cpu(c);
  auto p = c->MakeCKKSPackedPlaintext(v);
  auto out = c->Encrypt(keys.publicKey, p); sync_gpu(); return out;
}
double error(const Context& c, const KeyPair<DCRTPoly>& keys, const Ct& ct,
             const std::vector<double>& expected) {
  select_cpu(c); Plaintext p; auto copy = ct->Clone();
  c->Decrypt(keys.secretKey, copy, &p); p->SetLength(expected.size());
  const auto actual = p->GetRealPackedValue(); double worst = 0;
  for (std::size_t i=0; i<expected.size(); ++i) {
    if (!std::isfinite(actual.at(i))) throw std::runtime_error("nonfinite client output");
    worst = std::max(worst, std::abs(actual.at(i)-expected[i]));
  }
  return worst;
}

Ct refresh(const Context& c, Ct input) {
  while (input->GetNoiseScaleDeg()>1) c->RescaleInPlace(input);
  auto first = c->EvalBootstrap(input); sync_gpu();
  auto a = input->Clone(), b = first->Clone();
  while (b->GetNoiseScaleDeg()>1) c->RescaleInPlace(b);
  const auto target = std::max(a->GetLevel(), b->GetLevel());
  for (auto* x : {&a, &b})
    while ((*x)->GetLevel()<target) {
      c->EvalMultInPlace(*x, 1.0);
      while ((*x)->GetNoiseScaleDeg()>1) c->RescaleInPlace(*x);
    }
  auto residual = c->EvalSub(a,b); c->EvalMultInPlace(residual,4096.0);
  while (residual->GetNoiseScaleDeg()>1) c->RescaleInPlace(residual);
  auto second = c->EvalBootstrap(residual); sync_gpu();
  c->EvalMultInPlace(second, 1.0/4096.0);
  c->EvalMultInPlace(first, 1.0);
  auto out = c->EvalAdd(first,second); sync_gpu(); return out;
}


struct InternalSample { int input_level, output_level, scale_degree; bool mod; double error, seconds, scaling_factor; };
std::vector<InternalSample> internal_samples;
void internal_probe(const Context& c, const KeyPair<DCRTPoly>& keys) {
  std::vector<double> input(32768);
  const double delta=1.0/2048.0;
  const std::vector<double> edge={-1,1,0,-delta,delta,
      std::nextafter(-1.0,0.0),std::nextafter(1.0,0.0),
      std::nextafter(delta,0.0),std::nextafter(delta,1.0)};
  for(std::size_t i=0;i<input.size();++i) {
    if(i%3==0)input[i]=edge[(i/3)%edge.size()];
    else if(i%3==1)input[i]=-1.0+2.0*double(i)/double(input.size()-1);
    else { const int n=int((i/3)%1023)-511;
      const double offsets[]={-.0078125,-.0009765625,0,.0009765625,.0078125};
      input[i]=(n+offsets[(i/3)%5])/512.0;
    }
  }
  for(int level : {0,3}) for(bool mod : {false,true}) {
    auto out=encrypt(c,keys,input);out->SetLevel(level);sync_gpu();
    auto g=device(out);auto& cc=g->cc;
    if(!gpu::HasQualifiedEvenBootstrapSeed(cc))throw std::runtime_error("unqualified seed context");
    std::vector<double> expected;
    for(double x:input) {
      long double b1=0,b2=0;
      auto& coeff=cc.GetCoeffsChebyshev();
      for(int j=int(coeff.size())-1;j>=1;--j) {
        long double b=coeff[j]+2*static_cast<long double>(x)*b1-b2;b2=b1;b1=b;
      }
      long double y=coeff[0]/2.0L+static_cast<long double>(x)*b1-b2;
      if(mod)for(int j=1;j<=6;++j)
        y=2*y*y-1.0/std::pow(2.0*M_PI,std::pow(2.0,j-6));
      expected.push_back(double(y));
    }
    auto begin=Clock::now();
    if(mod)gpu::approxModReductionSparse(*g,1);
    else {
#if REFRESH_CANDIDATE
      gpu::EvalEvenBootstrapSeed(*g);
#else
      gpu::evalChebyshevSeries(*g,cc.GetCoeffsChebyshev(),-1,1);
#endif
    }
    while(out->GetNoiseScaleDeg()>1)c->RescaleInPlace(out);
    double duration=seconds(begin);double e=error(c,keys,out,expected);
    internal_samples.push_back({level,int(out->GetLevel()),int(out->GetNoiseScaleDeg()),mod,e,duration,device(out)->NoiseFactor});
    std::cerr<<"internal level="<<level<<" mod="<<mod<<" output_level="<<out->GetLevel()<<" error="<<std::setprecision(15)<<e<<'\n';
    if(e>(mod?1e-6:1e-8))throw std::runtime_error("internal polynomial error gate failed");
  }
}

struct Sample { int width, iteration; bool small; double ordinary, up, boot, down, err; int level; };

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::invalid_argument("usage: gpu_dual_ring OUTPUT qualification|abba|baab");
    const std::string mode = argv[2];
    if (mode!="qualification" && mode!="abba" && mode!="baab") throw std::invalid_argument("mode");
    const auto started = Clock::now();
    auto large = context(65536), small = context(32768);
    match_small_parameters(small, large);
    int exact_cases=exact_maps(large,small);
    select_cpu(large); auto lk = large->KeyGen(); large->EvalMultKeyGen(lk.secretKey);
    large->EvalRotateKeyGen(lk.secretKey, {1, -1, 24});
    large->EvalBootstrapSetup({4,4}, {0,0}, 32768, 0, true);
    large->EvalBootstrapKeyGen(lk.secretKey, 32768);
    large->LoadContext(lk.publicKey); sync_gpu();
    std::cerr << "large context ready\n";
    select_cpu(small); auto sk = small->KeyGen(); small->EvalMultKeyGen(sk.secretKey);
    small->EvalRotateKeyGen(sk.secretKey, {1,-1,24}); small->LoadContext(sk.publicKey); sync_gpu();
    std::cerr << "small context ready\n";
    auto lt = encrypt(large,lk,std::vector<double>(32768,.1));
    auto st = encrypt(small,sk,std::vector<double>(16384,.1));
    RingBridge bridge(large,small,lk,sk,lt,st);
    const double setup = seconds(started);
    internal_probe(large,lk);
    std::vector<Sample> samples; double worst = 0;
    // First qualify both transfer directions, all slots, and scale degrees.
    int cases = 0;
    for (int level : {18,26,34}) for (int width : {1,32,768,1536,16384}) {
      std::vector<double> sv(16384), lv(32768);
      for (int i=0;i<width;++i) sv[i]=.6*std::sin(i*.71+.1);
      for (int i=0;i<16384;++i) {
        lv[i]=sv[i];
        lv[i+16384]=i<width ? -.3*std::cos(i*.37+.2) : 0;
      }
      auto s = encrypt(small,sk,sv), l = encrypt(large,lk,lv);
      s->SetLevel(level); l->SetLevel(level);
      for (bool degree2 : {false,true}) {
        auto sc=s->Clone(),lc=l->Clone();
        auto se=sv,le=lv;
        if (degree2) {
          small->EvalMultInPlace(sc,.75); large->EvalMultInPlace(lc,.75);
          for(auto& v:se)v*=.75; for(auto& v:le)v*=.75;
        }
        std::vector<double> up_expected(32768),down_expected(16384);
        for(int i=0;i<32768;++i)up_expected[i]=se[i%16384];
        for(int i=0;i<16384;++i)down_expected[i]=(le[i]+le[i+16384])/2;
        auto up=bridge.up(sc), down=bridge.down(lc);
        auto e=std::max(error(large,lk,up,up_expected),error(small,sk,down,down_expected));
        worst=std::max(worst,e); ++cases;
        if(e>1e-6) throw std::runtime_error("transfer numerical failure: "+std::to_string(e));
      }
      std::cerr<<"transfer level="<<level<<" width="<<width<<" worst="<<worst<<'\n';
    }
    const std::vector<int> widths = mode=="qualification" ? std::vector<int>{1,32,768,1536} : std::vector<int>{1536};
    for (int width : widths) {
      std::vector<double> sv(16384),lv(32768);
      for(int i=0;i<width;++i)sv[i]=.6*std::sin(i*.71+.1);
      for(int i=0;i<32768;++i)lv[i]=sv[i%16384];
      auto s=encrypt(small,sk,sv),l=encrypt(large,lk,lv);
      s->SetLevel(18); l->SetLevel(18);
      const std::string order=mode=="qualification"?"ab":mode;
      for(int iteration=-1;iteration<int(order.size());++iteration) {
        const bool sm=iteration<0?true:order[iteration]=='b';
        auto c=sm?small:large; auto input=sm?s:l;
        auto expected=sm?sv:lv;
        sync_gpu(); auto begin=Clock::now();
        // Eight levels of bounded, nonconstant encrypted polynomial work.
        auto out=input->Clone();
        for(int k=0;k<8;++k) {
          out=c->EvalMult(out,out); c->EvalAddInPlace(out,.125);
          for(auto& v:expected)v=v*v+.125;
        }
        while(out->GetNoiseScaleDeg()>1)c->RescaleInPlace(out);
        double ordinary=seconds(begin),up=0,down=0;
        begin=Clock::now();
        auto full=sm?bridge.up(out):out;
        if(sm)up=seconds(begin);
        // Fixed model input budget; consumes scale through arithmetic, no reset.
        full->SetLevel(34); sync_gpu(); begin=Clock::now();
        full=refresh(large,full); double boot=seconds(begin);
        begin=Clock::now(); out=sm?bridge.down(full):full;
        if(sm)down=seconds(begin);
        begin=Clock::now(); out=c->EvalMult(out,out); ordinary+=seconds(begin);
        for(auto& v:expected)v*=v;
        auto e=error(c,sm?sk:lk,out,expected); worst=std::max(worst,e);
        samples.push_back({width,iteration,sm,ordinary,up,boot,down,e,int(out->GetLevel())});
        std::cerr<<"sample width="<<width<<" small="<<sm<<" ordinary="<<ordinary<<" up="<<up<<" refresh="<<boot<<" down="<<down<<" error="<<e<<'\n';
      }
    }
    std::ofstream report(argv[1]); report<<std::setprecision(15)
      <<"{\"passed\":"<<(worst<=1e-6?"true":"false")<<",\"mode\":\""<<mode
      <<"\",\"tolerance\":1e-6,\"setup_seconds\":"<<setup<<",\"transfer_cases\":"<<cases
      <<",\"exact_ntt_maps\":"<<exact_cases
      <<",\"max_abs_error\":"<<worst<<",\"evaluator_decryptions\":0,\"depth\":44,\"scale_bits\":59,\"bootstrap_passes\":2,\"s2c_first\":true,\"samples\":[";
    for(std::size_t i=0;i<samples.size();++i) {
      auto a=samples[i]; if(i)report<<',';
      report<<"{\"width\":"<<a.width<<",\"iteration\":"<<a.iteration<<",\"small\":"<<(a.small?"true":"false")
        <<",\"ordinary_seconds\":"<<a.ordinary<<",\"up_seconds\":"<<a.up<<",\"refresh_seconds\":"<<a.boot
        <<",\"down_seconds\":"<<a.down<<",\"error\":"<<a.err<<",\"output_level\":"<<a.level<<'}';
    }
    report<<"],\"candidate\":"<<(REFRESH_CANDIDATE?"true":"false")<<",\"internal_samples\":[";
    for(std::size_t i=0;i<internal_samples.size();++i){auto s=internal_samples[i];if(i)report<<',';
      report<<"{\"input_level\":"<<s.input_level<<",\"output_level\":"<<s.output_level
        <<",\"scale_degree\":"<<s.scale_degree<<",\"evalmod\":"<<(s.mod?"true":"false")
        <<",\"error\":"<<s.error<<",\"seconds\":"<<s.seconds<<",\"scaling_factor\":"<<s.scaling_factor<<'}';}
    report<<"]}\n"; return worst<=1e-6?0:2;
  } catch(const std::exception& e) { std::cerr<<"FAILED: "<<e.what()<<'\n'; return 2; }
}
