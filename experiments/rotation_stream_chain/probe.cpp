// Compare the compiled rotation against the pinned, barrier-separated sequence.
// Fresh clones and pending producers exercise first capture, replay and reuse.
#include "fideslib_dual_ring.hpp"
#include <complex>
using namespace fhemamba::dual_ring;

static void reference_rotate(gpu::Ciphertext& x, int shift, bool moddown) {
  gpu::SetCurrentContext(x.cc_);
  const int index = x.normalyzeIndex(shift);
  auto& in0 = x.cc.getKeySwitchAux2();
  auto& in1 = x.cc.getKeySwitchAux();
  in1.copy(x.c1);
  in0.copy(x.c0);
  sync_gpu();
  in1.modup();
  sync_gpu();
  auto& key = x.cc.GetRotationKey(index, x.keyID, x.slots);
  std::vector<int> indices{index};
  std::vector<gpu::RNSPoly*> c0{&x.c0}, c1{&x.c1};
  std::vector<gpu::RNSPoly*> a{&key.a}, b{&key.b};
  in1.hoistedRotationFused(indices, c0, c1, a, b, in0, in1);
  sync_gpu();
  if (moddown) {
    x.modDown(false);
    sync_gpu();
  }
}

static gpu::RawCipherText snapshot(const Ct& value) {
  gpu::RawCipherText raw{};
  device(value)->store(raw);
  sync_gpu();
  return raw;
}

static void require_equal(const gpu::RawCipherText& a, const gpu::RawCipherText& b) {
  if (a.sub_0 != b.sub_0 || a.sub_1 != b.sub_1 || a.numRes != b.numRes ||
      a.N != b.N || a.Noise != b.Noise || a.NoiseLevel != b.NoiseLevel ||
      a.slots != b.slots || a.keyid != b.keyid)
    throw std::runtime_error("RNS coefficients or metadata differ from barrier oracle");
}

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::invalid_argument("usage: rotation_probe OUTPUT N mamba2|mamba3");
    const int n = std::stoi(argv[2]);
    const std::string model = argv[3];
    if ((n != 32768 && n != 65536) || (model != "mamba2" && model != "mamba3"))
      throw std::invalid_argument("unsupported configuration");
    const bool mamba2 = model == "mamba2";
    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecurityLevel(HEStd_NotSet);
    p.SetSecretKeyDist(mamba2 ? SPARSE_TERNARY : UNIFORM_TERNARY);
    p.SetCKKSDataType(mamba2 ? COMPLEX : REAL);
    p.SetRingDim(n); p.SetBatchSize(n/2); p.SetMultiplicativeDepth(44);
    p.SetScalingModSize(59); p.SetFirstModSize(60);
    p.SetScalingTechnique(FLEXIBLEAUTO); p.SetKeySwitchTechnique(HYBRID);
    p.SetNumLargeDigits(3); p.SetDevices({0});
    p.SetPlaintextAutoload(false); p.SetCiphertextAutoload(true);
    auto c = GenCryptoContext(p);
    for (auto f : {PKE, KEYSWITCH, LEVELEDSHE}) c->Enable(f);
    auto keys = c->KeyGen(); c->EvalMultKeyGen(keys.secretKey);
    const std::vector<int32_t> shifts{1,-1,8,-8,1024,-1024};
    c->EvalRotateKeyGen(keys.secretKey, shifts);
    c->LoadContext(keys.publicKey); sync_gpu();
    std::vector<std::complex<double>> values(n/2);
    for (int i=0; i<n/2; ++i)
      values[i] = {std::sin(.37*i+.2)/8, mamba2 ? std::cos(.71*i)/16 : 0.0};
    auto plaintext = c->MakeCKKSPackedPlaintext(values);
    auto original = c->Encrypt(keys.publicKey, plaintext);
    const auto original_raw = snapshot(original);
    int exact_cases=0, extended_cases=0, chain_cases=0;
    double max_error=0;
    auto prepare = [&](int level, bool scaled) {
      auto out = original->Clone();
      out->SetLevel(level);
      if (scaled) c->EvalMultInPlace(out, .75);
      // Deliberately no host synchronization before rotation.
      return out;
    };
    for (int repeat=0; repeat<2; ++repeat)
      for (int level : {0,18,34,39,42})
        for (bool scaled : {false,true})
          for (int shift : shifts)
            for (bool moddown : {true,false}) {
              Ct actual;
              const bool candidate_first = repeat == 0;
              if (candidate_first) {
                actual=prepare(level,scaled);
                device(actual)->rotate(shift,moddown);
                if (!moddown) device(actual)->modDown(false);
              }
              auto reference=prepare(level,scaled);
              reference_rotate(*device(reference),shift,moddown);
              if (!moddown) device(reference)->modDown(false);
              const auto expected=snapshot(reference);
              // Prepare candidate only after the oracle has finished, so the
              // oracle's barriers cannot complete the candidate's producers.
              if (!candidate_first) {
                actual=prepare(level,scaled);
                device(actual)->rotate(shift,moddown);
                if (!moddown) device(actual)->modDown(false);
              }
              require_equal(expected,snapshot(actual));
              if (moddown) ++exact_cases; else ++extended_cases;
              if (repeat==0 && level==18 && scaled && moddown) {
                Plaintext decoded;
                auto copy=actual->Clone(); c->Decrypt(keys.secretKey,copy,&decoded);
                decoded->SetLength(n/2);
                const auto observed=decoded->GetCKKSPackedValue();
                for (int i=0;i<n/2;++i) {
                  const auto e=std::abs(observed[i]-.75*values[(i+shift+n/2)%(n/2)]);
                  if (!std::isfinite(e)) throw std::runtime_error("nonfinite result");
                  max_error=std::max(max_error,e);
                }
              }
            }
    for (int level : {0,18,34}) {
      auto ref=prepare(level,true);
      for (int i=0;i<24;++i) reference_rotate(*device(ref),shifts[i%shifts.size()],true);
      // A following product checks output completion, metadata and scratch reuse.
      c->EvalMultInPlace(ref,.25);
      auto expected=snapshot(ref);
      auto actual=prepare(level,true);
      for (int i=0;i<24;++i) device(actual)->rotate(shifts[i%shifts.size()],true);
      c->EvalMultInPlace(actual,.25);
      require_equal(expected,snapshot(actual));
      ++chain_cases;
    }
    require_equal(original_raw,snapshot(original));
    if (exact_cases!=120 || extended_cases!=120 || chain_cases!=3 || max_error>1e-6)
      throw std::runtime_error("coverage or numerical gate failed");
    std::ofstream out(argv[1]);
    out<<std::setprecision(15)<<"{\"passed\":true,\"ring_dimension\":"<<n
       <<",\"model\":\""<<model<<"\",\"exact_rns_cases\":"<<exact_cases
       <<",\"extended_fallback_cases\":"<<extended_cases<<",\"chain_cases\":"<<chain_cases
       <<",\"max_abs_error\":"<<max_error
       <<",\"tolerance\":1e-6,\"live_input_unchanged\":true,\"security\":\"not-set\"}\n";
    std::cout<<"passed "<<model<<" N="<<n<<" exact="<<exact_cases<<" fallback="<<extended_cases<<'\n';
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
}
