// Compare every output word with a separately linked, unmodified GPU kernel.
// Synthetic public inputs cover both halves and every forward-NTT fusion mode.
#include <fideslib.hpp>
#include <CKKS/Context.cuh>
#include <NTT.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace fideslib;
using namespace FIDESlib;
static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
  uint64_t* data = nullptr;
  explicit Buffer(std::size_t n) { check(cudaMalloc(&data, n * sizeof(uint64_t))); }
  ~Buffer() { cudaFree(data); }
  Buffer(const Buffer&) = delete;
  void set(const std::vector<uint64_t>& v) {
    check(cudaMemcpy(data, v.data(), v.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
  }
};
struct PointerTable {
  void** data = nullptr;
  explicit PointerTable(const std::vector<std::unique_ptr<Buffer>>& buffers) {
    std::vector<void*> pointers;
    for (const auto& p : buffers) pointers.push_back(p->data);
    check(cudaMalloc(&data, pointers.size() * sizeof(void*)));
    check(cudaMemcpy(data, pointers.data(), pointers.size() * sizeof(void*), cudaMemcpyHostToDevice));
  }
  ~PointerTable() { cudaFree(data); }
  PointerTable(const PointerTable&) = delete;
};
static uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
static std::vector<uint64_t> values(int n, uint64_t q, int pattern, int salt) {
  std::vector<uint64_t> v(n);
  for (int i = 0; i < n; ++i) {
    if (pattern == 0) v[i] = 0;
    else if (pattern == 1) v[i] = q - 1;
    else if (pattern == 2) v[i] = i % 2 ? q - 1 : 0;
    else v[i] = mix(static_cast<uint64_t>(salt) * n + i) % q;
  }
  return v;
}
template <bool second, NTT_MODE mode>
static void launch(const Global::Globals* globals, void** input, void** output,
                   void** pt, void** output2, void** kskb, int source_prime,
                   int log_n, int limbs) {
  const unsigned n = 1u << log_n;
  const unsigned block = 1u << (second ? log_n / 2 - 1 : (log_n + 1) / 2 - 1);
  NTT_<second, ALGO_SHOUP, mode><<<dim3(n / (block * 8), limbs), block, 8 * block * 10>>>(
      globals, input, PARTITION(0, 0), output, pt, source_prime, output2, kskb);
  check(cudaGetLastError());
}
template <bool second>
static void dispatch(int mode, const Global::Globals* g, void** in, void** out,
                     void** pt, void** out2, void** k, int source, int log_n, int limbs) {
#define RUN(M) case M: launch<second, M>(g, in, out, pt, out2, k, source, log_n, limbs); break
  switch (mode) {
    RUN(NTT_NONE); RUN(NTT_RESCALE); RUN(NTT_MULTPT); RUN(NTT_MODDOWN);
    RUN(NTT_KSK_DOT); RUN(NTT_KSK_DOT_ACC);
    default: throw std::runtime_error("unknown mode");
  }
#undef RUN
}
int main(int argc, char** argv) {
  try {
    if (argc < 4) throw std::invalid_argument("usage: ntt_warp_tail REPORT GOLDEN RING [--record] [--bench-only]");
    const int ring = std::stoi(argv[3]);
    if (ring != 1024 && ring != 2048 && ring != 4096 && ring != 32768 && ring != 65536 && ring != 131072)
      throw std::invalid_argument("unsupported ring");
    bool record = false, bench_only = false;
    bool special_primes = false;
    int digits = 3;
    for (int i = 4; i < argc; ++i) {
      const std::string flag = argv[i];
      if (flag == "--record") record = true;
      else if (flag == "--bench-only") bench_only = true;
      else if (flag == "--special-primes") special_primes = true;
      else if (flag == "--digits" && i + 1 < argc) digits = std::stoi(argv[++i]);
      else throw std::invalid_argument("unknown option");
    }
    if (digits < 1 || digits > 4 || (special_primes && bench_only))
      throw std::invalid_argument("invalid digits or incompatible probe modes");
    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecurityLevel(HEStd_NotSet); p.SetSecretKeyDist(UNIFORM_TERNARY);
    p.SetCKKSDataType(REAL); p.SetRingDim(ring); p.SetBatchSize(ring / 2);
    p.SetMultiplicativeDepth(44); p.SetScalingModSize(59); p.SetFirstModSize(60);
    p.SetScalingTechnique(FLEXIBLEAUTO); p.SetKeySwitchTechnique(HYBRID);
    p.SetNumLargeDigits(digits); p.SetDevices({0});
    auto cc = GenCryptoContext(p);
    for (auto feature : {fideslib::PKE, fideslib::KEYSWITCH, fideslib::LEVELEDSHE}) cc->Enable(feature);
    auto keys = cc->KeyGen();
    // Include an evaluation key in the hybrid-basis diagnostic context.
    if (special_primes) cc->EvalMultKeyGen(keys.secretKey);
    cc->LoadContext(keys.publicKey);
    auto& gpu = std::any_cast<FIDESlib::CKKS::Context&>(cc->gpu);
    FIDESlib::CKKS::SetCurrentContext(gpu);
    // The final ordinary modulus is a distinct source for the 44 targets.
    const int source_prime = 44;
    const auto* globals = gpu->precom.globals->globals[0];
    check(cudaDeviceSynchronize());

    std::fstream golden(argv[2], std::ios::binary | (record ? std::ios::out | std::ios::trunc : std::ios::in));
    if (!golden) throw std::runtime_error("cannot open golden stream");
    uint64_t words = 0;
    auto compare = [&](const uint64_t* data, std::size_t count) {
      const auto bytes = count * sizeof(uint64_t);
      if (record) golden.write(reinterpret_cast<const char*>(data), bytes);
      else {
        std::vector<uint64_t> expected(count);
        golden.read(reinterpret_cast<char*>(expected.data()), bytes);
        if (!golden || !std::equal(expected.begin(), expected.end(), data))
          throw std::runtime_error("golden word mismatch after " + std::to_string(words));
      }
      if (!golden) throw std::runtime_error("golden stream IO failed");
      words += count;
    };
    const uint64_t header[] = {0x4648454e54543032ull, static_cast<uint64_t>(ring),
                              static_cast<uint64_t>(bench_only), static_cast<uint64_t>(digits),
                              static_cast<uint64_t>(special_primes)};
    compare(header, 5);
    std::vector<uint64_t> primes;
    for (int i = 0; i <= source_prime; ++i) primes.push_back(gpu->prime[i].p);
    compare(primes.data(), primes.size());
    std::vector<uint64_t> all_primes = primes;
    std::vector<std::pair<int, uint64_t>> scalar_primes{{source_prime, primes[source_prime]}};
    if (special_primes) {
      if (gpu->specialPrime.empty() || gpu->specialMeta.at(0).size() != gpu->specialPrime.size())
        throw std::runtime_error("special-prime probe has no complete P metadata");
      for (std::size_t i = 0; i < gpu->specialPrime.size(); ++i) {
        const uint64_t q = gpu->specialPrime[i].p;
        if (q <= 1) throw std::runtime_error("invalid P prime");
        all_primes.push_back(q);
        scalar_primes.emplace_back(gpu->specialMeta.at(0).at(i).id, q);
      }
      compare(all_primes.data() + primes.size(), all_primes.size() - primes.size());
    }
    words = 0;

    std::vector<std::unique_ptr<Buffer>> in, out, pt, out2, k;
    for (int i = 0; i < 44; ++i) {
      in.push_back(std::make_unique<Buffer>(ring)); out.push_back(std::make_unique<Buffer>(ring));
      pt.push_back(std::make_unique<Buffer>(ring)); out2.push_back(std::make_unique<Buffer>(ring));
      k.push_back(std::make_unique<Buffer>(ring));
    }
    PointerTable in_table(in), out_table(out), pt_table(pt), out2_table(out2), k_table(k);
    cudaEvent_t begin, end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    struct Timing { int mode, second, iteration; float ms; };
    std::vector<Timing> samples;
    int cases = 0;
    for (int limbs : {1, 4, 44}) for (int mode = 0; mode < 6; ++mode)
      for (bool second : {false, true}) for (int pattern = 0; pattern < 4; ++pattern) {
        if (bench_only && (limbs != 44 || mode > 1 || pattern != 3)) continue;
        std::vector<std::vector<uint64_t>> initial_out, initial_out2;
        for (int i = 0; i < limbs; ++i) {
          auto q = primes[i];
          const auto in_q = !second && (mode == NTT_RESCALE || mode == NTT_MULTPT) && i == 0
                              ? primes[source_prime] : q;
          in[i]->set(values(ring, in_q, pattern, 10 + i));
          pt[i]->set(values(ring, q, 3, 100 + i)); k[i]->set(values(ring, q, 3, 200 + i));
          initial_out.push_back(values(ring, q, 3, 300 + i));
          initial_out2.push_back(values(ring, q, 3, 400 + i));
          out[i]->set(initial_out.back()); out2[i]->set(initial_out2.back());
        }
        auto run = [&] {
          if (second) dispatch<true>(mode, globals, in_table.data, out_table.data,
              pt_table.data, out2_table.data, k_table.data, source_prime, gpu->logN, limbs);
          else dispatch<false>(mode, globals, in_table.data, out_table.data,
              pt_table.data, out2_table.data, k_table.data, source_prime, gpu->logN, limbs);
        };
        run(); check(cudaDeviceSynchronize());
        std::vector<uint64_t> result(ring);
        for (int i = 0; i < limbs; ++i) {
          check(cudaMemcpy(result.data(), out[i]->data, ring * sizeof(uint64_t), cudaMemcpyDeviceToHost));
          compare(result.data(), result.size());
          check(cudaMemcpy(result.data(), out2[i]->data, ring * sizeof(uint64_t), cudaMemcpyDeviceToHost));
          if (second && mode >= NTT_KSK_DOT) compare(result.data(), result.size());
          else if (result != initial_out2[i]) throw std::runtime_error("untouched output changed");
        }
        ++cases;
        if (limbs == 44 && mode <= 1 && pattern == 3) {
          for (int iteration = -1; iteration < 16; ++iteration) {
            // Restore every overwritten/read-modify-write operand before timing.
            for (int i = 0; i < limbs; ++i) {
              out[i]->set(initial_out[i]); out2[i]->set(initial_out2[i]);
            }
            check(cudaDeviceSynchronize()); check(cudaEventRecord(begin));
            run(); check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
            float ms = 0; check(cudaEventElapsedTime(&ms, begin, end));
            samples.push_back({mode, second, iteration, ms});
          }
        }
      }
    int special_cases = 0;
    if (special_primes) {
      // Vector-fused cases above cover the Q targets. Exercise the remaining
      // Q source and every P prime through the scalar ordinary-NTT entry too.
      for (const auto& [prime, q] : scalar_primes) {
        if (q >= (uint64_t(1) << 62)) throw std::runtime_error("probe prime exceeds lazy bound");
        if (q < (uint64_t(1) << 32)) throw std::runtime_error("probe expects 64-bit primes");
        for (bool second : {false, true}) for (int pattern = 0; pattern < 4; ++pattern) {
          in[0]->set(values(ring, q, pattern, 700 + prime));
          const unsigned block = 1u << (second ? gpu->logN / 2 - 1 : (gpu->logN + 1) / 2 - 1);
          const dim3 grid(ring / (block * 8));
          if (second)
            NTT_<uint64_t, true, ALGO_SHOUP, NTT_NONE><<<grid, block, 8 * block * 10>>>(
                globals, in[0]->data, prime, out[0]->data);
          else
            NTT_<uint64_t, false, ALGO_SHOUP, NTT_NONE><<<grid, block, 8 * block * 10>>>(
                globals, in[0]->data, prime, out[0]->data);
          check(cudaGetLastError()); check(cudaDeviceSynchronize());
          std::vector<uint64_t> result(ring);
          check(cudaMemcpy(result.data(), out[0]->data, ring * sizeof(uint64_t), cudaMemcpyDeviceToHost));
          compare(result.data(), result.size());
          ++special_cases;
        }
      }
    }
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
    if (!record && golden.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("extra golden data");
    if (cases != (bench_only ? 4 : 144)) throw std::runtime_error("case coverage changed");
    std::ofstream report(argv[1]);
    if (!report) throw std::runtime_error("cannot write report");
    report << std::setprecision(12) << "{\"passed\":true,\"ring\":" << ring
           << ",\"record\":" << (record ? "true" : "false") << ",\"cases\":" << cases
           << ",\"output_words\":" << words << ",\"bench_only\":" << (bench_only ? "true" : "false")
           << ",\"digits\":" << digits << ",\"special_prime_cases\":" << special_cases
           << ",\"all_primes\":[";
    for (std::size_t i = 0; i < all_primes.size(); ++i) { if (i) report << ','; report << all_primes[i]; }
    report << ']'
           << ",\"samples\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
      const auto& s = samples[i]; if (i) report << ',';
      report << "{\"mode\":" << s.mode << ",\"second\":" << s.second
             << ",\"iteration\":" << s.iteration << ",\"kernel_ms\":" << s.ms << '}';
    }
    report << "]}\n";
    std::cout << "passed cases=" << cases << " output_words=" << words << '\n';
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
