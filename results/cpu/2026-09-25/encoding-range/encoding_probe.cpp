// Actual pinned OpenFHE MakeCKKSPackedPlaintext comparison, CPU only.
#include "openfhe.h"
#include <omp.h>
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lbcrypto;
using Values = std::vector<std::complex<double>>;
using EncodingPolyParams = DCRTPoly::Params;

Values fixture(int slots, int pattern, bool complex) {
  Values values(slots);
  if (pattern == 0) return values;
  if (pattern == 2) {
    values[0] = .25; values[31] = -.5; values.back() = .75;
    return values;
  }
  const double scale = pattern == 3 ? 0x1p20 : pattern == 4 ? 0x1p-1000 : .0625;
  for (int i = 0; i < slots; ++i)
    values[i] = {std::sin(i * .017 + .3) * scale,
                 complex ? std::cos(i * .031 + .7) * scale / 2 : 0};
  return values;
}

std::shared_ptr<EncodingPolyParams> coefficient_params(const CryptoContext<DCRTPoly>& cc, int level) {
  std::vector<NativeInteger> moduli, roots;
  const auto& full = cc->GetElementParams()->GetParams();
  for (std::size_t i = 0; i < full.size() - level; ++i) {
    moduli.push_back(full[i]->GetModulus()); roots.emplace_back(1);
  }
  return std::make_shared<EncodingPolyParams>(2 * cc->GetRingDimension(), moduli, roots);
}

class Golden {
 public:
  Golden(const std::string& filename, bool write) : write_(write) {
    if (write_) output_.open(filename, std::ios::binary | std::ios::trunc);
    else input_.open(filename, std::ios::binary);
    if (write_ ? !output_ : !input_) throw std::runtime_error("cannot open golden coefficients");
  }
  void words(const std::vector<uint64_t>& values) {
    const auto bytes = values.size() * sizeof(uint64_t);
    if (write_) {
      output_.write(reinterpret_cast<const char*>(values.data()), bytes);
      if (!output_) throw std::runtime_error("golden write failed");
    } else {
      std::vector<uint64_t> expected(values.size());
      input_.read(reinterpret_cast<char*>(expected.data()), bytes);
      if (!input_ || values != expected) throw std::runtime_error("coefficient or metadata mismatch");
    }
    words_ += values.size();
  }
  void finish() {
    if (!write_ && input_.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("trailing golden coefficients");
    if (write_) { output_.flush(); if (!output_) throw std::runtime_error("golden flush failed"); }
  }
  uint64_t count() const { return words_; }
 private:
  bool write_;
  std::ifstream input_;
  std::ofstream output_;
  uint64_t words_ = 0;
};

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::invalid_argument("usage: encoding_probe write|compare|bench GOLDEN OUTPUT_JSON");
    const std::string mode = argv[1];
    if (mode != "write" && mode != "compare" && mode != "bench") throw std::invalid_argument("bad mode");
    std::unique_ptr<Golden> golden;
    if (mode != "bench") golden = std::make_unique<Golden>(argv[2], mode == "write");
    std::ofstream out(argv[3]);
    if (!out) throw std::runtime_error("cannot open result");
    out << std::setprecision(17) << "{\"mode\":\"" << mode << "\",\"benchmarks\":[";
    int cases = 0, expected_errors = 0, encoded = 0;
    uint64_t coefficient_words = 0;
    bool first_benchmark = true;
    for (bool complex : {false, true}) {
      CCParams<CryptoContextCKKSRNS> p;
      p.SetSecurityLevel(HEStd_NotSet); p.SetRingDim(65536); p.SetBatchSize(32768);
      p.SetSecretKeyDist(complex ? SPARSE_TERNARY : UNIFORM_TERNARY);
      p.SetCKKSDataType(complex ? COMPLEX : REAL);
      p.SetMultiplicativeDepth(44); p.SetScalingModSize(59); p.SetFirstModSize(60);
      p.SetScalingTechnique(FLEXIBLEAUTO); p.SetKeySwitchTechnique(HYBRID); p.SetNumLargeDigits(3);
      auto cc = GenCryptoContext(p);
      cc->Enable(PKE); cc->Enable(LEVELEDSHE);
      // Match the identity-root encoder's existing serial suppression policy.
      const int previous = omp_get_max_active_levels();
      omp_set_max_active_levels(0);
      if (mode == "bench") {
        const auto params = coefficient_params(cc, 21);
        for (int pattern : {1, 2}) {
          const auto values = fixture(32768, pattern, complex);
          auto warm = cc->MakeCKKSPackedPlaintext(values, 1, 21, params, 32768);
          warm.reset();
          double seconds = 0;
          for (int i = 0; i < 24; ++i) {
            const auto start = std::chrono::steady_clock::now();
            auto plain = cc->MakeCKKSPackedPlaintext(values, 1, 21, params, 32768);
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (plain->GetElement<DCRTPoly>().GetNumOfElements() != params->GetParams().size())
              throw std::runtime_error("wrong benchmark limb count");
          }
          if (!first_benchmark) out << ',';
          first_benchmark = false;
          out << "{\"complex\":" << (complex ? "true" : "false") << ",\"pattern\":" << pattern
              << ",\"seconds\":" << seconds / 24 << ",\"repetitions\":24}";
        }
      } else {
        for (int slots : {512, 32768}) for (int level : {0, 21, 34}) for (int degree : {1, 2})
          for (int pattern = 0; pattern < 5; ++pattern) {
            golden->words({static_cast<uint64_t>(cases), static_cast<uint64_t>(complex),
                           static_cast<uint64_t>(slots), static_cast<uint64_t>(level),
                           static_cast<uint64_t>(degree), static_cast<uint64_t>(pattern)});
            const auto values = fixture(slots, pattern, complex);
            const auto params = coefficient_params(cc, level);
            Plaintext plain;
            try { plain = cc->MakeCKKSPackedPlaintext(values, degree, level, params, slots); }
            catch (const std::exception& error) {
              if (pattern != 4 || std::string(error.what()).find("Scaling factor too small") == std::string::npos)
                throw;
              golden->words({1}); ++expected_errors; ++cases;
              continue;
            }
            if (pattern == 4) throw std::runtime_error("expected scaling failure was not raised");
            const auto& poly = plain->GetElement<DCRTPoly>();
            golden->words({0, plain->GetSlots(), plain->GetLevel(), plain->GetNoiseScaleDeg(),
                           poly.GetRingDimension(), static_cast<uint64_t>(poly.GetFormat()),
                           std::bit_cast<uint64_t>(plain->GetScalingFactor()), poly.GetNumOfElements()});
            for (const auto& limb : poly.GetAllElements()) {
              const auto& data = limb.GetValues();
              golden->words({limb.GetModulus().ConvertToInt(), data.GetLength()});
              std::vector<uint64_t> words(data.GetLength());
              for (std::size_t i = 0; i < words.size(); ++i) words[i] = data[i].ConvertToInt();
              golden->words(words); coefficient_words += words.size();
            }
            ++encoded; ++cases;
          }
      }
      omp_set_max_active_levels(previous);
    }
    if (golden) golden->finish();
    if (mode != "bench" && (cases != 120 || encoded != 96 || expected_errors != 24))
      throw std::runtime_error("wrong correctness case count");
    out << "],\"passed\":true,\"cases\":" << cases << ",\"encoded_cases\":" << encoded
        << ",\"expected_scaling_failures\":" << expected_errors << ",\"coefficient_words\":" << coefficient_words
        << ",\"all_words\":" << (golden ? golden->count() : 0)
        << ",\"scope\":\"Local CPU actual OpenFHE coefficient-root Encode only; no GPU upload, encrypted inference, target FIDESlib integration patches, or model timing.\"}\n";
    std::cout << "passed cases=" << cases << " coefficients=" << coefficient_words << std::endl;
  } catch (const std::exception& error) { std::cerr << error.what() << std::endl; return 2; }
}
