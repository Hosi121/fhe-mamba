// Compare actual baseline/candidate OpenFHE FFT output bits in separate processes.
#include <math/dftransform.h>
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Values = std::vector<std::complex<double>>;
using FFT = lbcrypto::DiscreteFourierTransform;

static Values fixture(unsigned n, unsigned pattern) {
    Values v(n);
    for (unsigned i = 0; i < n; ++i) {
        switch (pattern) {
            case 0: v[i] = {i % 2 ? -0.0 : 0.0, -0.0}; break;
            case 1: v[i] = {std::sin(i * .71), std::cos(i * .31)}; break;
            case 2: v[i] = {i == n / 2 ? .5 : 0, 0}; break;
            case 3: v[i] = {i % 2 ? -.25 : .25, .125}; break;
            case 4: v[i] = {std::ldexp(double(i % 7) - 3, -1068), -0.0}; break;
            case 5: v[i] = {std::ldexp(double(i % 7) - 3, 900), 0}; break;
            case 6: v[i] = {i == 0 ? std::numeric_limits<double>::infinity() : .25, 0}; break;
            case 7: v[i] = {i == n / 2 ? std::bit_cast<double>(UINT64_C(0x7ff8000000000123)) : -.25, 0}; break;
        }
    }
    return v;
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::invalid_argument("fft_probe write|compare|bench GOLDEN JSON");
        const std::string mode = argv[1];
        if (mode != "write" && mode != "compare" && mode != "bench") throw std::invalid_argument("mode");
        std::ifstream input;
        std::ofstream output;
        if (mode == "write") output.open(argv[2], std::ios::binary);
        if (mode == "compare") input.open(argv[2], std::ios::binary);
        if ((mode == "write" && !output) || (mode == "compare" && !input)) throw std::runtime_error("golden open");
        unsigned cases = 0;
        uint64_t words = 0;
        std::ofstream report(argv[3]);
        if (!report) throw std::runtime_error("report open");
        report << std::setprecision(17) << "{\"samples\":[";
        bool first = true;
        // A non-power-of-two initialization cannot have a cached plan; n=2
        // remains a valid original transform and must use the unchanged path.
        if (mode != "bench") {
            FFT::Initialize(32, 3);
            auto v = fixture(2, 1);
            FFT::FFTSpecialInv(v, 32);
            for (auto z : v) for (double component : {z.real(), z.imag()}) {
                const auto bits = std::bit_cast<uint64_t>(component);
                if (mode == "write") output.write(reinterpret_cast<const char*>(&bits), sizeof bits);
                else {
                    uint64_t expected;
                    input.read(reinterpret_cast<char*>(&expected), sizeof expected);
                    if (!input || bits != expected) throw std::runtime_error("fallback bit mismatch");
                }
                ++words;
            }
            ++cases;
        }
        for (unsigned order : {65536u, 131072u}) {
            FFT::Initialize(order, order / 4);
            for (unsigned n = 1; n <= order / 4; n <<= 1) {
                for (unsigned pattern = 0; pattern < 8; ++pattern) {
                    auto v = fixture(n, pattern);
                    FFT::FFTSpecialInv(v, order);
                    if (mode != "bench") {
                        for (auto value : v) for (double component : {value.real(), value.imag()}) {
                            const auto bits = std::bit_cast<uint64_t>(component);
                            if (mode == "write") output.write(reinterpret_cast<const char*>(&bits), sizeof bits);
                            else {
                                uint64_t expected;
                                input.read(reinterpret_cast<char*>(&expected), sizeof expected);
                                if (!input || bits != expected) throw std::runtime_error("FFT bit mismatch: n=" + std::to_string(n) + " pattern=" + std::to_string(pattern));
                            }
                            ++words;
                        }
                        ++cases;
                    }
                }
                if (mode != "bench" || n < 512) continue;
                const auto original = fixture(n, 1);
                double seconds = 0;
                for (int rep = 0; rep < 128; ++rep) {
                    auto v = original;
                    const auto start = std::chrono::steady_clock::now();
                    FFT::FFTSpecialInv(v, order);
                    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                }
                if (!first) report << ',';
                first = false;
                report << "{\"order\":" << order << ",\"slots\":" << n << ",\"seconds\":" << seconds / 128 << ",\"repetitions\":128}";
            }
        }
        if (mode == "write") { output.flush(); if (!output) throw std::runtime_error("golden write"); }
        if (mode == "compare" && input.peek() != std::char_traits<char>::eof()) throw std::runtime_error("golden trailing data");
        report << "],\"passed\":true,\"cases\":" << cases << ",\"words\":" << words << "}\n";
        std::cout << "passed cases=" << cases << " words=" << words << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
