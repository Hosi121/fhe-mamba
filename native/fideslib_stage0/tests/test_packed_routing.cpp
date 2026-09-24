#include "packed_routing.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

// Compare with direct indexing, including dirty gather padding and every radix.
static void check(const std::vector<double>& indices, int slots) {
  for (bool scatter : {false, true}) for (int bits : {1, 2, 3, 4}) {
    std::vector<double> input(slots), expected(slots);
    for (int i = 0; i < slots; ++i) input[i] = std::sin(i + 0.5);
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
      if (scatter) expected[indices[i]] = input[i];
      else expected[i] = input[indices[i]];
    }
    auto out = input;
    for (const auto& stage : fhemamba::monotone_routing(indices, scatter, bits)) {
      std::vector<double> next(slots);
      for (const auto& [offset, positions] : stage) for (int position : positions) {
        int destination = position - offset;
        if (destination < 0 || destination >= slots) throw std::runtime_error("routing overflow");
        next[destination] += out[position];
      }
      std::vector<double> transformed(slots);
      for (const auto& [offset, mask] : fhemamba::packed_routing_masks(stage, slots))
        for (int i = 0; i < slots; ++i)
          transformed[i] += mask[i] * out[((i + offset) % slots + slots) % slots];
      if (transformed != next)
        throw std::runtime_error("routing destination masks changed the stage output");
      out = std::move(next);
    }
    if (out != expected) throw std::runtime_error("routing differs from direct indexing");
  }
}
static void check_diagonals(int slots, int base, int stride, int count, bool naf) {
  std::vector<int> offsets;
  std::map<int, std::vector<double>> masks;
  std::vector<double> input(slots), expected(slots), actual(slots);
  for (int i = 0; i < slots; ++i) input[i] = std::sin(i + 0.2);
  for (int j = 0; j < count; ++j) {
    const int offset = base + j * stride;
    offsets.push_back(offset);
    auto& mask = masks[offset]; mask.resize(slots);
    for (int i = 0; i < slots; ++i) {
      mask[i] = std::cos(i * 0.3 + j);
      expected[i] += mask[i] * input[((i + offset) % slots + slots) % slots];
    }
  }
  std::sort(offsets.begin(), offsets.end());
  const auto plan = fhemamba::plan_packed_diagonals(offsets, slots, naf);
  if (!plan.baby_step) return;
  const int step = plan.offsets[1] - plan.offsets[0];
  for (int first = 0; first < count; first += plan.baby_step) {
    const int giant = first * step;
    std::vector<double> inner(slots);
    for (int j = 0; j < plan.baby_step && first + j < count; ++j) {
      const auto mask = fhemamba::packed_diagonal_pre_mask(masks.at(plan.offsets[first + j]), giant);
      for (int i = 0; i < slots; ++i)
        inner[i] += mask[i] * input[((i + plan.offsets[j]) % slots + slots) % slots];
    }
    for (int i = 0; i < slots; ++i) actual[i] += inner[((i + giant) % slots + slots) % slots];
  }
  for (int i = 0; i < slots; ++i) if (std::abs(actual[i] - expected[i]) > 1e-12)
    throw std::runtime_error("BSGS differs from dense diagonal transform");
}
int main() {
  std::mt19937 random(42);
  // Exhaust every packed rotation (both signs and wrap boundaries). Composing
  // the steps must give the same permutation, using only existing ±2^k keys.
  for (int slots : {2, 8, 1024, 32768}) for (int offset = -slots; offset <= slots; ++offset) {
    const auto binary = fhemamba::rotation_steps(offset, slots, false);
    const auto naf = fhemamba::rotation_steps(offset, slots, true);
    int sum = 0, previous = 0;
    for (int step : naf) {
      const int magnitude = std::abs(step);
      if (!std::has_single_bit(static_cast<unsigned>(magnitude)) || magnitude >= slots ||
          (previous && magnitude < 4 * previous))
        throw std::runtime_error("NAF needs an unavailable or adjacent key");
      previous = magnitude; sum += step;
    }
    if ((sum - offset) % slots || naf.size() > binary.size())
      throw std::runtime_error("NAF changed the rotation or increased its cost");
  }
  // Check tails, wrap boundaries and raw IEEE-754 representations. A copy
  // kernel must preserve NaN payloads and signed zeros as well as normal data.
  for (int size : {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 1024, 32768}) {
    std::vector<double> mask(size);
    for (int i = 0; i < size; ++i) {
      const uint64_t bits = i % 3 == 0 ? UINT64_C(0x8000000000000000) :
          i % 3 == 1 ? UINT64_C(0x7ff8000000000123) : (uint64_t(random()) << 32) | random();
      std::memcpy(&mask[i], &bits, sizeof(bits));
    }
    for (int shift : {0, 1, -1, size - 1, size, size + 1, -size - 1}) {
      auto actual = fhemamba::rotate_plaintext_mask(mask, shift);
      for (int i = 0; i < size; ++i) {
        const int destination = ((i + shift) % size + size) % size;
        if (std::memcmp(&actual[destination], &mask[i], sizeof(double)) != 0)
          throw std::runtime_error("mask copy changed coefficient bits");
      }
    }
  }
  for (int slots : {8, 64, 1024, 32768}) {
    for (int trial = 0; trial < 30; ++trial) {
      std::vector<double> indices;
      for (int i = 0; i < slots; ++i) if (random() % (trial + 2) == 0) indices.push_back(i);
      if (indices.empty()) indices.push_back(0);
      check(indices, slots);
    }
  }
  for (int width : {24, 64, 128, 768}) {
    std::vector<double> strided(32768 / width), contiguous(width);
    for (int i = 0; i < static_cast<int>(strided.size()); ++i) strided[i] = i * width;
    std::iota(contiguous.begin(), contiguous.end(), 0);
    check(strided, 32768);
    check(contiguous, 32768);
  }
  for (const auto& invalid : std::vector<std::vector<double>>{{2, 1}, {1, 1}, {-1}, {1.5}, {32768}}) {
    bool rejected = false;
    try { fhemamba::monotone_routing(invalid, false); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid routing accepted");
  }
  for (int slots : {64, 1024, 32768}) for (int stride : {1, 3, 63, 127, -63})
    for (int base : {0, 17, -233}) for (int count : {3, 8, 24, 32})
      for (bool naf : {false, true}) check_diagonals(slots, base, stride, count, naf);
  const auto nonuniform = fhemamba::plan_packed_diagonals({0, 1, 4, 5}, 64);
  if (nonuniform.baby_step) throw std::runtime_error("nonuniform diagonals need the direct fallback");
  std::vector<int> heads(24);
  for (int i = 0; i < 24; ++i) heads[i] = 63 * i;
  if (!fhemamba::plan_packed_diagonals(heads, 32768).baby_step)
    throw std::runtime_error("head routing must exercise BSGS");
  std::cout << "routing matches direct indexing\n";
}
