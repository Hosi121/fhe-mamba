#include "plaintext_cache.hpp"
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

static void require(bool condition) {
  if (!condition) throw std::runtime_error("plaintext cache contract failed");
}

struct CollidingFingerprint {
  auto operator()(const std::vector<double>& values) const -> std::optional<uint64_t> {
    return fhemamba::MaskFingerprint{}(values) ? std::optional<uint64_t>(0) : std::nullopt;
  }
};

static void contracts(bool indexed) {
  // Deliberate hash collisions must never reuse different coefficients, sizes,
  // or CKKS levels. Signed zero is part of the exact encoding input.
  fhemamba::MaskPlaintextCache<std::shared_ptr<int>, CollidingFingerprint> cache(3, indexed);
  int encodes = 0;
  auto encode = [&] { return std::make_shared<int>(++encodes); };
  auto first = cache.get({0, 1, 1, 0}, 5, encode);
  require(cache.get({0, 1, 1, 0}, 5, encode) == first && encodes == 1);
  require(cache.get({0, 1, 1, 0}, 6, encode) != first);
  require(cache.get({-0.0, 1, 1, 0}, 5, encode) != first);
  require(cache.get({0, 1, 1, 0}, 5, encode) == first);  // Refresh LRU age.
  require(cache.get({0, 1, 1}, 5, encode) != first);
  require(cache.get({0, 1, 1, 0}, 5, encode) == first);
  require(cache.size() == 3 && cache.evictions == 1);
  require(cache.get({0, 1, 1, 0}, 6, encode) != first && encodes == 5);
  const auto misses = cache.misses;
  for (const auto& values : std::vector<std::vector<double>>{
           {}, {1, 2}, {1, -1}, {std::numeric_limits<double>::infinity()},
           {std::numeric_limits<double>::quiet_NaN()}}) {
    const auto a = cache.get(values, 5, encode), b = cache.get(values, 5, encode);
    require(a != b);
  }
  require(cache.bypasses == 10 && cache.misses == misses && cache.size() == 3);

  // Eviction releases backend ownership, while a current caller can still
  // hold a value safely. The production caller also synchronizes device use.
  fhemamba::MaskPlaintextCache<std::shared_ptr<int>> one(1, indexed);
  auto held = one.get({0, 2, 0}, 1, encode);
  std::weak_ptr<int> old = held;
  one.get({0, 3, 0}, 1, encode);
  require(!old.expired());
  held.reset();
  require(old.expired());
  auto last = one.get({0, 0, 0}, 1, encode);
  require(one.get({0, 0, 0}, 1, encode) == last);
  require(one.get({0, 0, -0.0}, 1, encode) != last);
  fhemamba::MaskPlaintextCache<std::shared_ptr<int>> disabled(0, indexed);
  auto uncached = disabled.get({1}, 0, encode);
  require(disabled.get({1}, 0, encode) != uncached && disabled.size() == 0);
}

int main() {
  contracts(false); contracts(true);
  fhemamba::MaskPlaintextCache<int> baseline(17), candidate(17, true);
  int base_encodes = 0, candidate_encodes = 0;
  std::mt19937 random(35);
  for (int iteration = 0; iteration < 3000; ++iteration) {
    std::vector<double> mask(128);
    const int length = random() % 9, offset = random() % 4;
    for (int i = offset; i < offset + length; ++i) mask[i] = iteration % 3 ? 1.0 : 2.0;
    if (iteration % 7 == 0) mask.back() = -0.0;
    if (iteration % 11 == 0) mask.front() = std::numeric_limits<double>::infinity();
    const int level = random() % 3;
    require(fhemamba::MaskFingerprint{}(mask).has_value() == fhemamba::RunMaskFingerprint{}(mask).has_value());
    require(baseline.get(mask, level, [&] { return ++base_encodes; }) ==
            candidate.get(mask, level, [&] { return ++candidate_encodes; }));
  }
  require(baseline.hits == candidate.hits && baseline.misses == candidate.misses &&
          baseline.evictions == candidate.evictions && baseline.bypasses == candidate.bypasses);
  // Moving an indexed cache must preserve the list iterators stored by its index.
  auto moved = std::move(candidate);
  require(moved.get({1, 0}, 2, [&] { return 17; }) == 17);
  require(moved.get({1, 0}, 2, [&] { return 18; }) == 17);
}
