#include "plaintext_cache.hpp"
#include <limits>
#include <memory>
#include <stdexcept>

static void require(bool condition) {
  if (!condition) throw std::runtime_error("plaintext cache contract failed");
}

struct CollidingFingerprint {
  auto operator()(const std::vector<double>& values) const -> std::optional<uint64_t> {
    return fhemamba::MaskFingerprint{}(values) ? std::optional<uint64_t>(0) : std::nullopt;
  }
};

int main() {
  // Deliberate hash collisions must never reuse different coefficients, sizes,
  // or CKKS levels. Signed zero is part of the exact encoding input.
  fhemamba::MaskPlaintextCache<std::shared_ptr<int>, CollidingFingerprint> cache(3);
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
  fhemamba::MaskPlaintextCache<std::shared_ptr<int>> one(1);
  auto held = one.get({0, 2, 0}, 1, encode);
  std::weak_ptr<int> old = held;
  one.get({0, 3, 0}, 1, encode);
  require(!old.expired());
  held.reset();
  require(old.expired());
  auto last = one.get({0, 0, 0}, 1, encode);
  require(one.get({0, 0, 0}, 1, encode) == last);
  require(one.get({0, 0, -0.0}, 1, encode) != last);
  fhemamba::MaskPlaintextCache<std::shared_ptr<int>> disabled(0);
  auto uncached = disabled.get({1}, 0, encode);
  require(disabled.get({1}, 0, encode) != uncached && disabled.size() == 0);
}
