#include <openfhe.h>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>

int main() {
  using Native = lbcrypto::NativeInteger;
  constexpr bool eligible = std::is_standard_layout_v<Native> &&
      sizeof(Native) == sizeof(uint64_t) &&
      std::is_same_v<typename Native::Integer, uint64_t> &&
      std::endian::native == std::endian::little && BLOCK_VECTOR_ALLOCATION != 1;
  if (!eligible) return 1;
  uint64_t seed = 0x973F7C912E5B61A4ULL;
  std::size_t checked = 0;
  for (std::size_t size : {1u, 32u, 1024u, 65536u}) {
    lbcrypto::NativeVector values(size, Native(1152921504606846883ULL));
    for (std::size_t i = 0; i < size; ++i) {
      seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
      values[i] = i == 0 ? 0 : i == 1 ? UINT64_MAX : seed;
    }
    std::vector<uint64_t> actual(size);
    std::memcpy(actual.data(), std::addressof(values[0]), size * sizeof(uint64_t));
    for (std::size_t i = 0; i < size; ++i) {
      if (actual[i] != values[i].ConvertToInt()) return 2;
      ++checked;
    }
  }
  std::cout << "{\"passed\":true,\"scope\":\"Local host representation check; target GPU exactness is separate\","
            << "\"words_checked\":" << checked << ",\"native_bytes\":" << sizeof(Native)
            << ",\"standard_layout\":true,\"block_vector_allocation\":" << BLOCK_VECTOR_ALLOCATION
            << ",\"trivially_copyable\":" << (std::is_trivially_copyable_v<Native> ? "true" : "false") << "}\n";
}
