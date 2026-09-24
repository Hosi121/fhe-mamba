#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
int main() {
  for (const char* name : {"memcpy", "memmove"}) {
    void* function = dlsym(RTLD_DEFAULT, name);
    Dl_info info{};
    if (!function || !dladdr(function, &info)) return 1;
    const auto offset = reinterpret_cast<std::uintptr_t>(function) - reinterpret_cast<std::uintptr_t>(info.dli_fbase);
    std::cout << name << ' ' << info.dli_fname << " 0x" << std::hex << offset << '\n';
  }
}
