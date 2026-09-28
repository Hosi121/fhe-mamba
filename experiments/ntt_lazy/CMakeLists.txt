cmake_minimum_required(VERSION 3.25.2)
project(ntt_lazy LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
find_package(fideslib REQUIRED CONFIG)
find_package(OpenFHE REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")
add_executable(ntt_lazy_oracle ../ntt_warp_tail/probe.cu)
target_link_libraries(ntt_lazy_oracle PRIVATE fideslib::fideslib CUDA::cudart)
target_include_directories(ntt_lazy_oracle PRIVATE ${OpenFHE_INCLUDE}
  ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe
  ${OpenFHE_INCLUDE}/third-party/include)
