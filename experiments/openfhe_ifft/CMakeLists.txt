cmake_minimum_required(VERSION 3.25)
project(openfhe_ifft_qualification LANGUAGES CXX)

find_package(OpenFHE REQUIRED CONFIG)
if(NOT OpenFHE_NATIVE_SIZE STREQUAL "64")
  message(FATAL_ERROR "This coefficient qualification targets the pinned 64-bit build")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenFHE_CXX_FLAGS}")

foreach(probe fft_probe encoding_probe)
  add_executable(${probe} ${probe}.cpp)
  target_include_directories(${probe} PRIVATE
    ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe ${OpenFHE_INCLUDE}/third-party/include)
  target_link_libraries(${probe} PRIVATE ${OpenFHE_STATIC_LIBRARIES})
endforeach()

option(FHE_IFFT_GPU_PROBE "Build the experimental exact-order GPU inverse FFT screen" OFF)
if(FHE_IFFT_GPU_PROBE)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(gpu_fft_probe gpu_fft_probe.cpp gpu_fft.cu)
  set_target_properties(gpu_fft_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_include_directories(gpu_fft_probe PRIVATE
    ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe ${OpenFHE_INCLUDE}/third-party/include)
  target_link_libraries(gpu_fft_probe PRIVATE ${OpenFHE_STATIC_LIBRARIES} CUDA::cudart)
endif()
