cmake_minimum_required(VERSION 3.25.2)
project(gpu_dual_ring LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
find_package(fideslib REQUIRED CONFIG)
find_package(OpenFHE REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")
set(NATIVE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../native/fideslib_stage0/src")
add_executable(gpu_dual_ring probe.cpp ${NATIVE_SOURCE}/fideslib_ring_map.cu)
target_link_libraries(gpu_dual_ring PRIVATE fideslib::fideslib CUDA::cudart)
target_include_directories(gpu_dual_ring PRIVATE ${NATIVE_SOURCE} ${OpenFHE_INCLUDE}
  ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke ${OpenFHE_INCLUDE}/binfhe
  ${OpenFHE_INCLUDE}/third-party/include)
