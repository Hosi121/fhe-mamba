cmake_minimum_required(VERSION 3.25.2)
project(refresh_correction LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(fideslib REQUIRED CONFIG)
find_package(OpenFHE REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")
add_executable(refresh_correction_probe probe.cpp)
target_link_libraries(refresh_correction_probe PRIVATE fideslib::fideslib CUDA::cudart)
target_include_directories(refresh_correction_probe PRIVATE ../../native/fideslib_stage0/src
  ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke
  ${OpenFHE_INCLUDE}/binfhe ${OpenFHE_INCLUDE}/third-party/include)
