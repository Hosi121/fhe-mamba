cmake_minimum_required(VERSION 3.25.2)
project(rotation_stream_chain LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(fideslib REQUIRED CONFIG)
find_package(OpenFHE REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")
set(NATIVE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../native/fideslib_stage0/src"
    CACHE PATH "Native helper source directory")
add_executable(rotation_probe probe.cpp)
target_link_libraries(rotation_probe PRIVATE fideslib::fideslib CUDA::cudart)
target_include_directories(rotation_probe PRIVATE ${NATIVE_SOURCE}
  ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke
  ${OpenFHE_INCLUDE}/binfhe ${OpenFHE_INCLUDE}/third-party/include)
