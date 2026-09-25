cmake_minimum_required(VERSION 3.25.2)

project(fideslib_stage0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FHE_STAGE0_BUILD_KERNEL "Build the FIDESlib Stage 0 kernel" ON)
option(FHE_STAGE0_BUILD_TESTS "Build FIDESlib-free Stage 0 layout tests" OFF)

file(
  READ
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/fhemamba/_version.py"
  FHEMAMBA_VERSION_SOURCE)
string(
  REGEX MATCH
  "__version__[ \t]*=[ \t]*\"([^\"]+)\""
  FHEMAMBA_VERSION_MATCH
  "${FHEMAMBA_VERSION_SOURCE}")
set(FHEMAMBA_VERSION "${CMAKE_MATCH_1}")
if(FHEMAMBA_VERSION STREQUAL "")
  message(FATAL_ERROR "Unable to read fhemamba package version")
endif()

add_library(stage0_layout INTERFACE)
target_include_directories(stage0_layout INTERFACE include)

add_library(stage1_tail_payload INTERFACE)
target_include_directories(stage1_tail_payload INTERFACE include)

add_library(stage1_rank_gate_payload INTERFACE)
target_include_directories(stage1_rank_gate_payload INTERFACE include)

add_library(stage1_mamba2_config STATIC src/stage1_mamba2_config.cpp)
target_include_directories(stage1_mamba2_config PUBLIC src)

add_library(stage1_mamba2_depth STATIC src/stage1_mamba2_depth.cpp)
target_include_directories(stage1_mamba2_depth PUBLIC src)
target_link_libraries(stage1_mamba2_depth PUBLIC stage1_mamba2_payload)

add_library(stage1_mamba2_artifact STATIC src/stage1_mamba2_artifact.cpp)
target_include_directories(stage1_mamba2_artifact PUBLIC src)
target_link_libraries(stage1_mamba2_artifact PUBLIC stage1_mamba2_config)

add_library(stage1_mamba2_payload STATIC src/stage1_mamba2_payload.cpp)
target_include_directories(stage1_mamba2_payload PUBLIC src include)

add_library(stage1_mamba2_plan STATIC src/stage1_mamba2_plan.cpp)
target_include_directories(stage1_mamba2_plan PUBLIC src)
target_link_libraries(stage1_mamba2_plan PUBLIC stage1_mamba2_payload)

add_executable(packed_resource_inventory src/packed_resource_inventory.cpp)
target_link_libraries(packed_resource_inventory PRIVATE stage1_mamba2_plan)

add_library(stage1_mamba2_process STATIC src/stage1_mamba2_process.cpp)
target_include_directories(stage1_mamba2_process PUBLIC src)

if(FHE_STAGE0_BUILD_KERNEL)
  find_package(fideslib REQUIRED CONFIG)
  find_package(CUDAToolkit REQUIRED)
  find_package(OpenFHE REQUIRED CONFIG)

  if(fideslib_FOUND)
    message(STATUS "FOUND PACKAGE FIDESlib")
    message(STATUS "FIDESlib Version: ${fideslib_VERSION}")
    message(STATUS "FIDESlib header files location: ${FIDESLIB_INCLUDE_PATH}")
    message(STATUS "FIDESlib lib files location: ${FIDESLIB_LIBRARIES_PATH}")
    message(STATUS "FIDESlib C++ compiler: ${FIDESLIB_CXX_COMPILER}")
  else()
    message(FATAL_ERROR "PACKAGE FIDESlib NOT FOUND")
  endif()

  if(FIDESLIB_CXX_COMPILER)
    get_filename_component(FIDESLIB_CXX_COMPILER_REAL "${FIDESLIB_CXX_COMPILER}" REALPATH)
    get_filename_component(CMAKE_CXX_COMPILER_REAL "${CMAKE_CXX_COMPILER}" REALPATH)
    if(NOT FIDESLIB_CXX_COMPILER_REAL STREQUAL CMAKE_CXX_COMPILER_REAL)
      message(
        FATAL_ERROR
        "Compiler mismatch: configure with -DCMAKE_CXX_COMPILER=${FIDESLIB_CXX_COMPILER}")
    endif()
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")

  add_library(fideslib_handoff STATIC src/fideslib_handoff.cpp)
  target_link_libraries(
    fideslib_handoff PUBLIC fideslib::fideslib CUDA::cudart)
  target_include_directories(
    fideslib_handoff PRIVATE
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_INCLUDE}/third-party/include)

  add_executable(stage1_bootstrap_probe src/stage1_bootstrap_probe.cpp)
  target_link_libraries(stage1_bootstrap_probe PRIVATE fideslib::fideslib)

  add_executable(packed_fideslib src/packed_fideslib.cpp)
  add_library(fideslib_plaintext_encoder STATIC src/fideslib_plaintext_encoder.cpp)
  target_link_libraries(fideslib_plaintext_encoder PUBLIC fideslib::fideslib CUDA::cudart)
  target_include_directories(fideslib_plaintext_encoder PUBLIC src
    ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe ${OpenFHE_INCLUDE}/third-party/include)
  target_link_libraries(packed_fideslib PRIVATE fideslib::fideslib CUDA::cudart stage0_layout stage1_mamba2_plan fideslib_plaintext_encoder)
  add_executable(packed_plaintext_probe src/packed_plaintext_probe.cpp)
  target_link_libraries(packed_plaintext_probe PRIVATE fideslib_plaintext_encoder)
  add_executable(owned_arithmetic_probe src/owned_arithmetic_probe.cpp)
  target_link_libraries(owned_arithmetic_probe PRIVATE fideslib_plaintext_encoder stage0_layout)

  add_executable(
    stage1_mamba2_decode_fideslib
    src/stage1_mamba2_decode_fideslib.cpp)
  target_link_libraries(
    stage1_mamba2_decode_fideslib PRIVATE fideslib::fideslib CUDA::cudart fideslib_handoff fideslib_plaintext_encoder stage1_mamba2_artifact stage1_mamba2_config stage1_mamba2_depth stage1_mamba2_payload stage1_mamba2_plan stage1_mamba2_process stage1_rank_gate_payload stage1_tail_payload)
  target_include_directories(
    stage1_mamba2_decode_fideslib PRIVATE
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_INCLUDE}/third-party/include)

  add_executable(fideslib_client_server_probe src/fideslib_client_server_probe.cpp)
  target_compile_definitions(
    fideslib_client_server_probe PRIVATE FHEMAMBA_VERSION="${FHEMAMBA_VERSION}")
  target_link_libraries(
    fideslib_client_server_probe PRIVATE fideslib::fideslib CUDA::cudart fideslib_handoff)
  target_include_directories(
    fideslib_client_server_probe PRIVATE
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_INCLUDE}/third-party/include)

  add_executable(fideslib_ctpt_probe src/fideslib_ctpt_probe.cpp)
  target_compile_definitions(
    fideslib_ctpt_probe PRIVATE FHEMAMBA_VERSION="${FHEMAMBA_VERSION}")
  target_link_libraries(fideslib_ctpt_probe PRIVATE fideslib::fideslib)

  add_executable(stage1_plaintext_add_probe src/stage1_plaintext_add_probe.cpp)
  target_compile_definitions(
    stage1_plaintext_add_probe PRIVATE FHEMAMBA_VERSION="${FHEMAMBA_VERSION}")
  target_link_libraries(stage1_plaintext_add_probe PRIVATE fideslib::fideslib CUDA::cudart)

  add_executable(stage1_coefficient_encoding_probe src/stage1_coefficient_encoding_probe.cpp)
  target_compile_definitions(
    stage1_coefficient_encoding_probe PRIVATE FHEMAMBA_VERSION="${FHEMAMBA_VERSION}")
  target_link_libraries(stage1_coefficient_encoding_probe PRIVATE fideslib::fideslib CUDA::cudart)

  add_executable(stage1_subring_encoding_probe src/stage1_subring_encoding_probe.cpp)
  target_compile_definitions(
    stage1_subring_encoding_probe PRIVATE FHEMAMBA_VERSION="${FHEMAMBA_VERSION}")
  target_link_libraries(stage1_subring_encoding_probe PRIVATE fideslib::fideslib CUDA::cudart)
  target_include_directories(stage1_subring_encoding_probe PRIVATE
    ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe ${OpenFHE_INCLUDE}/third-party/include)

  add_executable(stage1_normalization_probe src/stage1_normalization_probe.cpp)
  target_compile_definitions(
    stage1_normalization_probe PRIVATE FHEMAMBA_VERSION="${FHEMAMBA_VERSION}")
  target_link_libraries(stage1_normalization_probe PRIVATE fideslib::fideslib CUDA::cudart stage0_layout)

  add_executable(stage1_vector_rms_probe src/stage1_vector_rms_probe.cpp)
  target_compile_definitions(
    stage1_vector_rms_probe PRIVATE FHEMAMBA_VERSION="${FHEMAMBA_VERSION}")
  target_link_libraries(stage1_vector_rms_probe PRIVATE fideslib::fideslib CUDA::cudart stage0_layout)
endif()

if(FHE_STAGE0_BUILD_TESTS)
  enable_testing()
  add_executable(test_packed_schedule tests/test_packed_schedule.cpp)
  target_link_libraries(test_packed_schedule PRIVATE stage0_layout)
  add_test(NAME packed_schedule COMMAND test_packed_schedule)
  add_executable(test_owned_arithmetic tests/test_owned_arithmetic.cpp)
  target_link_libraries(test_owned_arithmetic PRIVATE stage0_layout)
  add_test(NAME owned_arithmetic COMMAND test_owned_arithmetic)
  add_executable(test_packed_lifetime tests/test_packed_lifetime.cpp)
  target_link_libraries(test_packed_lifetime PRIVATE stage0_layout)
  add_test(NAME packed_lifetime COMMAND test_packed_lifetime)
  add_executable(test_plaintext_cache tests/test_plaintext_cache.cpp)
  target_link_libraries(test_plaintext_cache PRIVATE stage0_layout)
  add_test(NAME plaintext_cache COMMAND test_plaintext_cache)
  add_executable(test_packed_depth tests/test_packed_depth.cpp)
  target_link_libraries(test_packed_depth PRIVATE stage0_layout)
  add_test(NAME packed_depth COMMAND test_packed_depth)
  add_executable(test_packed_routing tests/test_packed_routing.cpp)
  target_link_libraries(test_packed_routing PRIVATE stage0_layout)
  add_test(NAME packed_routing COMMAND test_packed_routing)
  add_executable(test_packed_program tests/test_packed_program.cpp)
  target_link_libraries(test_packed_program PRIVATE stage0_layout)
  add_test(NAME packed_program COMMAND test_packed_program)
  add_executable(test_stage0_layout tests/test_stage0_layout.cpp)
  target_link_libraries(test_stage0_layout PRIVATE stage0_layout)
  add_test(NAME stage0_layout COMMAND test_stage0_layout)

  add_executable(test_stage1_normalization tests/test_stage1_normalization.cpp)
  target_link_libraries(test_stage1_normalization PRIVATE stage0_layout)
  add_test(NAME stage1_normalization COMMAND test_stage1_normalization)

  add_executable(test_stage1_vector_rms tests/test_stage1_vector_rms.cpp)
  target_link_libraries(test_stage1_vector_rms PRIVATE stage0_layout)
  add_test(NAME stage1_vector_rms COMMAND test_stage1_vector_rms)

  add_executable(test_stage1_tail_payload tests/test_stage1_tail_payload.cpp)
  target_link_libraries(test_stage1_tail_payload PRIVATE stage1_tail_payload)
  add_test(NAME stage1_tail_payload COMMAND test_stage1_tail_payload)

  add_executable(test_stage1_rank_gate_payload tests/test_stage1_rank_gate_payload.cpp)
  target_link_libraries(
    test_stage1_rank_gate_payload PRIVATE stage1_rank_gate_payload stage1_tail_payload)
  add_test(NAME stage1_rank_gate_payload COMMAND test_stage1_rank_gate_payload)

  add_executable(
    test_stage1_mamba2_config
    tests/test_stage1_mamba2_config.cpp)
  target_link_libraries(test_stage1_mamba2_config PRIVATE stage1_mamba2_config)
  add_test(NAME stage1_mamba2_config COMMAND test_stage1_mamba2_config)

  add_executable(
    test_stage1_mamba2_depth
    tests/test_stage1_mamba2_depth.cpp)
  target_link_libraries(test_stage1_mamba2_depth PRIVATE stage1_mamba2_depth)
  add_test(NAME stage1_mamba2_depth COMMAND test_stage1_mamba2_depth)

  add_executable(
    test_stage1_mamba2_artifact
    tests/test_stage1_mamba2_artifact.cpp)
  target_link_libraries(test_stage1_mamba2_artifact PRIVATE stage1_mamba2_artifact)
  add_test(NAME stage1_mamba2_artifact COMMAND test_stage1_mamba2_artifact)

  add_executable(
    test_stage1_mamba2_payload
    tests/test_stage1_mamba2_payload.cpp)
  target_link_libraries(test_stage1_mamba2_payload PRIVATE stage1_mamba2_payload)
  add_test(NAME stage1_mamba2_payload COMMAND test_stage1_mamba2_payload)

  add_executable(test_stage1_joint_gates tests/test_stage1_joint_gates.cpp)
  target_link_libraries(test_stage1_joint_gates PRIVATE stage1_mamba2_depth)
  add_test(NAME stage1_joint_gates COMMAND test_stage1_joint_gates)

  add_executable(
    test_stage1_mamba2_plan
    tests/test_stage1_mamba2_plan.cpp)
  target_link_libraries(test_stage1_mamba2_plan PRIVATE stage1_mamba2_plan)
  add_test(NAME stage1_mamba2_plan COMMAND test_stage1_mamba2_plan)

  add_executable(
    test_stage1_mamba2_process
    tests/test_stage1_mamba2_process.cpp)
  target_link_libraries(test_stage1_mamba2_process PRIVATE stage1_mamba2_process)
  add_test(NAME stage1_mamba2_process COMMAND test_stage1_mamba2_process)
endif()
