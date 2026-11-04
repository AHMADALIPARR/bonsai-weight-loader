# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
# Copyright Ahmad Ali Parr / SnapKitty Research Lab, 2026
cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures (sm_XX numbers)")
endif()

project(bonsai_weight_loader LANGUAGES CXX)

option(BONSAI_CUDA "Build the CUDA loader, kernels, CLI and GPU tests" ON)
option(BONSAI_PTXAS_VERBOSE "Print -Xptxas=-v register/spill info for the kernels" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

set(BONSAI_HAVE_CUDA OFF)
if(BONSAI_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(BONSAI_HAVE_CUDA ON)
  else()
    message(STATUS "nvcc not found: building the CPU library, tools and tests only")
  endif()
endif()

# ---------------------------------------------------------------- CPU only --
add_library(bonsai_core STATIC
  src/ggml_types.cpp
  src/gguf.cpp
  src/raw_manifest.cpp
  src/mapped_file.cpp
  src/tile_layout.cpp
  src/quant.cpp
  src/synth.cpp
  src/block_scale_ref.cpp)
target_include_directories(bonsai_core PUBLIC include)

add_executable(make_synth_gguf tools/make_synth_gguf.cpp)
target_link_libraries(make_synth_gguf PRIVATE bonsai_core)
add_executable(bonsai_index tools/bonsai_index.cpp)
target_link_libraries(bonsai_index PRIVATE bonsai_core)
add_executable(cpu_bench bench/cpu_bench.cpp)
target_link_libraries(cpu_bench PRIVATE bonsai_core)

enable_testing()

function(bonsai_cpu_test name)
  add_executable(${name} tests/${name}.cpp)
  target_include_directories(${name} PRIVATE tests)
  target_link_libraries(${name} PRIVATE bonsai_core)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS cpu)
endfunction()

bonsai_cpu_test(test_gguf_parse)
bonsai_cpu_test(test_tiling)
bonsai_cpu_test(test_raw_manifest)
bonsai_cpu_test(test_quant)
bonsai_cpu_test(test_block_scale)
bonsai_cpu_test(test_int4_gemm)

# The synthetic file round-trips through the C++ generator and the CPU index tool.
add_test(NAME synth_index
  COMMAND ${CMAKE_COMMAND} -DGEN=$<TARGET_FILE:make_synth_gguf> -DINDEX=$<TARGET_FILE:bonsai_index>
          -DOUT=${CMAKE_CURRENT_BINARY_DIR}/synth.gguf -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/synth_index_test.cmake)
set_tests_properties(synth_index PROPERTIES LABELS cpu)

# -------------------------------------------------------------------- CUDA --
if(BONSAI_HAVE_CUDA)
  set(BONSAI_KERNEL_SOURCES src/loader.cu src/block_scale.cu src/int4_gemm.cu)
  add_library(bonsai_cuda STATIC ${BONSAI_KERNEL_SOURCES} src/load_bonsai_tensor_direct.cpp)
  target_link_libraries(bonsai_cuda PUBLIC bonsai_core CUDA::cudart)
  target_compile_options(bonsai_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
  if(BONSAI_PTXAS_VERBOSE)
    set_source_files_properties(${BONSAI_KERNEL_SOURCES} PROPERTIES COMPILE_OPTIONS "-Xptxas=-v")
  endif()

  add_executable(bonsai_load tools/bonsai_load.cu)
  target_include_directories(bonsai_load PRIVATE tools)
  target_link_libraries(bonsai_load PRIVATE bonsai_cuda)

  function(bonsai_gpu_test name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE tests)
    target_link_libraries(${name} PRIVATE bonsai_cuda)
    add_test(NAME ${name} COMMAND ${name})
    # exit code 77 = no CUDA device on this machine
    set_tests_properties(${name} PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  endfunction()

  bonsai_gpu_test(test_gpu_loader tests/test_gpu_loader.cu)
  bonsai_gpu_test(test_gpu_block_scale tests/test_gpu_block_scale.cu)
  bonsai_gpu_test(test_gpu_int4_gemm tests/test_gpu_int4_gemm.cu)
  bonsai_gpu_test(gpu_bench bench/gpu_bench.cu)
  set_tests_properties(gpu_bench PROPERTIES LABELS "gpu;bench")

  # No spills and no local memory in any kernel: PTX has no ld.local/st.local
  # and ptxas reports 0 spill bytes for every architecture built.
  add_test(NAME kernel_resources
    COMMAND ${CMAKE_COMMAND}
      -DNVCC=${CMAKE_CUDA_COMPILER}
      "-DARCHS=${CMAKE_CUDA_ARCHITECTURES}"
      -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      "-DSOURCES=${BONSAI_KERNEL_SOURCES}"
      -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/ptxas
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_kernel_resources.cmake)
  set_tests_properties(kernel_resources PROPERTIES LABELS "cpu;resources")
endif()
