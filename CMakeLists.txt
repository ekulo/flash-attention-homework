cmake_minimum_required(VERSION 3.20)

# 必须在 project() 之前设置:CUDA 编译器自检(CMakeDetermineCompilerId)会
# 用这里的 flags 编译测试文件。CUDA 13.x 的 host_config.h 官方只放行
# MSVC <= VS2022;VS2026(MSVC 19.50)需显式放行,否则自检直接失败:
#   #error: unsupported Microsoft Visual Studio version!
if(WIN32)
  if(NOT CMAKE_CUDA_FLAGS MATCHES "allow-unsupported")
    set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler ${CMAKE_CUDA_FLAGS}"
        CACHE STRING "nvcc flags (VS2026 override)" FORCE)
  endif()
endif()

project(flash_attn_homework LANGUAGES C CXX CUDA)

# 单配置生成器(Ninja)默认 Release,避免 Debug 的 /RTC1 /Od 与优化冲突
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# 大型作业:高性能 FlashAttention(简化版)
# 六个阶段共享一个 CMake 工程:
#   fa_api   —— 共享库(全部 stage 的 kernel + host wrapper,供 Python ctypes 调用)
#   fa_bench —— 命令行基准程序(输出 build/results/bench.json,供报告脚本使用)
#
# 常用配置:
#   -DFA_ARCHS="120"       覆盖目标架构(默认 "120";A100/RTX30 用 "80;86",
#                          RTX40 用 "89",H100 用 "90",RTX 5090 用 "120")
#   -DFA_USE_NCCL=ON       阶段5 使用 NCCL AllGather(需 Linux + 已安装 NCCL)
#   构建:cmake -S . -B build && cmake --build build -j
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # 默认按用户的 RTX 5060 Ti(Blackwell 消费级,sm_120)编译。
  # 注意:sm_120 需要 CUDA Toolkit >= 12.8。
  set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA arch list" FORCE)
endif()

option(FA_USE_NCCL "阶段5 使用 NCCL backend(否则使用 cudaMemcpyPeer 多流 backend)" OFF)

if(MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

file(GLOB FA_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)
list(REMOVE_ITEM FA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/fa_api.cu)

# ---- fa_api 共享库 ----------------------------------------------------------
add_library(fa_api SHARED ${FA_SOURCES} src/fa_api.cu)
target_include_directories(fa_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(fa_api PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
target_compile_options(fa_api PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# fp16/wmma 依赖 SM80+;架构列表中低于 80 的会被 nvcc 单独剔除编译错误。
# 注意:不再追加 -O3 —— Release 构建类型已含 -O3,Debug 下强加会与 /RTC1 冲突。

if(FA_USE_NCCL)
  find_path(NCCL_INCLUDE_DIR nccl.h HINTS $ENV{NCCL_HOME}/include /usr/include /usr/local/include)
  find_library(NCCL_LIBRARY nccl HINTS $ENV{NCCL_HOME}/lib /usr/lib /usr/local/lib)
  if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    target_include_directories(fa_api PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(fa_api PRIVATE ${NCCL_LIBRARY})
    target_compile_definitions(fa_api PRIVATE FA_HAVE_NCCL=1)
    message(STATUS "FA_HAVE_NCCL=1: 阶段5 使用 NCCL AllGather")
  else()
    message(WARNING "未找到 NCCL,阶段5 回退到 cudaMemcpyPeer backend")
  endif()
endif()

# ---- fa_bench 基准程序 ------------------------------------------------------
add_executable(fa_bench bench/bench_main.cu)
target_include_directories(fa_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(fa_bench PRIVATE fa_api)
set_target_properties(fa_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
target_compile_options(fa_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

enable_testing()
