cmake_minimum_required(VERSION 3.20)
project(vektor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clang-tidy

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VEKTOR_NATIVE "Tune for this machine's CPU (-march=native). Local benchmarks only." OFF)
option(VEKTOR_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer." OFF)
option(VEKTOR_TESTS "Build the test suite." ON)

# Lets `#pragma omp simd` vectorise the distance loops. No OpenMP threads.
add_compile_options(-fopenmp-simd)
if(VEKTOR_NATIVE)
  add_compile_options(-march=native)
endif()
if(VEKTOR_SANITIZE)
  # Applied to everything, including GoogleTest, so all code is checked the same way.
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

# Strict warnings for our own code only (not third-party code).
set(VEKTOR_WARNINGS -Wall -Wextra -Wpedantic -Werror)

add_library(vektor_core STATIC src/core/index.cpp src/core/hnsw.cpp)
target_include_directories(vektor_core PUBLIC src)
target_compile_options(vektor_core PRIVATE ${VEKTOR_WARNINGS})

add_executable(vektor-bench src/bench/main.cpp src/bench/dataset.cpp)
target_link_libraries(vektor-bench PRIVATE vektor_core)
target_compile_options(vektor-bench PRIVATE ${VEKTOR_WARNINGS})
# The compiler flags, written into every benchmark results file.
string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type)
get_directory_property(global_options COMPILE_OPTIONS)
list(JOIN global_options " " global_options)
target_compile_definitions(vektor-bench PRIVATE
  VEKTOR_FLAGS="${CMAKE_BUILD_TYPE}: ${CMAKE_CXX_FLAGS_${build_type}} ${global_options}")

if(VEKTOR_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/releases/download/v1.18.0/googletest-1.18.0.tar.gz
    URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(vektor_tests
    tests/distance_test.cpp
    tests/index_test.cpp
    tests/hnsw_test.cpp
    tests/dataset_test.cpp
    src/bench/dataset.cpp)
  target_link_libraries(vektor_tests PRIVATE vektor_core GTest::gtest_main)
  target_compile_options(vektor_tests PRIVATE ${VEKTOR_WARNINGS})
  add_test(NAME vektor_tests COMMAND vektor_tests)
endif()
