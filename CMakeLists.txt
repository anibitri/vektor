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
set(VEKTOR_SANITIZE "" CACHE STRING
  "address: AddressSanitizer + UndefinedBehaviorSanitizer; thread: ThreadSanitizer")
option(VEKTOR_TESTS "Build the test suite." ON)

# Lets `#pragma omp simd` vectorise the distance loops. No OpenMP threads.
add_compile_options(-fopenmp-simd)
if(VEKTOR_NATIVE)
  add_compile_options(-march=native)
endif()
# Sanitizers apply to everything, including third-party code, so all code is checked the same way.
if(VEKTOR_SANITIZE STREQUAL "address")
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
elseif(VEKTOR_SANITIZE STREQUAL "thread")
  add_compile_options(-fsanitize=thread)
  add_link_options(-fsanitize=thread)
elseif(VEKTOR_SANITIZE)
  message(FATAL_ERROR "VEKTOR_SANITIZE must be address, thread or empty")
endif()

# Third-party code, pinned by version and checksum.
include(FetchContent)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
FetchContent_Declare(httplib
  URL https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.59.0/httplib.h
  URL_HASH SHA256=dc1e4de3e0ef3a18f3a5818fab563eb3fa45d5d3bb1db527a8d389077ed5ee6c
  DOWNLOAD_NO_EXTRACT ON)
FetchContent_MakeAvailable(json httplib)
find_package(Threads REQUIRED)

# Strict warnings for our own code only (not third-party code).
set(VEKTOR_WARNINGS -Wall -Wextra -Wpedantic -Werror)

add_library(vektor_core STATIC src/core/index.cpp src/core/hnsw.cpp src/core/io.cpp)
target_include_directories(vektor_core PUBLIC src)
target_compile_options(vektor_core PRIVATE ${VEKTOR_WARNINGS})

add_executable(vektor-bench src/bench/main.cpp src/bench/dataset.cpp src/bench/rag_eval.cpp)
target_compile_options(vektor-bench PRIVATE ${VEKTOR_WARNINGS})
# The compiler flags, written into every benchmark results file.
string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type)
get_directory_property(global_options COMPILE_OPTIONS)
list(JOIN global_options " " global_options)
target_compile_definitions(vektor-bench PRIVATE
  VEKTOR_FLAGS="${CMAKE_BUILD_TYPE}: ${CMAKE_CXX_FLAGS_${build_type}} ${global_options}")

# The server's code, as a library so the tests can use it too.
add_library(vektor_api STATIC src/server/routes.cpp src/server/rag.cpp src/server/ollama_client.cpp)
target_include_directories(vektor_api SYSTEM PUBLIC ${httplib_SOURCE_DIR})
target_link_libraries(vektor_api PUBLIC vektor_core nlohmann_json::nlohmann_json Threads::Threads)
target_compile_options(vektor_api PRIVATE ${VEKTOR_WARNINGS})

# rag-eval talks to the server over HTTP, so vektor-bench needs the HTTP and JSON libraries too.
target_link_libraries(vektor-bench PRIVATE vektor_api)

add_executable(vektor-server src/server/main.cpp)
target_link_libraries(vektor-server PRIVATE vektor_api)
target_compile_options(vektor-server PRIVATE ${VEKTOR_WARNINGS})

if(VEKTOR_TESTS)
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
    tests/io_test.cpp
    tests/dataset_test.cpp
    tests/server_test.cpp
    src/bench/dataset.cpp)
  target_link_libraries(vektor_tests PRIVATE vektor_api GTest::gtest_main)
  target_compile_options(vektor_tests PRIVATE ${VEKTOR_WARNINGS})
  add_test(NAME vektor_tests COMMAND vektor_tests)
endif()
