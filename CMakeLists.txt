cmake_minimum_required(VERSION 3.24)
# Default to an optimised build. This must come before project(), because with MSVC
# CMake otherwise quietly picks Debug, which is ~10x slower.
if(NOT DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()
project(tierlm C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
  add_compile_options(/utf-8)
endif()

# Device backends (ggml). CPU is always built.
option(TIERLM_CUDA   "Build the CUDA backend (NVIDIA)"          OFF)
option(TIERLM_VULKAN "Build the Vulkan backend (Intel/AMD/NVIDIA)" OFF)
set(GGML_CUDA   ${TIERLM_CUDA}   CACHE BOOL "" FORCE)
set(GGML_VULKAN ${TIERLM_VULKAN} CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml)

add_library(tierlm_core src/model_map.cpp src/planner.cpp src/tokenizer.cpp src/olmoe.cpp)
target_include_directories(tierlm_core PUBLIC src)
target_link_libraries(tierlm_core PUBLIC ggml)

foreach(tool inspect plan tok_test logits run)
  string(REPLACE "_" "-" name ${tool})
  add_executable(tierlm-${name} tools/${tool}.cpp)
  target_link_libraries(tierlm-${name} PRIVATE tierlm_core)
endforeach()

add_executable(tierlm-kv tools/kv.cpp)
target_link_libraries(tierlm-kv PRIVATE ggml)

add_executable(tierlm-devices tools/devices.cpp)
target_link_libraries(tierlm-devices PRIVATE ggml)

# Web server + dashboard (third_party/httplib and third_party/json are single-header, MIT licensed)
find_package(Threads REQUIRED)
add_executable(tierlm-server tools/server.cpp)
target_include_directories(tierlm-server PRIVATE third_party/httplib third_party/json)
target_link_libraries(tierlm-server PRIVATE tierlm_core Threads::Threads)
if(WIN32)
  target_compile_definitions(tierlm-server PRIVATE NOMINMAX _WIN32_WINNT=0x0A00)
  if(MSVC)
    target_compile_options(tierlm-server PRIVATE /bigobj)
  endif()
  target_link_libraries(tierlm-server PRIVATE ws2_32)
endif()
