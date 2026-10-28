cmake_minimum_required(VERSION 3.24)
# Default to an optimised build. This must come before project(), because with MSVC
# CMake otherwise quietly picks Debug, which is ~10x slower.
if(NOT DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()
project(tierlm VERSION 0.3.0 LANGUAGES C CXX)
set(TIERLM_VERSION "0.3.0-dev" CACHE STRING "Version shown in the app")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
  # Read sources as UTF-8. Only for the C/C++ compiler: NVIDIA's nvcc needs it passed through as -Xcompiler.
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>" "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()

# Device backends (ggml). CPU is always built.
option(TIERLM_CUDA   "Build the CUDA backend (NVIDIA)"          OFF)
option(TIERLM_VULKAN "Build the Vulkan backend (Intel/AMD/NVIDIA)" OFF)
# Portable builds (for releases) run on every 64-bit Intel or AMD CPU: ggml's engines become separate files, with one
# CPU engine per x86 generation (plain x86-64, SSE4.2, AVX, AVX2, AVX-VNNI, AVX-512, Zen 4, AMX...), and the best one
# this computer supports is loaded when tierlm starts. A GPU engine that can't load (CUDA without an NVIDIA driver)
# is skipped. On Windows the C runtime is built in, so no Visual C++ redistributable is needed.
# Other builds are one program each, tuned for the build machine's CPU.
option(TIERLM_PORTABLE "Build binaries for distribution (all CPU generations, engines as separate files)" OFF)
if(TIERLM_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
  set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
endif()
# Programs and engine files all in the build folder, where ggml looks for the engines.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_BUILD_RPATH "$ORIGIN")
set(GGML_CUDA   ${TIERLM_CUDA}   CACHE BOOL "" FORCE)
set(GGML_VULKAN ${TIERLM_VULKAN} CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml)

# Windows builds with CUDA load NVIDIA's driver DLL (nvcuda.dll) only when it is first needed, which is only
# after an NVIDIA GPU has been found. Without this, the same program wouldn't even start on a PC with an Intel
# or AMD GPU, so one download couldn't serve everyone.
# (With engines as separate files, as in release builds, the CUDA engine file simply fails to load instead.)
if(MSVC AND TIERLM_CUDA AND NOT GGML_BACKEND_DL)
  add_link_options(/DELAYLOAD:nvcuda.dll)
  link_libraries(delayimp)
endif()

add_library(tierlm_core STATIC src/model_map.cpp src/planner.cpp src/tokenizer.cpp src/model.cpp src/gguf_split.cpp src/speculate.cpp src/backends.cpp)
target_include_directories(tierlm_core PUBLIC src)
target_link_libraries(tierlm_core PUBLIC ggml)

foreach(tool inspect plan tok_test logits run bench)
  string(REPLACE "_" "-" name ${tool})
  add_executable(tierlm-${name} tools/${tool}.cpp)
  target_link_libraries(tierlm-${name} PRIVATE tierlm_core)
endforeach()

# Test helper: with -DGGML_RPC=ON, serves the CPU as a stand-in GPU so VRAM/RAM splitting can be tested anywhere.
if(GGML_RPC AND NOT GGML_BACKEND_DL)
  add_executable(tierlm-rpc-server tools/rpc_server.cpp)
  target_link_libraries(tierlm-rpc-server PRIVATE ggml)
endif()

add_executable(tierlm-kv tools/kv.cpp)
target_link_libraries(tierlm-kv PRIVATE ggml)

add_executable(tierlm-devices tools/devices.cpp)
target_link_libraries(tierlm-devices PRIVATE tierlm_core)

# Web server + dashboard (third_party/httplib and third_party/json are single-header, MIT licensed)
find_package(Threads REQUIRED)
add_executable(tierlm-server tools/server.cpp)
target_include_directories(tierlm-server PRIVATE third_party/httplib third_party/json)
target_link_libraries(tierlm-server PRIVATE tierlm_core Threads::Threads)
target_compile_definitions(tierlm-server PRIVATE TIERLM_VERSION="${TIERLM_VERSION}")
target_compile_definitions(tierlm-devices PRIVATE TIERLM_VERSION="${TIERLM_VERSION}")
if(WIN32)
  target_compile_definitions(tierlm-server PRIVATE NOMINMAX _WIN32_WINNT=0x0A00)
  if(MSVC)
    target_compile_options(tierlm-server PRIVATE /bigobj)
  endif()
  target_link_libraries(tierlm-server PRIVATE ws2_32 ole32 shell32 uuid)
endif()
