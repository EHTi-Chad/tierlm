cmake_minimum_required(VERSION 3.24)
project(tierlm C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TIERLM_CUDA "Build ggml with CUDA" OFF)
set(GGML_CUDA ${TIERLM_CUDA} CACHE BOOL "" FORCE)
add_subdirectory(third_party/ggml)

add_library(tierlm_core src/model_map.cpp src/planner.cpp)
target_include_directories(tierlm_core PUBLIC src)
target_link_libraries(tierlm_core PUBLIC ggml)

add_executable(tierlm-inspect tools/inspect.cpp)
target_link_libraries(tierlm-inspect PRIVATE tierlm_core)

add_executable(tierlm-plan tools/plan.cpp)
target_link_libraries(tierlm-plan PRIVATE tierlm_core)
