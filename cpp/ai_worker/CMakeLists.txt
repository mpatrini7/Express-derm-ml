cmake_minimum_required(VERSION 3.22)

project(express_derm_ai_worker VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json 3.10 REQUIRED)

find_path(
  TENSORRT_INCLUDE_DIR
  NvInfer.h
  HINTS
    /usr/include/aarch64-linux-gnu
    /usr/local/cuda/include
    /usr/include
)
find_library(
  TENSORRT_LIBRARY
  nvinfer
  HINTS
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda/lib64
    /usr/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT C++ headers/library not found. Install the TensorRT development packages for this NVIDIA GPU system.")
endif()

add_executable(
  express-derm-ai-worker
  src/main.cpp
  src/sha256.cpp
)

target_include_directories(
  express-derm-ai-worker
  PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
  express-derm-ai-worker
  PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS}
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
)
target_compile_options(
  express-derm-ai-worker
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
    -Werror=return-type
)

install(TARGETS express-derm-ai-worker RUNTIME DESTINATION bin)
