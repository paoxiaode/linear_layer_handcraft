cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(linear LANGUAGES CXX CUDA)

set(TORCH_INSTALL_PREFIX "/opt/conda/lib/python3.8/site-packages/torch/")
set(Torch_DIR "/opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/")

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

include_directories(/opt/conda/include/python3.8/)
include_directories(common)
include_directories(ops/include)

set(SRCS ops/kernel/cudabind.cpp ops/kernel/gemm_cuda.cu src/gemm.cpp src/pybind.cpp)

add_library(linear SHARED ${SRCS})

target_link_libraries(linear "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
