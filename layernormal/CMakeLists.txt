cmake_minimum_required(VERSION 3.10)
project(CudaExample LANGUAGES CXX CUDA)
add_executable(main main.cpp)

# 启用 CUDA 编译
find_package(CUDAToolkit REQUIRED)

# 指定 CUDA 编译选项
# 指定 CUDA 编译选项
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_35")