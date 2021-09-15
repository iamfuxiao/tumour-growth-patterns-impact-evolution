cmake_minimum_required(VERSION 3.8)

PROJECT(tumour_growth_patterns)

enable_language(CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
cuda_add_executable(tumour_growth_patterns tumour_growth_patterns.cu)

# make std=c++11
set_property(TARGET tumour_growth_patterns PROPERTY CUDA_STANDARD 11)
