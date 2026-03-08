cmake_minimum_required(VERSION 3.18)
project(CudaPybind11Example LANGUAGES CXX CUDA)

# 1. Set C++ and CUDA Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 2. Target the L4 GPU (sm_89)
# This replaces the -arch=sm_89 flag
set(CMAKE_CUDA_ARCHITECTURES 89)

# 3. Find Dependencies
set(PYBIND11_FINDPYTHON ON)
find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# 4. Define the Library
# pybind11_add_module handles the -shared and -fPIC flags automatically
pybind11_add_module(cuda_kernels ../kernel.cu)

# 5. Tell CMake that cuda_lib needs to be treated as CUDA code
set_target_properties(cuda_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 6. Link libraries (like the CUDA runtime)
target_link_libraries(cuda_kernels PRIVATE pybind11::module)