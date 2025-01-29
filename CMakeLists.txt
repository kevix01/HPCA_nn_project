cmake_minimum_required(VERSION 3.28)

# Enable CUDA language support
project(hpca_nn LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)

# Automatically detect CUDA architectures or fall back to a default
set(CMAKE_CUDA_ARCHITECTURES native)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(WARNING "Failed to detect CUDA architectures. Defaulting to architecture 75 (Turing GPUs).")
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR})

# Add executable target
add_executable(hpca_nn
        main.cpp
        neural_network.cpp
        linear_layer.cpp
        device_type.h
        forward_cuda.h
        forward_cuda.cu
        parallel_impl_cpu.h
        parameters.h
        parameters.cpp
        dataset_loader.h
        dataset_loader.cpp
        backward_cuda.cu
        backward_cuda.h
        actf_type_cuda.h
        times_printing.cpp
        times_printing.h
)

# Set target properties
set_target_properties(hpca_nn PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CXX
)

# Link CUDA libraries (if needed)
target_link_libraries(hpca_nn ${CUDA_LIBRARIES})

# Enable verbose build output and compilation database
set(CMAKE_VERBOSE_MAKEFILE ON)

# Link OpenMP (if available)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hpca_nn PUBLIC OpenMP::OpenMP_CXX)
endif()
