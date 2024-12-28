cmake_minimum_required(VERSION 3.28)

# Set CUDA compiler and architecture
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)

project(hpca_nn LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)

# Include directories
include_directories(${CMAKE_SOURCE_DIR})

# Add executable target
add_executable(hpca_nn
        main.cpp
        neural_network.cpp
        linear_layer.cpp
        neuron.cpp
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

# Pass --fmad=false to the NVCC compiler
# target_compile_options(hpca_nn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)

# Link CUDA libraries (if needed)
target_link_libraries(hpca_nn ${CUDA_LIBRARIES})

# Enable verbose build output and compilation database
set(CMAKE_VERBOSE_MAKEFILE ON)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Link OpenMP (if available)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(hpca_nn PUBLIC OpenMP::OpenMP_CXX)
endif()



#########################################################
# FOR ONLY C++ PROJECT comment before and uncomment after
#########################################################

# cmake_minimum_required(VERSION 3.30)

# Define the project and specify C++ as the language
# project(hpca_nn LANGUAGES CXX)

# Set the C++ standard
# set(CMAKE_CXX_STANDARD 20)

# Include directories
# include_directories(${CMAKE_SOURCE_DIR})

# Add executable target with C++ source files
# add_executable(hpca_nn
        # src/main.cpp
        # src/neural_network.cpp
        # src/linear_layer.cpp
        # src/neuron.cpp
# )

# Specify include directories for headers
# target_include_directories(hpca_nn PRIVATE ${CMAKE_SOURCE_DIR}/include)

