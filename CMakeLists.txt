cmake_minimum_required(VERSION 3.30)

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
)

# Set target properties
set_target_properties(hpca_nn PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CXX
)

# Link CUDA libraries (if needed)
target_link_libraries(hpca_nn ${CUDA_LIBRARIES})

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

