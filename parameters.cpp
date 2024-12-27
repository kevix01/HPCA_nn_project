//
// Created by kevin on 18/12/24.
//

#include "parameters.h"
#include <cstdlib> // For std::atoi
#include <algorithm>
#include <iostream>

Parameters::Parameters(int argc, char* argv[]){
    parseArguments(argc, argv);
}

ParallelImplCpu Parameters::getParallelImplCpu() const {
    return parallel_impl_cpu;
}

int Parameters::getOpenmpThreads() const {
    return openmp_threads;
}

int Parameters::getCudaFTileSize() const {
    return cuda_f_tile_size;
}

int Parameters::getCudaBBlockSize() const {
    return cuda_b_block_size;
}

void Parameters::parseArguments(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--openmp_threads=") == 0) {
            openmp_threads = std::atoi(arg.substr(17).c_str());
        } else if (arg.find("--parallel_impl_cpu=") == 0) {
            std::string parallel_impl_cpu_text = arg.substr(20);
            if (parallel_impl_cpu_text == "openmp") {
                parallel_impl_cpu = OpenMP;
            }
        } else if (arg.find("--cuda_f_tile_size=") == 0) {
            cuda_f_tile_size = std::atoi(arg.substr(19).c_str());
        } else if (arg.find("--cuda_b_block_size=") == 0) {
            cuda_b_block_size = std::atoi(arg.substr(20).c_str());
        }
    }
}
