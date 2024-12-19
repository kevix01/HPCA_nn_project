//
// Created by kevin on 18/12/24.
//

#include "parameters.h"
#include <cstdlib> // For std::atoi
#include <algorithm>
#include <iostream>

Parameters::Parameters(int argc, char* argv[]) : num_threads(1), parallel_impl_cpu(No) {
    parseArguments(argc, argv);
}

int Parameters::getNumThreads() const {
    return num_threads;
}

ParallelImplCpu Parameters::getParallelImplCpu() const {
    return parallel_impl_cpu;
}

int Parameters::getInNeuronsNumThreads() const {
    return in_neurons_num_threads;
}

int Parameters::getOutNeuronsNumThreads() const {
    return out_neurons_num_threads;
}

void Parameters::parseArguments(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--num_threads=") == 0) {
            num_threads = std::atoi(arg.substr(14).c_str());
        } else if (arg.find("--parallel_impl_cpu=") == 0) {
            std::string parallel_impl_cpu_text = arg.substr(20);
            if (parallel_impl_cpu_text == "openmp") {
                parallel_impl_cpu = OpenMP;
            }
        } else if (arg.find("--out_neurons_num_threads=") == 0) {
            out_neurons_num_threads = std::atoi(arg.substr(26).c_str());
        } else if (arg.find("--in_neurons_num_threads=") == 0) {
            in_neurons_num_threads = std::atoi(arg.substr(25).c_str());
        }
    }
}
