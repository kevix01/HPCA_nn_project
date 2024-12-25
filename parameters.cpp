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

int Parameters::getFSamplesNumThreads() const {
    return f_samples_num_threads;
}

int Parameters::getBInNeuronsNumThreads() const {
    return b_in_neurons_num_threads;
}

int Parameters::getBOutNeuronsNumThreads() const {
    return b_out_neurons_num_threads;
}

int Parameters::getFOutNeuronsNumThreads() const {
    return f_out_neurons_num_threads;
}

void Parameters::parseArguments(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--f_samples_num_threads=") == 0) {
            f_samples_num_threads = std::atoi(arg.substr(24).c_str());
        } else if (arg.find("--parallel_impl_cpu=") == 0) {
            std::string parallel_impl_cpu_text = arg.substr(20);
            if (parallel_impl_cpu_text == "openmp") {
                parallel_impl_cpu = OpenMP;
            }
        } else if (arg.find("--f_out_neurons_num_threads=") == 0) {
            f_out_neurons_num_threads = std::atoi(arg.substr(28).c_str());
        } else if (arg.find("--b_out_neurons_num_threads=") == 0) {
            b_out_neurons_num_threads = std::atoi(arg.substr(28).c_str());
        } else if (arg.find("--b_in_neurons_num_threads=") == 0) {
            b_in_neurons_num_threads = std::atoi(arg.substr(27).c_str());
        }
    }
}
