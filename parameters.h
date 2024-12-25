//
// Created by kevin on 18/12/24.
//

#ifndef PARAMETERS_H
#define PARAMETERS_H


#include "parallel_impl_cpu.h"

class Parameters {
public:
    Parameters(int argc, char* argv[]);
    // int getNumThreads() const;
    ParallelImplCpu getParallelImplCpu() const;
    int getFOutNeuronsNumThreads() const;
    int getFSamplesNumThreads() const;
    int getBOutNeuronsNumThreads() const;
    int getBInNeuronsNumThreads() const;

private:
    // int num_threads;
    ParallelImplCpu parallel_impl_cpu = No;
    int f_samples_num_threads = 1;
    int f_out_neurons_num_threads = 1;
    int b_out_neurons_num_threads = 1;
    int b_in_neurons_num_threads = 1;

    void parseArguments(int argc, char* argv[]);
};

#endif // PARAMETERS_H

