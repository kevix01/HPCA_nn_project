//
// Created by kevin on 18/12/24.
//

#ifndef PARAMETERS_H
#define PARAMETERS_H


#include "parallel_impl_cpu.h"

class Parameters {
public:
    Parameters(int argc, char* argv[]);
    int getNumThreads() const;
    ParallelImplCpu getParallelImplCpu() const;
    int getOutNeuronsNumThreads() const;
    int getInNeuronsNumThreads() const;

private:
    int num_threads;
    ParallelImplCpu parallel_impl_cpu;
    int out_neurons_num_threads;
    int in_neurons_num_threads;

    void parseArguments(int argc, char* argv[]);
};

#endif // PARAMETERS_H

