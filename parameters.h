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
    int getOpenmpThreads() const;
    int getCudaFTileSize() const;
    int getCudaBBlockSize() const;

private:
    // int num_threads;
    ParallelImplCpu parallel_impl_cpu = No;
    int openmp_threads = 6;
    int cuda_f_tile_size = 16;
    int cuda_b_block_size = 512;

    void parseArguments(int argc, char* argv[]);
};

#endif // PARAMETERS_H

