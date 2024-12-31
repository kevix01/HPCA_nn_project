//
// Created by kevin on 18/12/24.
//

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "device_type.h"
#include "parallel_impl_cpu.h"

class Parameters {
public:
    Parameters(int argc, char* argv[]);
    ParallelImplCpu getParallelImplCpu() const;
    int getOpenmpThreads() const;
    int getCudaFTileSize() const;
    int getCudaBBlockSize() const;
    DeviceType getDevice() const;
    int getTrainBatchSize() const;
    int getPredictBatchSize() const;
    int getTrainEpochs() const;
    float getLearningRate() const;
    int getNeuronsFirstHiddenLayer() const;
    int getNeuronsSecondHiddenLayer() const;
    int getWeightsInitSeed() const;
    int getTraindataShuffleSeed() const;

private:
    int neurons_first_hidden_layer = 50;
    int neurons_second_hidden_layer = 10;
    DeviceType device = NONE;
    ParallelImplCpu parallel_impl_cpu = No;
    int openmp_threads = 1;
    int cuda_f_tile_size = 16;
    int cuda_b_block_size = 256;
    int train_batch_size = 50;
    int predict_batch_size = 50;
    int train_epochs = 20;
    float learning_rate = 0.01;
    int weights_init_seed = 0;
    int traindata_shuffle_seed = 0;

    void parseArguments(int argc, char* argv[]);
};

#endif // PARAMETERS_H

