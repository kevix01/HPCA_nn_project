//
// Created by kevin on 08/12/24.
//

#ifndef CUDA_MATMUL_H
#define CUDA_MATMUL_H

#include <actf_type_cuda.h>

extern std::chrono::duration<double> elapsed_f_kernel;

#define CHECK_CUDA_ERROR(call, cleanup) do {                                     \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess) {                                                    \
        std::cerr << "CUDA Error in Forward: " << cudaGetErrorString(err)       \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
        cleanup;                                                                 \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
} while(0)

#ifdef __CUDACC__
#define CUDA_FUNC_DECL __host__ __device__
#else
#define CUDA_FUNC_DECL
#endif

CUDA_FUNC_DECL float relu(float x);
CUDA_FUNC_DECL float sigmoid(float x);

#ifdef __CUDACC__
__global__ void matMulKernel(float *a, float *b, float *ab, int N, int K, int M, ActivationFunctionType act_type, int TILE_WIDTH);
#endif

void forwardMatMul(float *a, float *b, float *ab, int M, int K, int N, ActivationFunctionType act_type, int TILE_WIDTH);

#endif // CUDA_MATMUL_H

