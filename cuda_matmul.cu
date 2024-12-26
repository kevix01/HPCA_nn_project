//
// Created by kevin on 08/12/24.
//

#include <iostream>
#include <cuda_matmul.h>
#include <cuda_runtime.h>

typedef float (*activationFunc)(float);

__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float activate(float x, ActivationFunctionType act_type) {
    if (act_type == RELU) {
        return relu(x);
    } else if (act_type == SIGMOID) {
        return sigmoid(x);
    }
    return x;
}


#define TILE_WIDTH 16

__global__ void matMulKernel(float *a, float *b, float *ab, int N, int K, int M, ActivationFunctionType act_type) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    __shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float result = 0;

    for (int p = 0; p < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
        s_a[ty][tx] = (row < N && p * TILE_WIDTH + tx < K) ? a[row * K + p * TILE_WIDTH + tx] : 0.0;
        s_b[ty][tx] = (p * TILE_WIDTH + ty < K && col < M) ? b[(p * TILE_WIDTH + ty) * M + col] : 0.0;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            result += s_a[ty][k] * s_b[k][tx];
        __syncthreads();
    }

    if (row < N && col < M) {
        // ab[row * M + col] = ab[row * M + col] + result;
        //printf("Value before activation: %f\n", ab[row * M + col]);
        ab[row * M + col] = activate(ab[row * M + col] + result, act_type);
        //printf("Value after activation: %f\n", ab[row * M + col]);
    }
}

void matMul(float *a, float *b, float *ab, int M, int K, int N, ActivationFunctionType act_type) {
    float *d_a, *d_b, *d_ab;
    size_t sizeA = N * K * sizeof(float);
    size_t sizeB = K * M * sizeof(float);
    size_t sizeAB = N * M * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_a, sizeA);
    cudaMalloc(&d_b, sizeB);
    cudaMalloc(&d_ab, sizeAB);

    // Copy data from host to device
    cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ab, ab, sizeAB, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((M + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Select the activation function
    //activationFunc activate_dev;
    //cudaMemcpyFromSymbol(&activate_dev, activate, sizeof(activationFunc));

    // Launch the kernel
    matMulKernel<<<gridDim, blockDim>>>(d_a, d_b, d_ab, N, K, M, act_type);

    // Copy results from device to host
    cudaMemcpy(ab, d_ab, sizeAB, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ab);
}





