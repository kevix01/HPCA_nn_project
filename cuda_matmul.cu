//
// Created by kevin on 08/12/24.
//

#include "cuda_matmul.h"
#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 16

__global__ void matMulKernel(float *a, float *b, float *ab, int M, int K, int N) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    __shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float result = 0;

    for (int p = 0; p < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
        s_a[ty][tx] = (row < M && p * TILE_WIDTH + tx < K) ? a[row * K + p * TILE_WIDTH + tx] : 0.0;
        s_b[ty][tx] = (p * TILE_WIDTH + ty < K && col < N) ? b[(p * TILE_WIDTH + ty) * N + col] : 0.0;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            result += s_a[ty][k] * s_b[k][tx];
        __syncthreads();
    }

    if (row < M && col < N) {
        ab[row * N + col] = result;
        // Debug print to log the result
        printf("CUDA - Kernel result at row %d, col %d: %f\n", row, col, result);
    }
}

void matMul(float *a, float *b, float *ab, int M, int K, int N) {
    float *d_a, *d_b, *d_ab;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeAB = M * N * sizeof(float);

    cudaMalloc(&d_a, sizeA);
    cudaMalloc(&d_b, sizeB);
    cudaMalloc(&d_ab, sizeAB);

    cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    matMulKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_ab, M, K, N);

    cudaMemcpy(ab, d_ab, sizeAB, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ab);
}



