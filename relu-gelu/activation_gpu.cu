/*CPU ReLU
GPU ReLU
CPU GELU
GPU GELU
Correctness check
Timing with CUDA events
Benchmark different N values*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <algorithm>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                      << " -> " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// CPU ReLU
void relu_cpu(int n, float* x, float* y) {
    for (int i = 0; i < n; i++) {
        y[i] = std::max(0.0f, x[i]);
    }
}

// CPU GELU
void gelu_cpu(int n, float* x, float* y) {
    const float c = 0.7978845608f; // sqrt(2 / pi)

    for (int i = 0; i < n; i++) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
}

// GPU ReLU

__global__ void relu_gpu(int n, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        y[i] = fmaxf(0.0f, x[i]);
    }
}

// GPU GELU

__global__ void gelu_gpu(int n, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float v = x[i];
        float c = 0.7978845608f;
        y[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
    }
}

// Correctness check 

bool check_correctness()