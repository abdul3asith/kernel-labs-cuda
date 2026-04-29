#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                      << " -> " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

__device__ void warp_reduce(volatile float* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_warp(const float* input, float* partial, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int start = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;
    if (start < n) sum = input[start];
    if (start + blockDim.x < n) sum += input[start + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warp_reduce(sdata, tid);
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

float cpu_sum(const std::vector<float>& data) {
    return std::accumulate(data.begin(), data.end(), 0.0f);
}

int main() {
    int n = 1 << 20;
    int block_size = 256;
    int grid_size = (n + (block_size * 2) - 1) / (block_size * 2);

    std::vector<float> h_input(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        h_input[i] = dist(rng);
    }

    std::vector<float> h_partial(grid_size);

    float* d_input = nullptr;
    float* d_partial = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_partial, grid_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    reduce_warp<<<grid_size, block_size, block_size * sizeof(float)>>>(d_input, d_partial, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_partial.data(), d_partial, grid_size * sizeof(float), cudaMemcpyDeviceToHost));

    float gpu_sum = cpu_sum(h_partial);
    float ref_sum = cpu_sum(h_input);

    std::cout << "CPU sum: " << ref_sum << std::endl;
    std::cout << "GPU sum: " << gpu_sum << std::endl;
    std::cout << "Absolute error: " << std::fabs(ref_sum - gpu_sum) << std::endl;

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_partial));

    return 0;
}