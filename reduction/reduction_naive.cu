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

__global__ void reduce_naive(const float* input, float* partial, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (global_idx < n) ? input[global_idx] : 0.0f;
    __syncthreads();

    // Naive interleaved reduction
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0 && (tid + stride) < blockDim.x) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

float cpu_sum(const std::vector<float>& data) {
    return std::accumulate(data.begin(), data.end(), 0.0f);
}

int main() {
    int n = 50;          // 50 elements
    int block_size = 64;
    int grid_size = (n + block_size - 1) / block_size;

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

    reduce_naive<<<grid_size, block_size, block_size * sizeof(float)>>>(d_input, d_partial, n);
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