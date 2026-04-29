#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                      << " -> " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

__global__ void reduce_shared(const float* input, float* partial, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (global_idx < n) ? input[global_idx] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
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
    std::vector<int> sizes = {1 << 20, 1 << 22, 1 << 24};
    std::vector<int> block_sizes = {64, 128, 256, 512};

    std::ofstream csv("shared_results.csv");
    csv << "kernel,n,block_size,grid_size,kernel_ms,throughput_melems_per_sec,abs_error,rel_error\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int n : sizes) {
        std::vector<float> h_input(n);
        for (int i = 0; i < n; i++) h_input[i] = dist(rng);

        float ref_sum = cpu_sum(h_input);

        for (int block_size : block_sizes) {
            int grid_size = (n + block_size - 1) / block_size;

            float* d_input = nullptr;
            float* d_partial = nullptr;

            CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_partial, grid_size * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));

            std::vector<float> h_partial(grid_size);

            reduce_shared<<<grid_size, block_size, block_size * sizeof(float)>>>(d_input, d_partial, n);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));

            float total_ms = 0.0f;
            int iters = 20;

            for (int i = 0; i < iters; i++) {
                CHECK_CUDA(cudaEventRecord(start));
                reduce_shared<<<grid_size, block_size, block_size * sizeof(float)>>>(d_input, d_partial, n);
                CHECK_CUDA(cudaEventRecord(stop));
                CHECK_CUDA(cudaEventSynchronize(stop));
                CHECK_CUDA(cudaGetLastError());

                float ms = 0.0f;
                CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
                total_ms += ms;
            }

            float kernel_ms = total_ms / iters;

            CHECK_CUDA(cudaMemcpy(h_partial.data(), d_partial, grid_size * sizeof(float), cudaMemcpyDeviceToHost));

            float gpu_sum = cpu_sum(h_partial);
            float abs_error = std::fabs(ref_sum - gpu_sum);
            float rel_error = abs_error / (std::fabs(ref_sum) + 1e-12f);
            double throughput = ((double)n / (kernel_ms / 1000.0)) / 1e6;

            csv << "shared,"
                << n << ","
                << block_size << ","
                << grid_size << ","
                << std::fixed << std::setprecision(6) << kernel_ms << ","
                << std::fixed << std::setprecision(3) << throughput << ","
                << std::scientific << abs_error << ","
                << std::scientific << rel_error << "\n";

            std::cout << "shared | n=" << n
                      << " | block=" << block_size
                      << " | kernel_ms=" << std::fixed << std::setprecision(6) << kernel_ms
                      << " | throughput=" << std::fixed << std::setprecision(3) << throughput
                      << " M elems/s | abs_error=" << std::scientific << abs_error
                      << std::endl;

            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
            CHECK_CUDA(cudaFree(d_input));
            CHECK_CUDA(cudaFree(d_partial));
        }
    }

    csv.close();
    std::cout << "Saved shared_results.csv\n";
    return 0;
}