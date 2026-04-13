#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "timer.hpp"

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int runBenchmark(int n, int blockSize) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const int gridSize = (n + blockSize - 1) / blockSize;
    const int repeats = 10;

    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    checkCuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    checkCuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    checkCuda(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    CudaTimer timer;

    timer.start();
    checkCuda(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice), "copy h_a -> d_a");
    checkCuda(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice), "copy h_b -> d_b");
    float h2d_ms = timer.stop();

    float kernel_ms_total = 0.0f;
    for (int r = 0; r < repeats; ++r) {
        timer.start();
        vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(), "kernel sync");
        kernel_ms_total += timer.stop();
    }
    float kernel_ms_avg = kernel_ms_total / repeats;

    timer.start();
    checkCuda(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "copy d_c -> h_c");
    float d2h_ms = timer.stop();

    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != 3.0f) {
            correct = false;
            break;
        }
    }

    std::cout << n << ","
              << bytes << ","
              << blockSize << ","
              << gridSize << ","
              << h2d_ms << ","
              << kernel_ms_avg << ","
              << d2h_ms << ","
              << repeats << ","
              << (correct ? 1 : 0) << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return correct ? 0 : 1;
}

int main() {
    std::vector<int> n_values = {1 << 10, 1 << 30, 1 << 20, 1 << 24, 1 << 45, 1 << 60};
    std::vector<int> block_values = {64, 128, 256, 512, 1024};

    std::cout << "n,bytes,blockSize,gridSize,h2d_ms,kernel_ms_avg,d2h_ms,repeats,passed\n";

    for (int n : n_values) {
        for (int blockSize : block_values) {
            runBenchmark(n, blockSize);
        }
    }

    return 0;
}