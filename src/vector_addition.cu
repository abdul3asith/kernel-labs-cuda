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

int main() {
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 4.0f);
    std::vector<float> h_c(n, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    checkCuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    checkCuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    checkCuda(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    checkCuda(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice), "copy h_a -> d_a");
    checkCuda(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice), "copy h_b -> d_b");

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    CudaTimer timer;
    timer.start();

    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel sync");

    float kernelMs = timer.stop();

    checkCuda(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "copy d_c -> h_c");

    bool correct = true;
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != 3.0f) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": got " << h_c[i] << ", expected 3.0\n";
            break;
        }
    }

    std::cout << "n: " << n << "\n";
    std::cout << "bytes: " << bytes << "\n";
    std::cout << "blockSize: " << blockSize << "\n";
    std::cout << "gridSize: " << gridSize << "\n";
    std::cout << "Vector addition " << (correct ? "PASSED" : "FAILED") << "\n";
    std::cout << "Kernel time: " << kernelMs << " ms\n";
    std::cout << "Sample output: " << h_c[0] << ", " << h_c[1] << ", " << h_c[2] << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return correct ? 0 : 1;
}