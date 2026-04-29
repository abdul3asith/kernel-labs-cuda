#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void saxpy_gpu(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int n = 1 << 20;
    float a = 2.0f;

    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(n, 2.0f);

    for (int i = 0; i < n; i++) {
        h_x[i] = i + 1;
        h_y[i] = 10.0f;
    }

    float* d_x = nullptr;
    float* d_y = nullptr;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    saxpy_gpu<<<numBlocks, blockSize>>>(n, a, d_x, d_y);
    cudaDeviceSynchronize();
    

    cudaMemcpy(h_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        std::cout << "y[" << i << "] = " << h_y[i] << std::endl;
    }

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}