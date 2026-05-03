#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)              \
                      << " at line " << __LINE__ << std::endl; \
            exit(1);                                          \
        }                                                     \
    } while (0)

// ---------------- CPU ReLU ----------------
void relu_cpu(int n, const float* x, float* y) {
    for (int i = 0; i < n; i++) {
        y[i] = std::max(0.0f, x[i]);
    }
}

// ---------------- CPU GELU ----------------
void gelu_cpu(int n, const float* x, float* y) {
    const float c = 0.7978845608f; // sqrt(2 / pi)

    for (int i = 0; i < n; i++) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
}
// ReLU: hard cutoff at 0
// GELU: softly scales negative values
// GELU(x) = 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))

// ---------------- GPU ReLU ----------------
__global__ void relu_gpu(int n, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        y[i] = fmaxf(0.0f, x[i]);
    }
}

// ---------------- GPU GELU ----------------
__global__ void gelu_gpu(int n, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float v = x[i];
        float c = 0.7978845608f; // sqrt(2 / pi)

        y[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
    }
}

// ---------------- Correctness Check ----------------
bool check_correctness(const std::vector<float>& cpu,
                       const std::vector<float>& gpu,
                       float tolerance = 1e-4f) {
    for (size_t i = 0; i < cpu.size(); i++) {
        float diff = std::fabs(cpu[i] - gpu[i]);

        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i << std::endl;
            std::cout << "CPU: " << cpu[i] << ", GPU: " << gpu[i] << std::endl;
            std::cout << "Diff: " << diff << std::endl;
            return false;
        }
    }

    return true;
}

int main() {
    int n = 1 << 20; // 1,048,576 elements
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    std::cout << "N = " << n << std::endl;
    std::cout << "Block size = " << blockSize << std::endl;
    std::cout << "Grid size = " << gridSize << std::endl;

    // ---------------- Host vectors ----------------
    std::vector<float> h_x(n);
    std::vector<float> h_relu_cpu(n);
    std::vector<float> h_relu_gpu(n);
    std::vector<float> h_gelu_cpu(n);
    std::vector<float> h_gelu_gpu(n);

    // Initialize input values from negative to positive
    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<float>((i % 2000) - 1000) / 100.0f;
    }

    // ---------------- CPU computation ----------------
    relu_cpu(n, h_x.data(), h_relu_cpu.data());
    gelu_cpu(n, h_x.data(), h_gelu_cpu.data());

    // ---------------- Device pointers ----------------
    float *d_x, *d_relu_y, *d_gelu_y;

    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gelu_y, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // ---------------- Launch GPU ReLU ----------------
    relu_gpu<<<gridSize, blockSize>>>(n, d_x, d_relu_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------- Launch GPU GELU ----------------
    gelu_gpu<<<gridSize, blockSize>>>(n, d_x, d_gelu_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------- Copy results back ----------------
    CUDA_CHECK(cudaMemcpy(h_relu_gpu.data(), d_relu_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gelu_gpu.data(), d_gelu_y, n * sizeof(float), cudaMemcpyDeviceToHost));

    // ---------------- Check correctness ----------------
    bool relu_ok = check_correctness(h_relu_cpu, h_relu_gpu);
    bool gelu_ok = check_correctness(h_gelu_cpu, h_gelu_gpu, 1e-3f);

    std::cout << "\nCorrectness:" << std::endl;
    std::cout << "ReLU: " << (relu_ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "GELU: " << (gelu_ok ? "PASS" : "FAIL") << std::endl;

    // ---------------- Print sample outputs ----------------
    std::cout << "\nSample outputs:" << std::endl;

    for (int i = 0; i < 10; i++) {
        std::cout << "x[" << i << "] = " << h_x[i]
                  << " | ReLU = " << h_relu_gpu[i]
                  << " | GELU = " << h_gelu_gpu[i]
                  << std::endl;
    }
    std::cout << "\nSample around zero:" << std::endl;
for (int i = 995; i <= 1005; i++) {
    std::cout << "x[" << i << "] = " << h_x[i]
              << " | ReLU = " << h_relu_gpu[i]
              << " | GELU = " << h_gelu_gpu[i]
              << std::endl;
}

std::cout << "\nSample positive values:" << std::endl;
for (int i = 1500; i < 1505; i++) {
    std::cout << "x[" << i << "] = " << h_x[i]
              << " | ReLU = " << h_relu_gpu[i]
              << " | GELU = " << h_gelu_gpu[i]
              << std::endl;
}

    // ---------------- Free GPU memory ----------------
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_relu_y));
    CUDA_CHECK(cudaFree(d_gelu_y));

    return 0;
}