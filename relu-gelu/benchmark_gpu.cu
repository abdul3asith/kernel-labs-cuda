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

// ---------------- Benchmark Helper ----------------
float benchmark_relu(int n, int gridSize, int blockSize, const float* d_x, float* d_y, int runs) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 5; i++) {
        relu_gpu<<<gridSize, blockSize>>>(n, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < runs; i++) {
        relu_gpu<<<gridSize, blockSize>>>(n, d_x, d_y);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / runs;
}

float benchmark_gelu(int n, int gridSize, int blockSize, const float* d_x, float* d_y, int runs) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 5; i++) {
        gelu_gpu<<<gridSize, blockSize>>>(n, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < runs; i++) {
        gelu_gpu<<<gridSize, blockSize>>>(n, d_x, d_y);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_ms / runs;
}

int main() {
    int n = 1 << 20; // 1,048,576 elements
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    int runs = 100;

    std::cout << "N = " << n << std::endl;
    std::cout << "Block size = " << blockSize << std::endl;
    std::cout << "Grid size = " << gridSize << std::endl;
    std::cout << "Benchmark runs = " << runs << std::endl;

    // ---------------- Host vectors ----------------
    std::vector<float> h_x(n);
    std::vector<float> h_relu_cpu(n);
    std::vector<float> h_relu_gpu(n);
    std::vector<float> h_gelu_cpu(n);
    std::vector<float> h_gelu_gpu(n);

    // Values from -10.00 to +9.99
    for (int i = 0; i < n; i++) {
        h_x[i] = static_cast<float>((i % 2000) - 1000) / 100.0f;
    }

    // ---------------- CPU reference ----------------
    relu_cpu(n, h_x.data(), h_relu_cpu.data());
    gelu_cpu(n, h_x.data(), h_gelu_cpu.data());

    // ---------------- Device pointers ----------------
    float *d_x, *d_relu_y, *d_gelu_y;

    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_relu_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gelu_y, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // ---------------- Correctness launches ----------------
    relu_gpu<<<gridSize, blockSize>>>(n, d_x, d_relu_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    gelu_gpu<<<gridSize, blockSize>>>(n, d_x, d_gelu_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_relu_gpu.data(), d_relu_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gelu_gpu.data(), d_gelu_y, n * sizeof(float), cudaMemcpyDeviceToHost));

    bool relu_ok = check_correctness(h_relu_cpu, h_relu_gpu);
    bool gelu_ok = check_correctness(h_gelu_cpu, h_gelu_gpu, 1e-3f);

    std::cout << "\nCorrectness:" << std::endl;
    std::cout << "ReLU: " << (relu_ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "GELU: " << (gelu_ok ? "PASS" : "FAIL") << std::endl;

    // ---------------- Benchmark ----------------
    float relu_ms = benchmark_relu(n, gridSize, blockSize, d_x, d_relu_y, runs);
    float gelu_ms = benchmark_gelu(n, gridSize, blockSize, d_x, d_gelu_y, runs);

    // ReLU/GELU both read x and write y = 2 memory operations
    double bytes = static_cast<double>(n) * sizeof(float) * 2.0;

    double relu_bandwidth = bytes / (relu_ms / 1000.0) / 1e9;
    double gelu_bandwidth = bytes / (gelu_ms / 1000.0) / 1e9;

    std::cout << "\nBenchmark Results:" << std::endl;
    std::cout << "ReLU avg kernel time: " << relu_ms << " ms" << std::endl;
    std::cout << "GELU avg kernel time: " << gelu_ms << " ms" << std::endl;
    std::cout << "ReLU bandwidth: " << relu_bandwidth << " GB/s" << std::endl;
    std::cout << "GELU bandwidth: " << gelu_bandwidth << " GB/s" << std::endl;

    // ---------------- Sample values ----------------
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

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_relu_y));
    CUDA_CHECK(cudaFree(d_gelu_y));

    return 0;
}