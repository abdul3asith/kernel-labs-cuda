// it is used to turn model logits, class scores and attention scores to probabilities
// it is useful because of finding max, reduction/sum, exponential and normalization.
// for this we need naive --> stable --> parallel reduction --> tiled/block --> fused softmax. 
// stable softmax --> softmax(x_i) = e^(x_i - m)/sum (e^x_j - m) where m = max(x)

// e^1 = 2.72


#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>

inline void checkCuda(cudaError_t result, const char* message) {
    if (result != cudaSuccess) {
        std::cerr << message << ": "
                  << cudaGetErrorString(result)
                  << '\n';
        std::exit(EXIT_FAILURE);
    }
}

// there are 3 kernels 1. exponentiate kernel 2. addition kernel and 3. normalize kernel

// 1. exponentiate kernel

__global__ void exponentiateKernel(
    const float* input,
    float* exponentials,
    int n
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        exponentials[i] = expf(input[i]);
    }
}

// 2. sum kernel - atomicAdd to prevent threads from incorrectly updating sums at a time

__global__ void sumKernel(
    const float* exponentials,
    float* total,
    int n
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        atomicAdd(total, exponentials[i]);
    }
}

// 3. norm kernal


__global__ void normalizeKernel(
    const float* exponentials,
    const float* total,
    float* output,
    int n
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i < n ){
        output[i] = exponentials[i] / total[0];
    }
}

int main() {
    const std::vector<float> h_input = {
        1.0f,
        2.0f,
        3.0f,
        6.0f
    };

    const int n = static_cast<int>(h_input.size());
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_output(n);

    float* d_input = nullptr;
    float* d_exponentials = nullptr;
    float* d_output = nullptr;
    float* d_total = nullptr;

    checkCuda(
        cudaMalloc(&d_input, bytes),
        "cudaMalloc d_input failed"
    );

    checkCuda(
        cudaMalloc(&d_exponentials, bytes),
        "cudaMalloc d_exponentials failed"
    );

    checkCuda(
        cudaMalloc(&d_output, bytes),
        "cudaMalloc d_output failed"
    );

    checkCuda(
        cudaMalloc(&d_total, sizeof(float)),
        "cudaMalloc d_total failed"
    );

    checkCuda(
        cudaMemcpy(
            d_input,
            h_input.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "Input copy failed"
    );

    // The sum must begin at zero.
    checkCuda(
        cudaMemset(d_total, 0, sizeof(float)),
        "cudaMemset d_total failed"
    );

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    exponentiateKernel<<<gridSize, blockSize>>>(
        d_input,
        d_exponentials,
        n
    );

    checkCuda(
        cudaGetLastError(),
        "Exponentiate kernel launch failed"
    );

    sumKernel<<<gridSize, blockSize>>>(
        d_exponentials,
        d_total,
        n
    );

    checkCuda(
        cudaGetLastError(),
        "Sum kernel launch failed"
    );

    normalizeKernel<<<gridSize, blockSize>>>(
        d_exponentials,
        d_total,
        d_output,
        n
    );

    checkCuda(
        cudaGetLastError(),
        "Normalize kernel launch failed"
    );

    checkCuda(
        cudaDeviceSynchronize(),
        "Kernel execution failed"
    );

    checkCuda(
        cudaMemcpy(
            h_output.data(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "Output copy failed"
    );

    float probabilitySum = 0.0f;

    std::cout << "Input:   ";
    for (float value : h_input) {
        std::cout << value << " ";
    }

    std::cout << "\nSoftmax: ";

    for (float probability : h_output) {
        std::cout << probability << " ";
        probabilitySum += probability;
    }

    std::cout << "\nProbability sum: "
              << probabilitySum
              << '\n';

    cudaFree(d_input);
    cudaFree(d_exponentials);
    cudaFree(d_output);
    cudaFree(d_total);

    return 0;
}
