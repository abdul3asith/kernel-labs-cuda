#include <iostream>
#include <cuda_runtime.h>
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


__global__ void gmemm(float* input, float* output, int n) {
    
}