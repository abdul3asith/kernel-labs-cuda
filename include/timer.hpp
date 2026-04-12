/** timer.hpp - A cpp header file for timing utilities */

/**making a simple timing structure we'll reuse later for real kernels like vector add, reduction, and matmul. - A benchmark helper */

#ifndef TIMER_HPP
#define TIMER_HPP

#include <cuda_runtime.h>
#include <iostream>

class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        cudaEventRecord(start_);
    }

    float stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

inline void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << msg << " failed: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

#endif