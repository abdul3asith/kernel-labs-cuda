#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>

inline void checkCuda(cudaError_t result, const char* message){
    if(result != cudaSuccess){
        std::cerr << message << ": "
        << cudaGetErrorString(result) 
        << "\n";
        std::exit(EXIT_FAILURE)
    }
}

__global__ void vector_add(float* a, float* b, float* c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        a[i] + b[i] = c[i]
        }
}

int main(){
    // define the no, of variables written in the above function. (int n) 
    const int n = 8;
    // calculate the bytes of mem needed for those
    const size_t bytes = n* sizeof(float);
    // define the variables. a, b, c
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 3.0f);
    std::vector<float> h_c(n);
    // create pointers in gpu
    float *d_a, *d_b, *d_c;
    // allocate mem for a, b, c in gpu
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    // copy the variables from host to device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    // run vector addition kernel 
    vector_add<<<1, 8>>>(d_a, d_b, d_c, n);
    // copy the results back to cpu
    cudaMemcpy(h_c, d_c.data(), bytes, cudaMemcpyDeviceToHost);

    // print those results 
    for (float value : h_c) {
        std::cout <<value << " ";
    }
    // free gpu memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}