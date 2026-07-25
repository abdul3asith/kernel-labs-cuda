#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


//N = rows = how many tokens.
//d = columns = vector size per token.
//float* — a pointer to floats.
//const — the data it points to can't be modified through this pointer. Lets the compiler cache values and catches typos at compile time.
//__restrict__ — a promise that no other pointer in this function points to the same memory. Lets the compiler avoid re-reading values "just in case" something else wrote to them.

__global__ void example(const float* Q, const float* K, const float* S, int N, int d){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N || j>= N) return;

    float acc = 0.0f;
    for (int k = 0; k < d; ++k){
        acc+= Q[i*d+k] * K[j*d+k];
    }
    s[i*N+j] = acc;
}