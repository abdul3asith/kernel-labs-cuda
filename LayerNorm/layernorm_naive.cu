#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Cuda error at" << __FILE__ << ":" << __LINE__   \
            << " - " << cudaGetErrorString(err) << std::endl; 
            std::exit(1)
        }
    } while (0)

/*
Layer Normalization
For a single input vector x (say, the activations of one token in a transformer at one position), Layer Norm does four steps:

Compute the mean of the vector's elements: μ = average(x)
Compute the variance of the vector's elements: σ² = average((x - μ)²)
Normalize: x̂ = (x - μ) / √(σ² + ε)
Scale and shift with learnable parameters: y = γ * x̂ + β

y_i = (gamma)γ_i * (x_i - μ) / √(σ² + ε) + β_i
*/

__global__ void layer_norm_kernel(
    const float* __restrict__ x, 
    float* __restrict__ y, 
    const float* __restrict__ gamma, 
    const float* __restrict__ beta, 
    int N, //number of rows
    int D, //hidden_dim (features per row)
    flaot eps) //numerical stability
    {
        //one bock handles one row. 
        int row = blockIdx.x;
        int tid = threadIdx.x;
        int block_size = blockDim.x;

        const float* x_row = x + row * D;
        const float* y_row = y + row * D;

        extern __shared__ float sdata[];

        float partial_sum = 0.0f;
        for (int i = tid; i<D; i += block_size) {
            partial_sum += x_row[i];
        }

        sdata[tid] = partial_sum;
        __syncthreads();

        //tree reduction. 
        for(int stride = block_size / 2; stride > 0; stride /= 2){
            if(tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }
}
    