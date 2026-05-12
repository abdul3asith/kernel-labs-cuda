
/*
Layer Normalization
For a single input vector x (say, the activations of one token in a transformer at one position), Layer Norm does four steps:

Compute the mean of the vector's elements: μ = average(x)
Compute the variance of the vector's elements: σ² = average((x - μ)²)
Normalize: x̂ = (x - μ) / √(σ² + ε)
Scale and shift with learnable parameters: y = γ * x̂ + β

y_i = (gamma)γ_i * (x_i - μ) / √(σ² + ε) + β_i
*/

// layernorm_naive.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__
void layer_norm_kernel(
    const float* __restrict__ x,
    float*       __restrict__ y,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int   N,
    int   D,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const float* x_row = x + row * D;
          float* y_row = y + row * D;

    extern __shared__ float sdata[];

    // Pass 1: mean
    float partial_sum = 0.0f;
    for (int i = tid; i < D; i += block_size) {
        partial_sum += x_row[i];
    }

    sdata[tid] = partial_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float mean = sdata[0] / D;

    // Pass 2: variance
    float partial_sq = 0.0f;
    for (int i = tid; i < D; i += block_size) {
        float diff = x_row[i] - mean;
        partial_sq += diff * diff;
    }

    sdata[tid] = partial_sq;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    float variance = sdata[0] / D;
    float rstd = rsqrtf(variance + eps);

    // Pass 3: write normalized output
    for (int i = tid; i < D; i += block_size) {
        float normalized = (x_row[i] - mean) * rstd;
        y_row[i] = gamma[i] * normalized + beta[i];
    }
}

int main() {
    const int N = 1024;
    const int D = 4096;
    const float eps = 1e-5f;

    size_t x_bytes = (size_t)N * D * sizeof(float);
    size_t d_bytes = D * sizeof(float);

    float* h_x     = new float[N * D];
    float* h_y     = new float[N * D];
    float* h_gamma = new float[D];
    float* h_beta  = new float[D];

    for (int i = 0; i < N * D; i++) h_x[i] = (float)((i * 7 + 3) % 17) - 8.0f;
    for (int i = 0; i < D; i++)     h_gamma[i] = 1.0f;
    for (int i = 0; i < D; i++)     h_beta[i]  = 0.0f;

    float *d_x, *d_y, *d_gamma, *d_beta;
    cudaMalloc(&d_x,     x_bytes);
    cudaMalloc(&d_y,     x_bytes);
    cudaMalloc(&d_gamma, d_bytes);
    cudaMalloc(&d_beta,  d_bytes);

    cudaMemcpy(d_x,     h_x,     x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, d_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,  h_beta,  d_bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    dim3 grid(N);
    dim3 block(block_size);
    size_t shmem_bytes = block_size * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    layer_norm_kernel<<<grid, block, shmem_bytes>>>(d_x, d_y, d_gamma, d_beta, N, D, eps);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    layer_norm_kernel<<<grid, block, shmem_bytes>>>(d_x, d_y, d_gamma, d_beta, N, D, eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_y, d_y, x_bytes, cudaMemcpyDeviceToHost);

    float bytes_moved = 2.0f * N * D * sizeof(float) + 2.0f * D * sizeof(float);
    float bw_gb_s = bytes_moved / (ms / 1000.0f) / 1e9f;

    std::cout << "LayerNorm: N=" << N << " D=" << D << "\n";
    std::cout << "Time:      " << ms      << " ms\n";
    std::cout << "Bandwidth: " << bw_gb_s << " GB/s\n";

    // CPU reference check on row 0
    double sum = 0.0;
    for (int i = 0; i < D; i++) sum += h_x[i];
    double ref_mean = sum / D;

    double sq = 0.0;
    for (int i = 0; i < D; i++) {
        double d = h_x[i] - ref_mean;
        sq += d * d;
    }
    double ref_var  = sq / D;
    double ref_rstd = 1.0 / std::sqrt(ref_var + eps);

    float max_err = 0.0f;
    for (int i = 0; i < D; i++) {
        double ref = (h_x[i] - ref_mean) * ref_rstd;
        float diff = std::fabs((float)ref - h_y[i]);
        if (diff > max_err) max_err = diff;
    }
    std::cout << "Max error on row 0 vs CPU reference: " << max_err << "\n";

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma); cudaFree(d_beta);
    delete[] h_x; delete[] h_y; delete[] h_gamma; delete[] h_beta;
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}