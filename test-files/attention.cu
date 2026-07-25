// attention.cu
// Basic scaled dot-product attention in CUDA.
//
// Computes: O = softmax(Q @ K^T / sqrt(d_k)) @ V
//
// Shapes:
//   Q, K, V : [N, d]   (N tokens, d = head dim)
//   S       : [N, N]   (attention scores)
//   P       : [N, N]   (attention probabilities after softmax)
//   O       : [N, d]   (output)
//
// This is the straightforward 3-kernel version (matmul, softmax, matmul).
// It's meant to be readable, not as fast as FlashAttention. Each kernel
// uses shared-memory tiling where it matters.
//
// Build:   nvcc -O3 -arch=sm_80 attention.cu -o attention
// Run:     ./attention

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

// -----------------------------------------------------------------------------
// Kernel 1: tiled matmul.  C[M, N] = A[M, K] @ B[K, N], optionally scaled.
// Used twice: once for Q @ K^T (with B = K^T), once for P @ V.
// We pass a `transB` flag so we don't need a separate transpose kernel.
// -----------------------------------------------------------------------------
constexpr int TILE = 32;

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              float scale,
                              bool transB)
{
    // Each block computes a TILE x TILE tile of C.
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    // Walk along the K dimension one tile at a time.
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // For transB, B is treated as [N, K] and we read B[col, b_row].
        if (transB) {
            Bs[threadIdx.y][threadIdx.x] =
                (col < N && b_row < K) ? B[col * K + b_row] : 0.0f;
        } else {
            Bs[threadIdx.y][threadIdx.x] =
                (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc * scale;
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: row-wise softmax with the standard max-subtraction trick.
// One block per row; threads in the block cooperate via shared memory
// to compute row max and row sum.
// -----------------------------------------------------------------------------
__global__ void softmax_kernel(float* __restrict__ S, int N)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    extern __shared__ float sdata[];

    float* row_ptr = S + row * N;

    // ---- Pass 1: row max ----
    float local_max = -INFINITY;
    for (int j = tid; j < N; j += bsz) {
        local_max = fmaxf(local_max, row_ptr[j]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int stride = bsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = sdata[0];

    // ---- Pass 2: exponentiate and accumulate denominator ----
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += bsz) {
        float e = expf(row_ptr[j] - row_max);
        row_ptr[j] = e;            // store numerator in place
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = bsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = sdata[0];
    float inv_sum = 1.0f / row_sum;

    // ---- Pass 3: normalize ----
    for (int j = tid; j < N; j += bsz) {
        row_ptr[j] *= inv_sum;
    }
}

// -----------------------------------------------------------------------------
// Host wrapper: full attention forward.
// -----------------------------------------------------------------------------
void attention_forward(const float* dQ, const float* dK, const float* dV,
                       float* dO, float* dS,
                       int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);

    // S = Q @ K^T * scale       -> [N, N]
    {
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
        matmul_kernel<<<grid, block>>>(dQ, dK, dS, N, N, d, scale, /*transB=*/true);
        CUDA_CHECK(cudaGetLastError());
    }

    // P = softmax(S)            -> in-place in dS
    {
        int threads = 256;
        size_t shmem = threads * sizeof(float);
        softmax_kernel<<<N, threads, shmem>>>(dS, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // O = P @ V                 -> [N, d]
    {
        dim3 block(TILE, TILE);
        dim3 grid((d + TILE - 1) / TILE, (N + TILE - 1) / TILE);
        matmul_kernel<<<grid, block>>>(dS, dV, dO, N, d, N, 1.0f, /*transB=*/false);
        CUDA_CHECK(cudaGetLastError());
    }
}

// -----------------------------------------------------------------------------
// CPU reference for correctness check.
// -----------------------------------------------------------------------------
void attention_cpu(const std::vector<float>& Q,
                   const std::vector<float>& K,
                   const std::vector<float>& V,
                   std::vector<float>& O,
                   int N, int d)
{
    float scale = 1.0f / std::sqrt((float)d);
    std::vector<float> S(N * N);

    // S = Q K^T * scale
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < d; ++k) s += Q[i * d + k] * K[j * d + k];
            S[i * N + j] = s * scale;
        }
    }
    // softmax per row
    for (int i = 0; i < N; ++i) {
        float m = -INFINITY;
        for (int j = 0; j < N; ++j) m = std::max(m, S[i * N + j]);
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            S[i * N + j] = std::exp(S[i * N + j] - m);
            sum += S[i * N + j];
        }
        for (int j = 0; j < N; ++j) S[i * N + j] /= sum;
    }
    // O = S V
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < d; ++k) {
            float s = 0.0f;
            for (int j = 0; j < N; ++j) s += S[i * N + j] * V[j * d + k];
            O[i * d + k] = s;
        }
    }
}

// -----------------------------------------------------------------------------
// Driver
// -----------------------------------------------------------------------------
int main()
{
    const int N = 128;   // sequence length
    const int d = 64;    // head dim

    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> hQ(N * d), hK(N * d), hV(N * d);
    for (auto& x : hQ) x = dist(rng);
    for (auto& x : hK) x = dist(rng);
    for (auto& x : hV) x = dist(rng);

    float *dQ, *dK, *dV, *dO, *dS;
    CUDA_CHECK(cudaMalloc(&dQ, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dS, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup + timed run
    attention_forward(dQ, dK, dV, dO, dS, N, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    attention_forward(dQ, dK, dV, dO, dS, N, d);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    std::vector<float> hO(N * d);
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, N * d * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare to CPU
    std::vector<float> refO(N * d);
    attention_cpu(hQ, hK, hV, refO, N, d);

    double max_abs = 0.0, sum_sq = 0.0;
    for (int i = 0; i < N * d; ++i) {
        double diff = std::fabs(hO[i] - refO[i]);
        max_abs = std::max(max_abs, diff);
        sum_sq += diff * diff;
    }
    double rmse = std::sqrt(sum_sq / (N * d));

    printf("N=%d, d=%d\n", N, d);
    printf("GPU time:  %.3f ms\n", ms);
    printf("Max |err|: %.3e\n", max_abs);
    printf("RMSE:      %.3e\n", rmse);
    printf("%s\n", (max_abs < 1e-3) ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dS);
    return 0;
}
