// naive_qkt.cu
// Naive CUDA matmul, applied to attention's first step: S = Q @ K^T.
// Q, K, V have shape [N, d]. We compute S = Q @ K^T -> [N, N].
// (V isn't used yet — it comes in after softmax. We allocate it so the
//  setup mirrors a real attention pipeline.)

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

// Naive Q @ K^T.
// Q : [N, d]    row i is the query vector for token i
// K : [N, d]    row j is the key vector   for token j
// S : [N, N]    S[i, j] = dot(Q[i, :], K[j, :])
//
// One thread = one output element S[i, j].
// Because we want K^T, we read K[j, k] instead of K[k, j] — no transpose needed.
__global__ void naive_qkt_kernel(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 float* __restrict__ S,
                                 int N, int d)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // query / row of S
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // key   / col of S

    if (i >= N || j >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < d; ++k) {
        // Q[i, k] * K[j, k]  -> dot product over the head dim
        acc += Q[i * d + k] * K[j * d + k];
    }
    S[i * N + j] = acc;
}

// CPU reference for Q @ K^T.
void qkt_cpu(const std::vector<float>& Q,
             const std::vector<float>& K,
             std::vector<float>& S,
             int N, int d)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < d; ++k) {
                s += Q[i * d + k] * K[j * d + k];
            }
            S[i * N + j] = s;
        }
    }
}

int main()
{
    const int N = 512;   // sequence length
    const int d = 64;    // head dim

    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> hQ(N * d), hK(N * d), hV(N * d);
    std::vector<float> hS(N * N), refS(N * N);
    for (auto& x : hQ) x = dist(rng);
    for (auto& x : hK) x = dist(rng);
    for (auto& x : hV) x = dist(rng);   // unused for now, but here for the pipeline

    float *dQ, *dK, *dV, *dS;
    CUDA_CHECK(cudaMalloc(&dQ, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dS, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), N * d * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);                              // 256 threads/block
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    // Warmup
    naive_qkt_kernel<<<grid, block>>>(dQ, dK, dS, N, d);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    naive_qkt_kernel<<<grid, block>>>(dQ, dK, dS, N, d);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    CUDA_CHECK(cudaMemcpy(hS.data(), dS, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    qkt_cpu(hQ, hK, refS, N, d);
    double max_abs = 0.0;
    for (int i = 0; i < N * N; ++i) {
        max_abs = std::max(max_abs, (double)std::fabs(hS[i] - refS[i]));
    }

    // 2*N*N*d flops: one mul + one add per inner step.
    double gflops = (2.0 * N * N * d) / (ms * 1e6);

    printf("N=%d, d=%d\n", N, d);
    printf("GPU time:   %.3f ms\n", ms);
    printf("Throughput: %.1f GFLOP/s\n", gflops);
    printf("Max |err|:  %.3e\n", max_abs);
    printf("%s\n", (max_abs < 1e-2) ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dS);
    return 0;
}
// Q = [[1, 0], [0, 1]] K = [[1, 0], [0, 1]] ==> S = [[1+0, ]
