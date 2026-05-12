# CUDA Kernel Lab


### session - 1

### session - 2
## Observations

- For small input sizes, kernel runtime changes only slightly because launch overhead dominates.
- As input size grows, kernel time increases more clearly because more elements are processed.
- Block size affects performance by changing how work is grouped into blocks and scheduled on the GPU.
- In my benchmark runs, kernel execution time was higher than both host-to-device and device-to-host transfer time for the tested configurations. This indicates that, in this setup, the compute stage was the dominant contributor to runtime.


### session - 3
## Time Breakdown for Representative Run

For `n = 1 << 30` and `blockSize = 512`:

- H2D copy: 62.68%
- Kernel execution: 1.89%
- D2H copy: 35.41%

In this setup, H2D Copy was the largest contributor to runtime.

### session - 4

## reduction_warp(sets of 32 threads)
Added a warp-level reduction finish to eliminate unnecessary full-block synchronization once only 32 threads remain active. This better matches CUDA’s execution model and reduces overhead in the final reduction stages.

### session - 5

<----- finish documenting relu and gelu activation functions ----->
Coalescing means combining many small memory accesses into one big memory transaction.

### session - 6

<--- LayerNorm Kernel --->

LayerNorm combines reduction, normalization, and memory bandwidth thinking in one operator. 

LayerNorm is basically:

For every row:
    summarize the row using reductions - reduction  = mean
    then scale every value in that row
    1. mean - sum(x) / cols
    2. Variance - sum((x - mean)^2) / cols
    3. Normalize - (x - mean) / sqrt(variance + eps) - eps is a small safety value.
    after LayerNorm, the numbers are easier for the neural network to work with.