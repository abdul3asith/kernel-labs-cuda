# CUDA Kernel Lab


session - 1

session - 2
## Observations

- For small input sizes, kernel runtime changes only slightly because launch overhead dominates.
- As input size grows, kernel time increases more clearly because more elements are processed.
- Block size affects performance by changing how work is grouped into blocks and scheduled on the GPU.
- In my benchmark runs, kernel execution time was higher than both host-to-device and device-to-host transfer time for the tested configurations. This indicates that, in this setup, the compute stage was the dominant contributor to runtime.


session - 3
## Time Breakdown for Representative Run

For `n = 1 << 30` and `blockSize = 512`:

- H2D copy: 62.68%
- Kernel execution: 1.89%
- D2H copy: 35.41%

In this setup, H2D Copy was the largest contributor to runtime.

session - 4

# reduction_warp(sets of 32 threads)
Added a warp-level reduction finish to eliminate unnecessary full-block synchronization once only 32 threads remain active. This better matches CUDA’s execution model and reduces overhead in the final reduction stages.