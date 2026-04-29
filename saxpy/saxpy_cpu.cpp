/* single precision a * x plus y (saxpy) - y[i] = a * x[i] + y[i]*/ 
/*CPU SAXPY
CUDA SAXPY
Benchmark different N values
Benchmark different block sizes
Log results to CSV
Plot:
- input size vs kernel time
- input size vs bandwidth
- block size vs performance

CPU sequential SAXPY: one loop does all work
GPU parallel SAXPY: thousands of threads split the work*/


#include <iostream>
#include <vector>

void saxpy_cpu(int n, float a, const float* x, float* y) { // float* x - Pointer to a float array.
    // float is a single scalar value, but float* is a pointer to a float array. We can use it to access the elements of the array.
    // const represents that the function will not modify the contents of the array pointed to by x. This is a common practice to indicate that the input data will not be changed.
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int n = 10;
    float a = 2.0f;
    std::vector<float> x(n, 1.0f); // Initialize x with 1.0f
    std::vector<float> y(n, 2.0f); // Initialize y with 2.0f

    for (int i = 0; i < n; i++) {
        x[i] = i + 1;      // 1, 2, 3...
        y[i] = 10.0f;     // all 10
    }
    saxpy_cpu(n, a, x.data(), y.data());

    for (int i = 0; i < n; i++) {
        std::cout << "y[" << i << "] = " << y[i] << std::endl;
    }

    return 0;
}

/*Expected output:
y[0] = 12
y[1] = 14
y[2] = 16
y[3] = 18
y[4] = 20
y[5] = 22
y[6] = 24
y[7] = 26
y[8] = 28
y[9] = 30
*/

/*Explanation:
y[i] = 2 * (0 + 1) + 10 = 12
y[i] = 2 * (1 + 1) + 10 = 14
y[i] = 2 * (2 + 1) + 10 = 16 
.
.
.
.
y[i] = 2 * (9 + 1) + 10 = 30
*/