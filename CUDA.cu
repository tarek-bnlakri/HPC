#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// Device function to check primality on GPU
__device__ bool is_prime_gpu(long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

// CUDA Kernel
__global__ void find_twin_primes_kernel(long *data, long count, unsigned long long *total_twins) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < count - 1) {
        // Skip duplicates
        if (i > 0 && data[i] == data[i-1]) return;

        long p = data[i];
        if (is_prime_gpu(p)) {
            long target = p + 2;
            long low = 0, high = count - 1;
            
            // Binary search for p + 2
            while (low <= high) {
                long mid = low + (high - low) / 2;
                if (data[mid] == target) {
                    if (is_prime_gpu(target)) {
                        atomicAdd(total_twins, 1);
                    }
                    break;
                }
                if (data[mid] < target) low = mid + 1;
                else high = mid - 1;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <file.csv>\n", argv[0]);
        return 1;
    }

    // 1. Load Data on Host
    long capacity = 10000000;
    long *h_data = (long*)malloc(capacity * sizeof(long));
    long count = 0;

    FILE* file = fopen(argv[1], "r");
    if (!file) { printf("File error\n"); return 1; }
    while (count < capacity && fscanf(file, "%ld,", &h_data[count]) != EOF) {
        count++;
    }
    fclose(file);

    // 2. Allocate and Copy to Device (GPU)
    long *d_data;
    unsigned long long *d_result;
    unsigned long long h_result = 0;

    cudaMalloc(&d_data, count * sizeof(long));
    cudaMalloc(&d_result, sizeof(unsigned long long));
    cudaMemcpy(d_data, h_data, count * sizeof(long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // 3. Sort on GPU (using Thrust library for convenience)
    thrust::device_ptr<long> d_ptr(d_data);
    thrust::sort(d_ptr, d_ptr + count);

    // 4. Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    
    find_twin_primes_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, count, d_result);

    // 5. Copy Result Back
    cudaMemcpy(&h_result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    printf("Total elements: %ld\n", count);
    printf("Unique twin prime pairs: %llu\n", h_result);

    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);

    return 0;
}
