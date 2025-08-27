#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include "kernels.h"
#include "../utils/timing.hpp"


// Initialize matrix with random values
void init_matrix(float* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}



int main(){

  const int N = 256;
  const float size = N*N*sizeof(float);
  dim3 threadsPerBlock(64, 64);
  dim3 numBlocks(N/threadsPerBlock.x, N/threadsPerBlock.y);

  // 1. Initialize array, allocate host memory
  // 2. Allocate host memory
  // 3. Allocate device memory
  // 4. Copy arrays from host to device
  // 5. Execute on device
  // 6. Copy results from device to host
  // 7. synchronize and cleanup

  // step1: initializing arrays, allocate host memory
  float* h_A = new float[N*N];
  float* h_B = new float[N*N];
  float* h_C = new float[N*N];

  // - updating host arrays 
  init_matrix(h_A, N);
  init_matrix(h_B, N);

  // step3: device memory allocation
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  // step4: copy host -> device
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  cuda_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  delete[] h_A; delete[] h_B; delete[] h_C;
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

  return 0;
}
