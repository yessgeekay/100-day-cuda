#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <array>
#include <cuda_runtime.h>



// Initialize 1d array with random values
void init1d(float* A, int size) {
    for (int i = 0; i < size; i++) {
        A[i] = (float)rand() / RAND_MAX;
    }
    return;
}

// adding 1D vectors (host)
__host__ void add1d(const std::vector<float>& A, 
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int size) {
  for (int i=0; i<size; ++i) {
    C[i] = A[i] + B[i];
  }
  return;
}

// adding 1D vectors (host)
__host__ void add1d(const float* A, 
                    const float* B,
                    float* C, int size) {
  for (int i=0; i<size; ++i) {
    C[i] = A[i] + B[i];
  }
  return;
}

// adding 1D vectors (device)
__global__ void add1d(const float* A, const float* B, float* C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}
    


// adding 2D vectors
__host__ void add2d(const std::vector<std::vector<float>>& A, 
                    const std::vector<std::vector<float>>& B,
                    std::vector<std::vector<float>>& C, int rows, int cols) {
  for (int i=0; i<rows; ++i) {
    for (int j=0; j<cols; ++j) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
  return;
}


// adding 2D arrays
__host__ void add2d(const float* A, const float* B, float* C, int rows, int cols) {
    int index = 0;
    for (int i=0; i<rows; ++i) {
      for (int j=0; j<cols; ++j) {
        index = i*rows + j;
        C[index] = A[index] + B[index];
      }
    }
    return;
}


int main(){
  const int N = 1024;
  float* A = new float[N];
  float* B = new float[N];
  float* C = new float[N];

  init1d(A, N);
  init1d(B, N);
  add1d(A, B, C, N);
  return 0;
}
  
