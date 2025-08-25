#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include "../utils/timing.hpp"

// version0: simple 
__host__ void matmul_simple(const std::vector<std::vector<float>>& A, 
                            const std::vector<std::vector<float>>& B,
                            std::vector<std::vector<float>>& C) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    int cols_B = B[0].size();
    
    // Initialize result matrix with zeros
    C.assign(rows_A, std::vector<float>(cols_B, 0.0f));
    
    // Triple nested loop - just like Python but with explicit types
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            for (int k = 0; k < cols_A; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// version1: optimized
__host__ void matmul_optimized(const float* A, const float* B, float* C,
                               int rows_A, int cols_A, int cols_B) {
    // Initialize result matrix
    for (int i = 0; i < rows_A * cols_B; i++) {
        C[i] = 0.0f;
    }
    
    // Matrix multiplication with better memory access pattern
    for (int i = 0; i < rows_A; i++) {
        for (int k = 0; k < cols_A; k++) {
            float a_ik = A[i * cols_A + k];
            for (int j = 0; j < cols_B; j++) {
                C[i * cols_B + j] += a_ik * B[k * cols_B + j];
            }
        }
    }
}


// CUDA kernel - runs on GPU
__global__ void matmul_kernel(float* A, float* B, float* C, int N) {
    // Each thread computes one element of result matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Which row
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Which column
    
    // Make sure we're within bounds
    if (row < N && col < N) {
        float sum = 0.0f;
        // Compute dot product for this element
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Helper function to create random matrix
std::vector<std::vector<float>> create_random_matrix(int rows, int cols) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}


// Initialize matrix with random values
void init_matrix(float* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}



// Convert 2D vector to 1D array for optimized version
void flatten_matrix(const std::vector<std::vector<float>>& matrix, float* flat) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat[i * cols + j] = matrix[i][j];
        }
    }
}


void print_few(std::vector<std::vector<float>>& matrix) {
    std::cout << "Verification (first 3x3 block):\n";
    std::cout << "Simple version:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
    return;
}
  
void print_few(float* matrix, int rows) {
    std::cout << "Verification (first 3x3 block):\n";
    std::cout << "Optimized version:\n";
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << matrix[i*rows + j] << " ";
        }
        std::cout << "\n";
    }
    return;
}

// Verify if CPU and GPU computations match
bool verify_result(float* cpu_result, float* gpu_result, 
                   int N, float tolerance=1e-3) {
    for (int i = 0; i < N * N; i++) {
        if (abs(cpu_result[i] - gpu_result[i]) > tolerance) {
            std::cout << "Mismatch at element " << i << ": CPU=" << cpu_result[i] 
                      << " GPU=" << gpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}


int main() {
    const int N = 1024;  // Matrix size (N x N)
    const int size = N * N * sizeof(float);
    
    std::cout << "Creating " << N << "x" << N << " matrices...\n";
    
    // Create random matrices
    auto A = create_random_matrix(N, N);
    auto B = create_random_matrix(N, N);
    std::vector<std::vector<float>> C;
    
    // Time the simple version
    auto start = std::chrono::high_resolution_clock::now();
    matmul_simple(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Simple version took: " << duration.count() << " ms\n";
    
    // Now try the optimized version with 1D arrays
    float* A_flat = new float[N * N];
    float* B_flat = new float[N * N];
    float* C_flat = new float[N * N];
    
    flatten_matrix(A, A_flat);
    flatten_matrix(B, B_flat);
    
    start = std::chrono::high_resolution_clock::now();
    matmul_optimized(A_flat, B_flat, C_flat, N, N, N);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Optimized version took: " << duration.count() << " ms\n";

    print_few(C);
    print_few(C_flat, N);
    
    std::cout << std::endl;
    std::cout << "Simple CUDA Matrix Multiplication (" << N << "x" << N << ")\n";
    
    // 1. Allocate host (CPU) memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C_cpu = (float*)malloc(size);  // CPU result
    float* h_C_gpu = (float*)malloc(size);  // GPU result copied back
    
    // 2. Initialize matrices
    srand(42);  // For reproducible results
    init_matrix(h_A, N);
    init_matrix(h_B, N);
    
    // 3. Allocate device (GPU) memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // 4. Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 5. Define grid and block dimensions
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;
    const int maxThreadsx = static_cast<int>(std::sqrt(maxThreads));
    std::cout << "max threads = " << maxThreadsx << std::endl;
    dim3 blockSize(maxThreadsx, maxThreadsx);  // 16x16 = 256 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (N + blockSize.y - 1) / blockSize.y);
    
    std::cout << "Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;
    std::cout << "Block size: " << blockSize.x << "x" << blockSize.y << std::endl;
    
    // 6. Launch CUDA kernel
    auto gpu_time = time_it_device([&](){
      matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    });
    
    // 7. Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    
    // 8. Run CPU version for comparison
    auto cpu_time = time_it_host([&](){
    matmul_optimized(h_A, h_B, h_C_cpu, N, N, N);
    });
    //auto cpu_start = std::chrono::high_resolution_clock::now();
    //auto cpu_end = std::chrono::high_resolution_clock::now();
    //auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    
    // 9. Verify results
    bool correct = verify_result(h_C_cpu, h_C_gpu, N);
    
    // 10. Print results
    std::cout << "\nResults:\n";
    std::cout << "CPU Time: " << cpu_time.count() << " ms\n";
    std::cout << "GPU Time: " << gpu_time.count() << " ms\n";
    std::cout << "Speedup: " << (float)cpu_time.count() / gpu_time.count() << "x\n";
    std::cout << "Verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
    
    // Show a few result elements
    print_few(h_C_gpu, N);
   
    // 11. Cleanup
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    // Clean up memory
    delete[] A_flat;
    delete[] B_flat;
    delete[] C_flat;
    
    return 0;
}
