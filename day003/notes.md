# Day 02: Setting up development environment and CUDA kernel for addition

## Table of Contents
1. [Overview](#1-overview)
2. [Kernels](#2-kernels)
3. [Threads & Blocks](#3-threads)
3. [Results](#4-results)

## 1. Overview
I explore what CUDA kernels, threads, blocks and clusters. I also look at memory allocation for arrays.

## 2. Kernels
Kernels are code which are executed on the GPU. A kernel is defined by the `__global__` declaration specifier. Each kernel call is specified using an execution configuration `<<<...>>>` which lets the kernel know how many parallel computing threads to use. Kernel invocation with `N` threads can be written as 
```
__global__ void vecAdd(const float* A, const float* B, float* C, int size) {
  int i = threadIdx.x;
  if (i<size) {
  C[i] = A[i] + B[i];
  }
  return;
}

int main(){
  ...
  // kernel invocation with N threads
  int Nthreads = 30;
  int size = 20;
  vecAdd<<<1, N>>>(A, B, C, size);
  ...
}
```

### Thread hierarchy
`threadIdx` is a 3D thread index forming a *thread block*. The thread and its ID are related in the following way
1. For 1D block, thread ID is the same as the thread index, threadIdx.x
2. For 2D block, (Dx, Dy), thread ID = threadIdx.x + threadIdx.y * blockDim.x
3. For 3D block, (Dx, Dy, Dz), thread ID = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y

```
// 2D matrix addition kernel
__global__ void matAdd(...){
  ...
  int i = threadIdx.x;
  int j = threadIdx.y;
  C[i][j] = A[i][j] + B[i][j];
  ...
}

int main(){
  ...
  int numBlocks = 1;
  dim3 threadsPerBlock(N, N);
  matAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
  ...
}
```
Each chip has its own limitation on the maximum allowed threads per block. This can be obtained from functions included in `cuda_runtime.h` [1].

```
int main() {
  ...
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(N/threadsPerBlock.x, N/threadsPerBlock.y);
  matAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
  ...
}
```
- Thread blocks can be executed independently.
- Threads within a block can share data using _shared memory_ and coordinate synchornizing the execution using `__syncthreads()` (this waits for all threads in a block to finish execution before proceeeding).

### Thread block clusters
Thread blocks can be "tied" together in clusters. Cluster[Block1, Block2, ..., Block8], and each of these blocks have threads; Block[Thread1, Thread2, ... Thread256]. Currently a maximum of 8 blocks per cluster are used. This depends on the GPU used. To enable block cluster, we need to use the compile time kernel attribute `__cluster_dims__(X, Y, Z)` or using the CUDA kernel launch API `cudaLaunchKernelEx`.
```
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float* input, float* output) {
...
}

__global__ void cluster2kernel(float* input, float* output) {
...
}

int main(){
  float *input, *output;
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(N/threadsPerBlock.x, N/threadsPerBlock.y);
  cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);

  {
    cudaLaunchConfig_t config = {0};
    config.gridDim = numBlocks;
    config.blockDim = threadsPerBlock;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, cluster2kernel, input, output);
  }
}
```
At first glance, it seems that using the compile time kernel attribute is easier. I dont understand this yet.

## 3. Memory allocation
The host and device arrays are different. For each, the process of memory allocation, population and cleanup are necessary. The computing process can be summarized as follows.
1. Allocate host memory
2. Populate elements of host array
3. Allocate device memory
4. Copy data from host to device
5. Execute on device and synchronize
6. Copy data from device to host
7. Free memory

For memory allocation on the host, we could do it the standard C way or the C++ way
```
const int N = 16;
const int size = N*N*sizeof(float);

// standard C method
float *h_A = (float*)malloc(size);
...
free(h_A);

// C++ method 
float* h_A = new float[N*N];
...
delete[] h_A;

// C++ method using vectors
std::vector<std::vector<float>> h_A(N);
// No delete needed, memory is cleared when out of scope!
```
For memory allocation on device, we use
```
float *d_A;
cudaMalloc(&d_A, size);

cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
cudaFree(d_A);
```


## References
1. [Notes for Day 1](day001/notes.md)
2. [CUDA 120 Days challenge](https://github.com/AdepojuJeremy/CUDA-120-DAYS--CHALLENGE)
3. [CUDA C++ programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
