#include <cuda_runtime.h>

__global__ void cuda_kernel(float* a, float* b, float* c){
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
  return;
}
