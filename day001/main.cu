#include<stdio.h>
#include<iostream>
#include<cuda_runtime.h>

/* 
- functions in c++ which execute on gpu are called as kernels
- each kernel call results in n-executions through n parallel threads
- a kernel is specified with a __global__ declaration specifier
- the kernel call needs to specify the number of threads using <<<...>>>
- each execution is assigned a unique thread id, exposed via built-int variables,
  which are accessible only inside the kernels (e.g., threadidx.x)
*/

__host__ void helloFromHost(){
  // execution on the cpu
  printf("hello from the cpu");
  return;
}

__device__ int getBlockIdx(){
  return static_cast<int>(blockIdx.x);
}


/* 
- by default deviceID=0 is taken.
- if you choose to use a different device, it has to be set before calling the kernel
- cannot specify device in the device code, which is speicifed by the __global)) declaration specifier.
- device code is executed on an already set device
*/
__global__ void helloFromDevice(int deviceID){
  printf("hello from GPU[%d]: (block:%d, threadx:%d, thready:%d)\n",
         deviceID, getBlockIdx(), threadIdx.x, threadIdx.y);
  return;
  // note that no variables are passed to the function
  // and all the information printed use the built in variables
}


/* 
- device properties can also be obtained by using built-in functions
  given in the cuda_runtime.h library.
- int deviceCount; cudaGetDeviceCount(&deviceCount)
- cudaDeviceProp prop; cudaGetDeviceProperties(&prop, i);
  - prop.name, prop.major
  - prop.totalGlobalMem (in B)
  - prop.multiProcessorCount
  - prop.clockRate (kHz)
*/
__host__ int getNumberDevices(){
  int deviceCount = 0;
  
  // calling CUDA runtime to get device count
  // It takes the pointer to the variable and returns an error code
  cudaError_t error = cudaGetDeviceCount(&deviceCount);

  if (error != cudaSuccess){
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (deviceCount == 0){
    std::cout << "There are no available CUDA devices" << std::endl;
  } else {
    std::cout << "Found " << deviceCount << " devices." << std::endl;
  }

  return deviceCount;
}


__host__ int printDeviceDetails(int deviceCount){
  int currentDevice;
  cudaError_t err;
  std::cout << "----------------------------------------------------" << std::endl;
  for (int i=0; i<deviceCount; ++i){
    err = cudaSetDevice(i);
    if (err != cudaSuccess){
      std::cerr << "Set device failed for deviceID: " << i << std::endl;
      return -1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    err = cudaGetDevice(&currentDevice);
    if (err != cudaSuccess){
      std::cerr << "Get device failed for deviceID: " << i << std::endl;
      return -1;
    }
    std::cout << "[" << i << "," << currentDevice << "] " << "Name: " << prop.name << std::endl;
    std::cout << "[" << i << "," << currentDevice << "] " << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "[" << i << "," << currentDevice << "] " << "Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "[" << i << "," << currentDevice << "] " << "Multiprocessor Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "[" << i << "," << currentDevice << "] " << "MaxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "[" << i << "," << currentDevice << "] " << "Clock Rate: " << prop.clockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
  }
  return 0;
}

/*
- threadIdx is a 3 component vector
- index of thread can be written in terms of 

*/

__host__ int main(){
  int deviceCount = 0;
  deviceCount = getNumberDevices();
  std::cout << "Printing device properties using device count" << std::endl;
  printDeviceDetails(deviceCount);
  std::cout << "Querying properties for N>device count" << std::endl;
  printDeviceDetails(20);
  int numBlocks = 3;
  int N = 2;
  dim3 threadsPerBlock(N, N);
  /* dim3 is an integer vector type defined by CUDA (vector_types.h)
     dim3 gridName(Nx, Ny, Nz) can be used to define 3D thread
  */
  printf("----------------------------\n");

  // Launching GPU kernel with 1 block and 1 thread
  for (int i=0; i<deviceCount; ++i){
    cudaError_t err = cudaSetDevice(i);
    if (err != cudaSuccess){
      std::cout << "cuda error: " << cudaGetErrorString(err) << std::endl;
    } else {
    helloFromDevice<<<numBlocks, threadsPerBlock>>>(i);
    cudaDeviceSynchronize(); // Synchronize before exiting GPU
    }
  }
  printf("----------------------------\n");
  helloFromHost();
  return 0;
}

