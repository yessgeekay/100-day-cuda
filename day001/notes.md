# Day 1: Setting up CUDA and first CUDA program
I have access to two compute nodes, one with very old hardware, which I get more time on and one with a fairly new hardware (NVIDIA A100), which is less available. Luckily, the A100 node has drivers installed and works out of the box. For the older node, which runs an NVIDIA K40m, I had to install `cuda-toolkit 10.2`. Downloading and installing this was fairly straightforward and provided on the NVIDIA website. CUDA programs are written with `.cu` extension and are compiled using `nvcc`

```cpp
// hello.cu
#include<iostream>
#include<stdio.h>
#include<cuda_runtime.h>

int main(){
std::cout << "Hello world" << std::endl;
return 0;
}
```
To compile this program, we do it the following way.
```
nvcc hello.cu -o hello
```
Execute the program using 
```
./hello
```

## Things learnt
Accessing the CUDA device related properties needs built-in functions loaded by the `cuda_runtime.h` header file. These functions typically return a custom defined `cudaError_t` datatype and we can check for success using `cudaSuccess`
- Counting number of available CUDA devices. . Important functions are 
  - `int deviceID=0; cudaSetDevice(deviceID);`
  - `int deviceID; cudaGetDevice(&deviceID);`
  - `int deviceCount; cudaGetDeviceCount(&deviceCount)` this function updates the pointer to int.
  It seems that, typically, querying a value involves passing the pointer to a variable. It's better to look at the documentation, for e.g. [docuementation for cudaGetDevice](https://developer.download.nvidia.cn/compute/DevZone/docs/html/C/doc/html/group__CUDART__DEVICE_g1795645d53ca669d84d2aff6f3706917.html).
- Setting cuda device. When you have access to multiple cuda devices on your node, they are exposed to you using integers 0, 1 .. They can be set using the `cudaSetDevice` command.
- Catching cuda errors while getting/setting devices: `cudaError_t` is the datatype and `cudaSuccess` is the value of the datatype when there are no errors.

### Types of functions
For a CPU code, which is executed serially, `function(args)` is executed serially for each argument passed. For a GPU kernel, i.e. a device code, specified using the `__global__` declaration specifier, `kernel<<<nBlk, nThr>>>(args)` the launch of a kernel involves parallel processing on a large number of threads on the GPU. The maximum allowed threads for you device can be accessed using the `maxThreadsPerBlock` property of a `cudaDeviceProp` type. The function calls can be differentiated as follows:
- `__device__ float deviceFunc()`: Executed on device: Only callable from device.
- `__global__ float globalFunc()`: Executed on device: Only callable from host.
- `__host__ float hostFunc()`:     Executed on host  : Only callable from host.

The `__device__, __global__, __host__` declaration specifiers are used to instruct the compiler to generate a kernel function, device function or a host function. Threads and blocks will be studied in the next session.

## Practical exercises
- Write device function and host function.
- Set deviceID and check if the said device is indeed activated.
- Print deviceID, threadIdx, blockIdx.
- Write the first kernel to print the same line in different blocks and threads

## Pitfalls
- You cannot specify device in the device code i.e. functions specified by the `__global__` declaration specifier. While fairly obvious, I ended up trying to change the device within the device code and ended up with an error. Set the device in the main program/host code before launching a kernel. The kernel then selects whatever device has been set. By default, deviceID=0 is selected, even when there are multiple available devices.

## References, resources used, suggestions
1. CUDA documentation 
2. Introduciton to CUDA, Chapter 3, Programming massively parallel processors, Kirk and Hwu
