# Day 02: Setting up development environment and CUDA kernel for addition

## Table of Contents
1. [Overview](#1-overview)
2. [Organizing my project](#2-project-organization)
3. [Compiling and building](#3-compilation)
3. [Results](#4-results)

## 1. Overview
Understanding how to setup environment. Having written most of my scientific computing scripts in python, it's important for me to understand how to build a scalable, maintainable environment for CUDA code. I'm heavily relying on [2] for setting this up. This is to understand how to build larger project, importing files, libraries, using appropriate compiler flags, etc. Broadly, I explore the following:
- Setting up an organized folder structure for multi-file application.
- Configuration of compilation flags and environment variables.
- Using debugging and profiling tools.
In addition to this, I also write my first CUDA kernel for addition and multiplication of matrices and compute speedup of GPU computation.

## 2. Organizing my project
```
100-days-cuda/
├── day001/
│   ├── Makefile         # Local makefile
│   ├── main.cu          # Host code & kernel invocations
│   ├── kernels.cu       # Device kernels (can be multiple .cu files)
│   └── utils.cu         # Additional device or host utilities
├── day002/
│   ├── Makefile         # Local makefile
│   ├── main.cu          # Host code & kernel invocations
│   ├── kernels.cu       # Device kernels (can be multiple .cu files)
│   └── utils.cu         # Additional device or host utilities
├── utils/
│   ├── timing.hpp       # timing function
│   └── utils.h          # other utilities
├── Makefile (or CMakeLists.txt)
├── docs/                # Optional: documentation, design notes
└── README.md
```
While the idea is to have separate files for the host code and device kernels [2], the files up to `day002` are written in a single file. I have `Makefile` in the parent directory, that lets me compile the main programs for the work done on any day. This parent makefile builds the `Makefile` within the selected directory. For e.g., `make 2` performs `make all` for `day002`. This lets me build custom makefiles for each day's work. Advice from [2] is to keep separate `.h` files for device and host function prototypes. I have the timing scripts in the `utils/timing.hpp` file, which uses `std::chrono` for timing the device as well as host executions.

## 3. Compiling and building
I use `nvcc` for compilation. 
```
nvcc -O2 -o output main.cu
-o is used to set the output filename
-O2 is the compiler optimization flag
-G can be used to enable debug info for device code
-lineinfo for better profiling and debugging
```
An example makefile for `day002` is given below
```
PROJECT = ./output.o
NVCC = nvcc
SRCS = ./matmul.cu
FLAGS = -O2

all: $(PROJECT) 

$(PROJECT): $(SRCS)
        $(NVCC) $(ARCH) $(FLAGS) $(INCLUDES) -o $(PROJECT) $(SRCS)

clean:
        rm -f $(PROJECT) *.o
```
Other additions/improvements will be done as I learn more.

### Using compiler flags `Ox`
Using the optimizer flag for compilation significantly improves the execution time. This can be chosen between `O0` to `O3`, with `O3` producing the most optimized code, while also taking the longest for compilation. Typically use `O0` during development and `O3` for the release version. [1] Note that `O3` aggressively optimizes the code loop vectorization and unrolling, which can dramatically increase executable size. It could also result in "wrong" results for scientific computation as `O3` optimizing can re-order floating point operations and could potentially result in output being different from, say, an `O2` compiled code. Always benchmark it before production.

## 4. First CUDA kernel
Wrote a simple CUDA kernel for matrix multiplication. The idea is to parallelize the computation of every element in the output matrix. This means that the kernel computes a dot product of two vectors. For a 1024x1024 matrix, I achieved a speedup of 250x.

## References
1. [C++ compiler flags reference](https://caiorss.github.io/C-Cpp-Notes/compiler-flags-options.html#org68aa48b)
2. [CUDA 120 Days challenge](https://github.com/AdepojuJeremy/CUDA-120-DAYS--CHALLENGE)
3. [CUDA C++ programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
