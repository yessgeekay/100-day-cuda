PROJECT0 = ./output0.o
PROJECT1 = ./output1.o
PROJECT2 = ./output2.o
PROJECT3 = ./output3.o
NVCC = nvcc
SRCS = ./main.cu ./kernels.cu
# INCLUDES = -I./include
# ARCH = -arch=sm_75
# FLAGS = -Xcompiler "-Wall -O2"
FLAGS0 = -O0 
FLAGS1 = -O1
FLAGS2 = -O2 
FLAGS3 = -O3 

debug: $(PROJECT1)

$(PROJECT1): $(SRCS)
	$(NVCC) $(ARCH) $(FLAGS1) $(INCLUDES) -o $(PROJECT1) $(SRCS)

all: $(PROJECT0) $(PROJECT1) $(PROJECT2) $(PROJECT3)

$(PROJECT0): $(SRCS)
	$(NVCC) $(ARCH) $(FLAGS0) $(INCLUDES) -o $(PROJECT0) $(SRCS)

$(PROJECT1): $(SRCS)
	$(NVCC) $(ARCH) $(FLAGS1) $(INCLUDES) -o $(PROJECT1) $(SRCS)

$(PROJECT2): $(SRCS)
	$(NVCC) $(ARCH) $(FLAGS2) $(INCLUDES) -o $(PROJECT2) $(SRCS)

$(PROJECT3): $(SRCS)
	$(NVCC) $(ARCH) $(FLAGS3) $(INCLUDES) -o $(PROJECT3) $(SRCS)

clean:
	rm -f $(PROJECT0) *.o
	rm -f $(PROJECT1) *.o
	rm -f $(PROJECT2) *.o
	rm -f $(PROJECT3) *.o

