PROJECT = ./a.out
NVCC = nvcc
SRCS = ./print.cu
# INCLUDES = -I./include
# ARCH = -arch=sm_75
# FLAGS = -Xcompiler "-Wall -O2"
FLAGS = -O2

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDES) -o $(PROJECT) $(SRCS)

clean:
	rm -f $(PROJECT) *.o

