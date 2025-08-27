PROJECT = ./output.o
NVCC = nvcc
SRCS = ./main.cu
FLAGS = -O2

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDES) -o $(PROJECT) $(SRCS)

clean:
	rm -f $(PROJECT) *.o

