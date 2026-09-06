NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17 -arch=native

all: matmul

matmul: matmul.cu
	$(NVCC) $(NVCCFLAGS) -o matmul matmul.cu

clean:
	rm -f matmul

.PHONY: all clean
