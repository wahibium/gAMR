# Specify Program Name
PROGRAM = run.out

# Specify Object Files
OBJS  = gAMR_cpu.o gAMR_gpu.o main.o benchmark.o

# CUDA home
CUDA_HOME = /usr/local/cuda-7.0

# Specify Compilers and Flags
CC  = gcc
CFLAGS = -O3 -Wall -g
CXXFLAGS = -I../.. -O3 -Wall -g
LDFLAGS = -I $(CUDA_HOME)/include -lcudart -lm

OPENMP_CFLAGS = -fopenmp
OPENMP_LDFLAGS = -fopenmp


NVCC = nvcc
NVCC_CFLAGS = -I../.. -O3 -Xcompiler -Wall -Xptxas -v -arch sm_35 # -keep
CUDA_INC = $(patsubst %bin/nvcc,%include, $(shell which $(NVCC)))
ifeq (,$(findstring Darwin,$(shell uname)))
        CUDA_LDFLAGS = -lcudart -L$(patsubst %bin/nvcc,%lib64, \
                $(shell which $(NVCC)))
 else
        NVCC_CFLAGS += -m64
        CUDA_LDFLAGS = -lcudart -L$(patsubst %bin/nvcc,%lib, \
                $(shell which $(NVCC)))
endif

.SUFFIXES: .cc     $(SUFFIXES)
.SUFFIXES: .cu     $(SUFFIXES)

.cc.o:
		$(CC) $(CFLAGS) $(OPENMP_CFLAGS) $(LDFLAGS) $(CUDA_LDFLAGS) -c $?  -o $@
.cu.o:
		$(NVCC) $(NVCC_CFLAGS) $(CUDA_LDFLAGS) -c $? -o $@

all: $(OBJS)
	$(CC) -o $(PROGRAM) $(OBJS) $(CFLAGS) $(OPENMP_CFLAGS) $(LDFLAGS) $(CUDA_LDFLAGS) -lstdc++

######################
######################
clean: 
	rm -f *.o *~ benchmark_result.*.out $(PROGRAM)
