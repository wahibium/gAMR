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
cuda:
	$(NVCC) gAMR.cu $(NVCC_CFLAGS) -o run.out
##################################################



clean:
	-$(RM) *.out
	-$(RM) diffusion3d_result.*.out
	
