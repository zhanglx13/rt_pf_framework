TLPI_PATH ?= /home/lixun/Dropbox/PhD/tlpi-dist
CUDA_PATH ?= /usr/local/cuda
CUDA_SDK_PATH ?= /usr/local/cuda/samples/common/inc

HOST_COMPILER ?= gcc
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

ifeq ($(verbose),1)
	MACRO=-DVERBOSE
endif
MACRO += -D$(MODEL)

all: main SI_gpu

main.o: main.c
	gcc -I$(TLPI_PATH)/lib -L$(TLPI_PATH) $(MACRO) -o $@ -c $< -ltlpi

main: main.o model_$(MODEL).o
	gcc -I$(TLPI_PATH)/lib -L$(TLPI_PATH) -o $@ $^ -lrt -ltlpi -lm

model_VAR.o: model_VAR.c model_VAR.h
	gcc -I$(TLPI_PATH)/lib -L$(TLPI_PATH) -o $@ -c $< -lrt -ltlpi

model_triv.o: model_triv.c model_triv.h
	$(NVCC) -I$(TLPI_PATH)/lib -L$(TLPI_PATH) -o $@ -c $< -lrt -ltlpi -lm

model_EKD.o: model_EKD.c model_EKD.h
	gcc -I$(TLPI_PATH)/lib -L$(TLPI_PATH) -o $@ -c $< -lrt -ltlpi

SI_gpu: SI_gpu.o model_$(MODEL).o
	$(NVCC) -I$(TLPI_PATH)/lib -L$(TLPI_PATH) -o $@ $^ -ltlpi -lrt -lm -lcurand

SI_gpu.o: SI_gpu.cu
	$(NVCC) -I$(TLPI_PATH)/lib -L$(TLPI_PATH) $(MACRO) -o $@ -c $<  -ltlpi

clean:
	rm -f main *.o
