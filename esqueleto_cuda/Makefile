# Paths & Includes
CUDA_INSTALL_PATH       = /usr/local/cuda
CUDA_LIB_PATH           = $(CUDA_INSTALL_PATH)/lib64
INCLUDES +=  -I/home/karina/Downloads/instaladores -I$(CUDA_INSTALL_PATH)/include  

# Flags
LIBS                    = -L$(CUDA_LIB_PATH) -L../common/lib/linux/x86_64 -L/usr/X11R6/lib
LDFLAGS                 =  -lglut -lGLU -lGL -lcufft \

NVCC_FLAGS              =  #-O2 --use_fast_math

OMP_FLAGS               = -O2 -fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -lgomp -DOMP -I$(CUDA_INSTALL_PATH)/include
TBB_FLAGS               = -O2 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB -ltbb
#TBB_FLAGS               = -O2 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB -ltbb -L$(TBB_PATH)/lib



# Parametros default
#Nciudades			= 9
#TRUN			= 100


PARAMETERS		= -D__STDC_CONSTANT_MACROS  
 

#NVCC                    = $(CUDA_INSTALL_PATH)/bin/nvcc -arch=sm_21
NVCC                    = nvcc
GCC			= g++
ICC			= /opt/intel/bin/icc
LINKER			= $(NVCC)


# Rules
all: clean mosquitas

mosquitas: celdas.h main.cu parametros.h Makefile
	$(LINKER) $(NVCC_FLAGS) -o mosquitas main.cu  $(LDFLAGS) $(INCLUDES) $(LIBS) $(PARAMETERS) 

clean:
	rm -f $(BIN) *.o *.linkinfo cuda_profile_?.log mosquitas


