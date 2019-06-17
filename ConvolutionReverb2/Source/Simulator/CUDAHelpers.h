#include "Simulator.h"

#include <cuda_runtime.h>

#ifdef _DEBUG
#define ON_ERROR __debugbreak(); abort();
#else
#define ON_ERROR abort();
#endif

#define CUDA(x) { cudaError_t err = x; \
	if(err != cudaSuccess) { \
		printf("CUDA error!\n\tfile: %s\n\tline: %d\n\tfunc: %s\n\tcode: %s\n", __FILE__, __LINE__, #x, cudaGetErrorName(err)); ON_ERROR;\
	} }

