#include <cuda_runtime.h>

void CudaAssert( cudaError_t error, const char* code, const char* file, int line );

#define CUCHECK( x ) CudaAssert( x, #x, __FILE__, __LINE__ )
