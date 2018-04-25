#include <cuda_runtime.h>

void CudaListRank( int* devRank, int N, int* devNext, int threadsPerBlockX, int blocksPerGridX );

void CudaAssert( cudaError_t error, const char* code, const char* file, int line );

void CudaPrintTab( int* tab, int size );

#define CUCHECK( x ) CudaAssert( x, #x, __FILE__, __LINE__ )
