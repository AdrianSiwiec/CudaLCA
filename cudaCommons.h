#include <cuda_runtime.h>

void CudaListRank( int* devRank, int N, int* devNext, int threadsPerBlockX, int blocksPerGridX );
void CudaAssert( cudaError_t error, const char* code, const char* file, int line );

#define CUCHECK( x ) CudaAssert( x, #x, __FILE__, __LINE__ )
