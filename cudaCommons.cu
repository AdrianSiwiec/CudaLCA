#include <cuda_runtime.h>
#include <iostream>

using namespace std;

void CudaAssert( cudaError_t error, const char* code, const char* file, int line )
{
  if ( error != cudaSuccess )
  {
    cerr << "Cuda error :" << code << ", " << file << ":" << endl;
    exit( 1 );
  }
}