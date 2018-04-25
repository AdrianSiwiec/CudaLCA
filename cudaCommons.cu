#include <cuda_runtime.h>
#include <iostream>
#include "cudaCommons.h"

using namespace std;

__global__ void cuCalcRankRead( int V, int *next, int *depth, int *tmp, int *notAllDone );
__global__ void cuCalcRankWrite( int V, int *next, int *depth, int *tmp );
__global__ void cuMoveNextRead( int V, int *next, int *tmp );
__global__ void cuMoveNextWrite( int V, int *next, int *tmp, int *notAllDone );

void CudaListRank( int *devRank, int N, int *devNext, int threadsPerBlockX, int blocksPerGridX )
{
  int *notAllDone = (int *) malloc( sizeof( int ) );

  int *devTmp;
  int *devNotAllDone;

  CUCHECK( cudaMalloc( (void **) &devTmp, sizeof( int ) * N ) );
  CUCHECK( cudaMalloc( (void **) &devNotAllDone, sizeof( int ) ) );

  do
  {
    cuCalcRankRead<<<blocksPerGridX, threadsPerBlockX>>>( N, devNext, devRank, devTmp, devNotAllDone );
    CUCHECK( cudaDeviceSynchronize() );

    cuCalcRankWrite<<<blocksPerGridX, threadsPerBlockX>>>( N, devNext, devRank, devTmp );
    CUCHECK( cudaDeviceSynchronize() );

    cuMoveNextRead<<<blocksPerGridX, threadsPerBlockX>>>( N, devNext, devTmp );
    CUCHECK( cudaDeviceSynchronize() );

    cuMoveNextWrite<<<blocksPerGridX, threadsPerBlockX>>>( N, devNext, devTmp, devNotAllDone );
    CUCHECK( cudaDeviceSynchronize() );

    CUCHECK( cudaMemcpy( notAllDone, devNotAllDone, sizeof( int ), cudaMemcpyDeviceToHost ) );
  } while ( *notAllDone );

  free( notAllDone );
  CUCHECK( cudaFree( devTmp ) );
  CUCHECK( cudaFree( devNotAllDone ) );
}

__global__ void cuCalcRankRead( int V, int *next, int *rank, int *tmp, int *notAllDone )
{
  int thid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
  if ( thid == 0 ) *notAllDone = 0;

  if ( thid >= V || next[thid] == -1 ) return;

  tmp[thid] = rank[next[thid]] + 1;
}

__global__ void cuCalcRankWrite( int V, int *next, int *rank, int *tmp )
{
  int thid = ( blockIdx.x * blockDim.x ) + threadIdx.x;

  if ( thid >= V || next[thid] == -1 ) return;

  rank[thid] += tmp[thid];
}

__global__ void cuMoveNextRead( int V, int *next, int *tmp )
{
  int thid = ( blockIdx.x * blockDim.x ) + threadIdx.x;

  if ( thid >= V || next[thid] == -1 ) return;

  tmp[thid] = next[next[thid]];
}

__global__ void cuMoveNextWrite( int V, int *next, int *tmp, int *notAllDone )
{
  int thid = ( blockIdx.x * blockDim.x ) + threadIdx.x;

  if ( thid >= V || next[thid] == -1 ) return;

  next[thid] = tmp[thid];

  *notAllDone = 1;
}

void CudaAssert( cudaError_t error, const char *code, const char *file, int line )
{
  if ( error != cudaSuccess )
  {
        cerr << "Cuda error :" << code << ", " << file << ":" << error << endl;
    exit( 1 );
  }
}

void CudaPrintTab( int *tab, int size )
{
  int *tmp = (int *) malloc( sizeof( int ) * size );
  CUCHECK( cudaMemcpy( tmp, tab, sizeof( int ) * size, cudaMemcpyDeviceToHost ) );

  for ( int i = 0; i < size; i++ )
  {
    cerr << tmp[i] << " ";
  }
  cerr << endl;

  free( tmp );
}