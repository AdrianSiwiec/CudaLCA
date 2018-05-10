#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <moderngpu/transform.hxx>
#include "commons.h"
#include "cudaCommons.h"

using namespace std;
using namespace mgpu;

__global__ void cuCalcRankRead( int V, int *next, int *depth, int *tmp, int *notAllDone );
__global__ void cuCalcRankWrite( int V, int *next, int *depth, int *tmp );
__global__ void cuMoveNextRead( int V, int *next, int *tmp );
__global__ void cuMoveNextWrite( int V, int *next, int *tmp, int *notAllDone );
__device__ int cuAbs( int i );

void CudaSimpleListRank( int *devRank, int N, int *devNext, int threadsPerBlockX, int blocksPerGridX )
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

void CudaFastListRank( int *devRank, int N, int head, int *devNext, standard_context_t &context )
{
  Timer listTimer( "List Rank" );
  int s;
  if ( N > 1000000 )
    s = 50000;
  else
    s = N / 100;
  if ( s == 0 ) s = 1;


  int *sum;
  int *last;
  int *next;
  sum = new int[s + 1];
  last = new int[s + 1];
  next = new int[N];
  // cudaMallocHost( &sum, sizeof( int ) * ( s + 1 ) );
  // cudaMallocHost( &last, sizeof( int ) * ( s + 1 ) );
  // cudaMallocHost( &next, sizeof( int ) * N );

  listTimer.measureTime( "Host Allocs" );

  int *devSum;
  CUCHECK( cudaMalloc( (void **) &devSum, sizeof( int ) * ( s + 1 ) ) );
  int *devSublistHead;
  CUCHECK( cudaMalloc( (void **) &devSublistHead, sizeof( int ) * ( s + 1 ) ) );
  int *devSublistId;
  CUCHECK( cudaMalloc( (void **) &devSublistId, sizeof( int ) * N ) );
  int *devLast;
  CUCHECK( cudaMalloc( (void **) &devLast, sizeof( int ) * ( s + 1 ) ) );

  listTimer.measureTime( "Device Allocs" );

  transform(
      [=] MGPU_DEVICE( int i ) {
        curandState state;
        curand_init( 123, i, 0, &state );

        int p = i * ( N / s );
        int q = min( p + N / s, N );

        int splitter;
        do
        {
          splitter = ( cuAbs( curand( &state ) ) % ( q - p ) ) + p;
        } while ( devNext[splitter] == -1 );

        devSublistHead[i + 1] = devNext[splitter];
        devNext[splitter] = -i - 2;  // To avoid confusion with -1

        if ( i == 0 )
        {
          devSublistHead[0] = head;
        }
      },
      s,
      context );

  listTimer.measureTime( "CPU generating splitters" );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        int current;
        current = devSublistHead[thid];

        int counter = 0;
        while ( current >= 0 )
        {
          devRank[current] = counter;
          counter++;

          int n = devNext[current];

          if ( n < 0 )
          {
            devSum[thid] = counter;
            devLast[thid] = current;
          }

          devSublistId[current] = thid;
          current = n;
        }
      },
      s + 1,
      context );
  context.synchronize();

  listTimer.measureTime( "GPU sublist rank calculation" );

  CUCHECK( cudaMemcpy( sum, devSum, sizeof( int ) * ( s + 1 ), cudaMemcpyDeviceToHost ) );
  CUCHECK( cudaMemcpy( next, devNext, sizeof( int ) * N, cudaMemcpyDeviceToHost ) );
  CUCHECK( cudaMemcpy( last, devLast, sizeof( int ) * ( s + 1 ), cudaMemcpyDeviceToHost ) );

  listTimer.measureTime( "Copy sublists to Host" );


  int tmpSum = 0;
  int current = head;
  int currentSublist = 0;
  for ( int i = 0; i <= s; i++ )
  {
    tmpSum += sum[currentSublist];
    sum[currentSublist] = tmpSum - sum[currentSublist];

    current = last[currentSublist];
    currentSublist = -next[current] - 1;
  }


  CUCHECK( cudaMemcpy( devSum, sum, sizeof( int ) * ( s + 1 ), cudaMemcpyHostToDevice ) );

  listTimer.measureTime( "CPU adding sums" );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        int sublistId = devSublistId[thid];
        devRank[thid] += devSum[sublistId];
      },
      N,
      context );
  context.synchronize();

  listTimer.measureTime( "GPU final rank" );

  delete[] next;
  delete[] sum;
  delete[] last;
  // cudaFreeHost( next );
  // cudaFreeHost( sum );
  // cudaFreeHost( last );

  CUCHECK( cudaFree( devSum ) );
  CUCHECK( cudaFree( devSublistHead ) );
  CUCHECK( cudaFree( devSublistId ) );
  CUCHECK( cudaFree( devLast ) );

  listTimer.measureTime( "Free moemory" );
  listTimer.setPrefix( "" );
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
__device__ int cuAbs( int i )
{
  return i < 0 ? -i : i;
}