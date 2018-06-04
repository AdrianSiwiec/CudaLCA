#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <iostream>
#include <moderngpu/transform.hxx>
#include "commons.h"
#include "cudaCommons.h"

#define ull unsigned long long

using namespace std;
using namespace mgpu;

__device__ int cuAbs( int i );

void CudaSimpleListRank( int *devRank, int N, int *devNext, standard_context_t &context )
{
  int *notAllDone;
  cudaMallocHost( &notAllDone, sizeof( int ) );

  ull *devRankNext;
  int *devNotAllDone;

  CUCHECK( cudaMalloc( (void **) &devRankNext, sizeof( ull ) * N ) );
  CUCHECK( cudaMalloc( (void **) &devNotAllDone, sizeof( int ) ) );

  transform(
      [=] MGPU_DEVICE( int thid ) { devRankNext[thid] = ( ( (ull) 0 ) << 32 ) + devNext[thid] + 1; }, N, context );

  const int loopsWithoutSync = 10;

  do
  {
    transform(
        [=] MGPU_DEVICE( int thid ) {
          ull rankNext = devRankNext[thid];
          for ( int i = 0; i < loopsWithoutSync; i++ )
          {
            if ( thid == 0 ) *devNotAllDone = 0;

            int rank = rankNext >> 32;
            int nxt = rankNext - 1;

            if ( nxt != -1 )
            {
              ull grandNext = devRankNext[nxt];

              rank += ( grandNext >> 32 ) + 1;
              nxt = grandNext - 1;

              rankNext = ( ( (ull) rank ) << 32 ) + nxt + 1;
              atomicExch( devRankNext + thid, rankNext );

              if ( i == loopsWithoutSync - 1 ) *devNotAllDone = 1;
            }
          }
        },
        N,
        context );

    context.synchronize();

    CUCHECK( cudaMemcpy( notAllDone, devNotAllDone, sizeof( int ), cudaMemcpyDeviceToHost ) );
  } while ( *notAllDone );

  transform( [=] MGPU_DEVICE( int thid ) { devRank[thid] = devRankNext[thid] >> 32; }, N, context );
  context.synchronize();

  cudaFree( notAllDone );
  cudaFree( devRankNext );
  CUCHECK( cudaFree( devNotAllDone ) );
}

void CudaFastListRank( int *devRank, int N, int head, int *devNext, standard_context_t &context )
{
  Timer listTimer( "List Rank" );
  int s;
  if ( N > 1000000 )
  {
    double tmp = N;
    tmp /= log2( N );
    tmp /= log2( N );
    s = tmp;
  }
  else
    s = N / 100;
  if ( s == 0 ) s = 1;

  cerr << s << endl;


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

  context.synchronize();
  listTimer.measureTime( "GPU generating splitters" );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        int current = devSublistHead[thid];
        int counter = 0;

        while ( current >= 0 )
        {
          devRank[current] = counter++;

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

  transform(
      [=] MGPU_DEVICE( int thid ) {
        int tmpSum = 0;
        int current = head;
        int currentSublist = 0;
        for ( int i = 0; i <= s; i++ )
        {
          tmpSum += devSum[currentSublist];
          devSum[currentSublist] = tmpSum - devSum[currentSublist];

          current = devLast[currentSublist];
          currentSublist = -devNext[current] - 1;
        }
      },
      1,
      context );

  context.synchronize();
  listTimer.measureTime( "GPU Adding Times" );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        int sublistId = devSublistId[thid];
        devRank[thid] += devSum[sublistId];
      },
      N,
      context );
  context.synchronize();

  listTimer.measureTime( "GPU final rank" );

  CUCHECK( cudaFree( devSum ) );
  CUCHECK( cudaFree( devSublistHead ) );
  CUCHECK( cudaFree( devSublistId ) );
  CUCHECK( cudaFree( devLast ) );

  listTimer.measureTime( "Free moemory" );
  listTimer.setPrefix( "" );
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