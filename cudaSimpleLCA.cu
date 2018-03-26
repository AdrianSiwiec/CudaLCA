#include <iostream>
#include "commons.h"
#include "cudaCommons.h"

using namespace std;

__global__ void cuInit( int V, int *father, int *next, int *depth );
__global__ void cuCalcDepthRead( int V, int *next, int *depth, int *tmp, int *notAllDone );
__global__ void cuCalcDepthWrite( int V, int *next, int *depth, int *tmp );
__global__ void cuMoveNextRead( int V, int *next, int *tmp );
__global__ void cuMoveNextWrite( int V, int *next, int *tmp, int *notAllDone );
__global__ void cuCalcQueries( int Q, int *father, int *depth, int *queries, int *answers );

int main( int argc, char *argv[] )
{
  Timer timer = Timer();

  TestCase tc;
  if ( argc == 1 )
  {
    tc = readFromStdIn();
  }
  else
  {
    tc = readFromFile( argv[1] );
  }

  timer.measureTime( "Read Input" );

  int *devFather;
  int *devDepth;
  int *devNext;
  int *devNotAllDone;
  int *devQueries;
  int *devAnswers;
  int *devTmp;

  const int V = tc.tree.V;

  CUCHECK( cudaMalloc( (void **) &devFather, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devDepth, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devNext, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devTmp, sizeof( int ) * V ) );

  CUCHECK( cudaMalloc( (void **) &devNotAllDone, sizeof( int ) ) );

  timer.measureTime( "Cuda Allocs" );

  CUCHECK( cudaMemcpy( devFather, tc.tree.father.data(), sizeof( int ) * V, cudaMemcpyHostToDevice ) );

  int threadsPerBlockX = 1024;
  int blockPerGridX = ( V + threadsPerBlockX - 1 ) / threadsPerBlockX;

  cuInit<<<blockPerGridX, threadsPerBlockX>>>( V, devFather, devNext, devDepth );
  CUCHECK( cudaDeviceSynchronize() );

  timer.measureTime( "Copy Input and Init data" );

  int *notAllDone = (int *) malloc( sizeof( int ) );
  do
  {
    cuCalcDepthRead<<<blockPerGridX, threadsPerBlockX>>>( V, devNext, devDepth, devTmp, devNotAllDone );
    CUCHECK( cudaDeviceSynchronize() );

    cuCalcDepthWrite<<<blockPerGridX, threadsPerBlockX>>>( V, devNext, devDepth, devTmp );
    CUCHECK( cudaDeviceSynchronize() );

    cuMoveNextRead<<<blockPerGridX, threadsPerBlockX>>>( V, devNext, devTmp );
    CUCHECK( cudaDeviceSynchronize() );

    cuMoveNextWrite<<<blockPerGridX, threadsPerBlockX>>>( V, devNext, devTmp, devNotAllDone );
    CUCHECK( cudaDeviceSynchronize() );

    CUCHECK( cudaMemcpy( notAllDone, devNotAllDone, sizeof( int ), cudaMemcpyDeviceToHost ) );

  } while ( *notAllDone );

  timer.measureTime( "Cuda Preprocessing" );

  // int *depth = (int *) malloc( sizeof( int ) * V );

  // res = cuMemcpyDtoH( depth, devDepth, sizeof( int ) * V );
  // testRes( res, "Copy devDepth to host" );

  //   for ( int i = 0; i < V; i++ )
  //   {
  //     cout << i << ": " << depth[i] << endl;
  //   }

  int Q = tc.q.N;

  CUCHECK( cudaMalloc( (void **) &devQueries, sizeof( int ) * Q * 2 ) );
  CUCHECK( cudaMalloc( (void **) &devAnswers, sizeof( int ) * Q ) );

  CUCHECK( cudaMemcpy( devQueries, tc.q.tab.data(), sizeof( int ) * Q * 2, cudaMemcpyHostToDevice ) );

  timer.measureTime( "Copy Queries to Dev" );

  blockPerGridX = ( Q + threadsPerBlockX - 1 ) / threadsPerBlockX;

  cuCalcQueries<<<blockPerGridX, threadsPerBlockX>>>( Q, devFather, devDepth, devQueries, devAnswers );
  CUCHECK( cudaDeviceSynchronize() );

  timer.measureTime( "Cuda calc queries" );

  int *answers = (int *) malloc( sizeof( int ) * Q );

  CUCHECK( cudaMemcpy( answers, devAnswers, sizeof( int ) * Q, cudaMemcpyDeviceToHost ) );

  timer.measureTime( "Copy answers to Host" );

  if ( argc < 3 )
  {
    writeAnswersToStdOut( Q, answers );
  }
  else
  {
    writeAnswersToFile( Q, answers, argv[2] );
  }

  timer.measureTime( "Write Output" );
}

__global__ void cuInit( int V, int *father, int *next, int *depth )
{
  int thid = ( blockIdx.x * blockDim.x ) + threadIdx.x;

  if ( thid >= V ) return;

  next[thid] = father[thid];
  depth[thid] = 0;
}

__global__ void cuCalcDepthRead( int V, int *next, int *depth, int *tmp, int *notAllDone )
{
  int thid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
  if ( thid == 0 ) *notAllDone = 0;

  if ( thid >= V || next[thid] == -1 ) return;

  tmp[thid] = depth[next[thid]] + 1;
}

__global__ void cuCalcDepthWrite( int V, int *next, int *depth, int *tmp )
{
  int thid = ( blockIdx.x * blockDim.x ) + threadIdx.x;

  if ( thid >= V || next[thid] == -1 ) return;

  depth[thid] += tmp[thid];
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

__global__ void cuCalcQueries( int Q, int *father, int *depth, int *queries, int *answers )
{
  int thid = ( blockIdx.x * blockDim.x ) + threadIdx.x;

  if ( thid >= Q ) return;

  int p = queries[thid * 2];
  int q = queries[thid * 2 + 1];

  if ( p == q ) answers[thid] = p;

  while ( depth[p] != depth[q] )
  {
    if ( depth[p] > depth[q] )
      p = father[p];
    else
      q = father[q];
  }

  while ( p != q )
  {
    p = father[p];
    q = father[q];
  }

  answers[thid] = p;
}