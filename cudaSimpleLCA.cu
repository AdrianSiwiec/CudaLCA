#include <cuda_runtime.h>
#include <iostream>
#include <moderngpu/transform.hxx>
#include "commons.h"
#include "cudaCommons.h"

using namespace std;
using namespace mgpu;

__global__ void cuCalcQueries( int Q, int *father, int *depth, int *queries, int *answers );

int main( int argc, char *argv[] )
{
  Timer timer = Timer( "Parse Input" );

  standard_context_t context( 0 );

  TestCase tc;
  if ( argc == 1 )
  {
    tc = readFromStdIn();
  }
  else
  {
    tc = readFromFile( argv[1] );
  }

  timer.setPrefix( "Preprocessing" );

  int *devFather;
  int *devDepth;
  int *devNext;
  int *devQueries;
  int *devAnswers;

  const int V = tc.tree.V;

  CUCHECK( cudaMalloc( (void **) &devFather, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devDepth, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devNext, sizeof( int ) * V ) );

  timer.measureTime( "Cuda Allocs" );

  CUCHECK( cudaMemcpy( devFather, tc.tree.father.data(), sizeof( int ) * V, cudaMemcpyHostToDevice ) );

  int threadsPerBlockX = 1024;
  int blockPerGridX = ( V + threadsPerBlockX - 1 ) / threadsPerBlockX;

  transform(
      [=] MGPU_DEVICE( int thid ) {
        devNext[thid] = devFather[thid];
        devDepth[thid] = 0;
      },
      V,
      context );

  context.synchronize();

  timer.measureTime( "Copy Input and Init data" );

  CudaSimpleListRank( devDepth, V, devNext, threadsPerBlockX, blockPerGridX, context );

  timer.setPrefix( "Queries" );

  int Q = tc.q.N;

  CUCHECK( cudaMalloc( (void **) &devQueries, sizeof( int ) * Q * 2 ) );
  CUCHECK( cudaMalloc( (void **) &devAnswers, sizeof( int ) * Q ) );

  CUCHECK( cudaMemcpy( devQueries, tc.q.tab.data(), sizeof( int ) * Q * 2, cudaMemcpyHostToDevice ) );

  timer.measureTime( "Copy Queries to Dev" );

  blockPerGridX = ( Q + threadsPerBlockX - 1 ) / threadsPerBlockX;

  cuCalcQueries<<<blockPerGridX, threadsPerBlockX>>>( Q, devFather, devDepth, devQueries, devAnswers );
  CUCHECK( cudaDeviceSynchronize() );

  timer.measureTime( Q );

  int *answers = (int *) malloc( sizeof( int ) * Q );

  CUCHECK( cudaMemcpy( answers, devAnswers, sizeof( int ) * Q, cudaMemcpyDeviceToHost ) );

  timer.measureTime( "Copy answers to Host" );
  timer.setPrefix( "Write Output" );

  if ( argc < 3 )
  {
    writeAnswersToStdOut( Q, answers );
  }
  else
  {
    writeAnswersToFile( Q, answers, argv[2] );
  }

  timer.setPrefix( "" );
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