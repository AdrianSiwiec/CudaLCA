#include <iostream>
#include "../common/cuda.cpp"
#include "../common/time.cpp"
#include "cuda.h"

using namespace std;

void testRes( CUresult res, string msg );

int main()
{
  ios_base::sync_with_stdio( 0 );

  measureTime();

  CUmodule cuModule = cudaInit( "bruteLca.ptx" );

  CUfunction cuInit = cudaGetFunction( cuModule, "cuInit" );
  CUfunction cuCalcDepthRead = cudaGetFunction( cuModule, "cuCalcDepthRead" );
  CUfunction cuCalcDepthWrite = cudaGetFunction( cuModule, "cuCalcDepthWrite" );
  CUfunction cuMoveNextRead = cudaGetFunction( cuModule, "cuMoveNextRead" );
  CUfunction cuMoveNextWrite = cudaGetFunction( cuModule, "cuMoveNextWrite" );
  CUfunction cuCalcQueries = cudaGetFunction( cuModule, "cuCalcQueries" );

  measureTime( "Cuda Init" );

  CUdeviceptr devFather;
  CUdeviceptr devDepth;
  CUdeviceptr devNext;
  CUdeviceptr devNotAllDone;
  CUdeviceptr devQueries;
  CUdeviceptr devAnswers;
  CUdeviceptr devTmp;

  int V;
  cin >> V;
  int *father = (int *) malloc( sizeof( int ) * V );


  cudaMemAlloc( &devFather, sizeof( int ) * V, "devFather" );
  cudaMemAlloc( &devDepth, sizeof( int ) * V, "devDepth" );
  cudaMemAlloc( &devNext, sizeof( int ) * V, "devNext" );
  cudaMemAlloc( &devNotAllDone, sizeof( int ), "devNotAllDone" );
  cudaMemAlloc( &devTmp, sizeof( int ) * V, "devTmp" );

  measureTime( "Cuda Allocs" );

  for ( int i = 1; i < V; i++ )
  {
    cin >> father[i];
  }
  father[0] = -1;

  measureTime( "Read Input" );

  CUresult res = cuMemHostRegister( father, sizeof( int ) * V, 0 );
  testRes( res, "Mem Register father" );
  res = cuMemcpyHtoD( devFather, father, sizeof( int ) * V );
  testRes( res, "Copy Father to dev" );

  void *argsCalcDepth[] = {&V, &devFather, &devNext, &devDepth, &devTmp, &devNotAllDone};

  int threadsPerBlockX = 1024;
  int blockPerGridX = ( V + threadsPerBlockX - 1 ) / threadsPerBlockX;

  cudaLaunchKernel( cuInit, blockPerGridX, threadsPerBlockX, argsCalcDepth, "cuInit" );

  measureTime( "Copy Input and Init data" );

  int *notAllDone = (int *) malloc( sizeof( int ) );
  do
  {
    cudaLaunchKernel( cuCalcDepthRead, blockPerGridX, threadsPerBlockX, argsCalcDepth, "cuCalcDepthRead" );
    cudaLaunchKernel( cuCalcDepthWrite, blockPerGridX, threadsPerBlockX, argsCalcDepth, "cuCalcDepthWrite" );
    cudaLaunchKernel( cuMoveNextRead, blockPerGridX, threadsPerBlockX, argsCalcDepth, "cuMoveNextRead" );
    cudaLaunchKernel( cuMoveNextWrite, blockPerGridX, threadsPerBlockX, argsCalcDepth, "cuMoveNextWrite" );

    res = cuMemcpyDtoH( notAllDone, devNotAllDone, sizeof( int ) );
    testRes( res, "Copy devNotAllDone to host" );

  } while ( *notAllDone );

  measureTime( "Cuda Preprocessing" );

  // int *depth = (int *) malloc( sizeof( int ) * V );

  // res = cuMemcpyDtoH( depth, devDepth, sizeof( int ) * V );
  // testRes( res, "Copy devDepth to host" );

  //   for ( int i = 0; i < V; i++ )
  //   {
  //     cout << i << ": " << depth[i] << endl;
  //   }

  int Q;
  cin >> Q;
  int *queries = (int *) malloc( sizeof( int ) * Q * 2 );
  for ( int i = 0; i < Q; i++ )
  {
    cin >> queries[2 * i] >> queries[2 * i + 1];
  }

  measureTime( "Read queries" );

  res = cuMemHostRegister( queries, sizeof( int ) * Q * 2, 0 );
  testRes( res, "Mem Register queries" );

  cudaMemAlloc( &devQueries, sizeof( int ) * Q * 2 );
  cudaMemAlloc( &devAnswers, sizeof( int ) * Q );

  res = cuMemcpyHtoD( devQueries, queries, sizeof( int ) * Q * 2 );
  testRes( res, "Copy queries to dev" );

  measureTime( "Copy Queries to Dev" );

  blockPerGridX = ( Q + threadsPerBlockX - 1 ) / threadsPerBlockX;
  void *argsCalcQueries[] = {&Q, &devFather, &devDepth, &devQueries, &devAnswers};

  cudaLaunchKernel( cuCalcQueries, blockPerGridX, threadsPerBlockX, argsCalcQueries, "cuCalcQueries" );

  measureTime( "Cuda calc queries" );

  int *answers = (int *) malloc( sizeof( int ) * Q );
  res = cuMemHostRegister( answers, sizeof( int ) * Q, 0 );
  testRes( res, "Mem Register answers" );

  res = cuMemcpyDtoH( answers, devAnswers, sizeof( int ) * Q );
  testRes( res, "Copy devAnswers to host" );

  measureTime( "Copy answers to Host" );

  for ( int i = 0; i < Q; i++ )
  {
    cout << answers[i] << endl;
  }

  measureTime( "Write Output" );
}