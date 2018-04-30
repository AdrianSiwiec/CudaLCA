#include <cuda_runtime.h>
#include <iostream>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>
#include "commons.h"
#include "cudaCommons.h"

using namespace std;
using namespace mgpu;

__device__ int CudaGetEdgeStart( int *father, int edgeCode );
__device__ int CudaGetEdgeEnd( int *father, int edgeCode );
__device__ int CudaGetEdgeCode( int a, bool toFather );
__device__ bool isEdgeToFather( int edgeCode );


int main( int argc, char *argv[] )
{
  // cudaSetDevice( 1 );
  Timer timer( "Parse Input" );

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

  timer.measureTime( "Read" );

  NextEdgeTree path( tc.tree );  // TODO: clear it up, scratch NextEdgeTree

  timer.measureTime( "Generate Next Edge Tree" );
  timer.setPrefix( "Preprocessing" );

  const int V = tc.tree.V;
  const int root = tc.tree.root;

  int *devNextEdge;
  int *devEdgeRank;

  CUCHECK( cudaMalloc( (void **) &devNextEdge, sizeof( int ) * V * 2 ) );
  CUCHECK( cudaMalloc( (void **) &devEdgeRank, sizeof( int ) * V * 2 ) );

  timer.measureTime( "Allocs" );

  CUCHECK( cudaMemcpy( devNextEdge, path.next.data(), sizeof( int ) * V * 2, cudaMemcpyHostToDevice ) );


  transform( [=] MGPU_DEVICE( int thid ) { devEdgeRank[thid] = 0; }, V * 2, context );
  context.synchronize();

  timer.measureTime( "Copy Input to Dev and Init data" );

  // int threadsPerBlockX = 1024;
  //int blocksPerGridX = ( V * 2 + threadsPerBlockX - 1 ) / threadsPerBlockX;
  // CudaSimpleListRank( devEdgeRank, V * 2, devNextEdge, threadsPerBlockX, blocksPerGridX );
  CudaFastListRank( devEdgeRank, V * 2, path.firstEdge, devNextEdge, context );

  timer.measureTime( "List Rank" );

  int *edgeRank = new int[V * 2];
  CUCHECK( cudaMemcpy( edgeRank, devEdgeRank, sizeof( int ) * V * 2, cudaMemcpyDeviceToHost ) );

  timer.measureTime( "Copy Ranks to Host" );

  int *devSortedEdges;

  int E = V * 2 - 2;

  CUCHECK( cudaMalloc( (void **) &devSortedEdges, sizeof( int ) * E ) );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        // int edgeRank = E - devEdgeRank[thid];
        int edgeRank = devEdgeRank[thid] - 1;

        devEdgeRank[thid] = edgeRank;

        if ( edgeRank == -1 || edgeRank == V * 2 - 1 ) return;  // edges from root

        devSortedEdges[edgeRank] = thid;
      },
      V * 2,
      context );
  context.synchronize();

  int *sortedEdges = (int *) malloc( sizeof( int ) * E );
  CUCHECK( cudaMemcpy( sortedEdges, devSortedEdges, sizeof( int ) * E, cudaMemcpyDeviceToHost ) );

  int *devW1;
  int *devW2;
  int *devW1Sum;
  int *devW2Sum;

  int *devFather;

  CUCHECK( cudaMalloc( (void **) &devW1, sizeof( int ) * E ) );
  CUCHECK( cudaMalloc( (void **) &devW1Sum, sizeof( int ) * E ) );
  CUCHECK( cudaMalloc( (void **) &devW2, sizeof( int ) * E ) );
  CUCHECK( cudaMalloc( (void **) &devW2Sum, sizeof( int ) * E ) );
  CUCHECK( cudaMalloc( (void **) &devFather, sizeof( int ) * V ) );

  CUCHECK( cudaMemcpy( devFather, tc.tree.father.data(), sizeof( int ) * V, cudaMemcpyHostToDevice ) );

  timer.measureTime( "Inlabel allocs" );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        int edge = devSortedEdges[thid];
        if ( isEdgeToFather( edge ) )
        {
          devW1[thid] = 0;
          devW2[thid] = -1;
        }
        else
        {
          devW1[thid] = 1;
          devW2[thid] = 1;
        }
      },
      E,
      context );

  scan<scan_type_inc>( devW1, E, devW1Sum, context );
  scan<scan_type_inc>( devW2, E, devW2Sum, context );

  timer.measureTime( "W1 W2 scans" );

  int *devPreorder;
  int *devPrePlusSize;
  int *devLevel;

  CUCHECK( cudaMalloc( (void **) &devPreorder, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devPrePlusSize, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devLevel, sizeof( int ) * V ) );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        int codeFromFather = CudaGetEdgeCode( thid, 0 );
        int codeToFather = CudaGetEdgeCode( thid, 1 );
        if ( thid == root )
        {
          devPreorder[thid] = 1;
          devPrePlusSize[thid] = V;
          devLevel[thid] = 0;
          return;
        }
        devPreorder[thid] = devW1Sum[devEdgeRank[codeFromFather]] + 1;
        devPrePlusSize[thid] = devW1Sum[devEdgeRank[codeToFather]] + 1;
        devLevel[thid] = devW2Sum[devEdgeRank[codeFromFather]];
      },
      V,
      context );

  context.synchronize();

  timer.measureTime( "Pre PrePlusSize, Level" );

  int *devInlabel;

  CUCHECK( cudaMalloc( (void **) &devInlabel, sizeof( int ) * V ) );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        int i = 31 - __clz( ( devPreorder[thid] - 1 ) ^ ( devPrePlusSize[thid] ) );
        devInlabel[thid] = ( ( devPrePlusSize[thid] ) >> i ) << i;
      },
      V,
      context );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        if ( thid == root ) return;
        int f = devFather[thid];
        int inLabel = devInlabel[thid];
        int fatherInLabel = devInlabel[f];
        if ( inLabel != fatherInLabel )
        {
          int i = __ffs( inLabel ) - 1;
          devW1[devEdgeRank[CudaGetEdgeCode( thid, 0 )]] = ( 1 << i );
          devW1[devEdgeRank[CudaGetEdgeCode( thid, 1 )]] = -( 1 << i );
        }
        else
        {
          devW1[devEdgeRank[CudaGetEdgeCode( thid, 0 )]] = 0;
          devW1[devEdgeRank[CudaGetEdgeCode( thid, 1 )]] = 0;
        }
      },
      V,
      context );

  scan<scan_type_inc>( devW1, E, devW1Sum, context );

  int l = 31 - __builtin_clz( V );

  int *devAscendant;
  CUCHECK( cudaMalloc( (void **) &devAscendant, sizeof( int ) * V ) );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        if ( thid == root )
        {
          devAscendant[thid] = ( 1 << l );
          return;
        }
        devAscendant[thid] = ( 1 << l ) + devW1Sum[devEdgeRank[CudaGetEdgeCode( thid, 0 )]];
      },
      V,
      context );

  timer.measureTime( "Ascendant scan and calculation" );

  int *devHead;
  CUCHECK( cudaMalloc( (void **) &devHead, sizeof( int ) * ( V + 1 ) ) );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        if ( thid == root || devInlabel[thid] != devInlabel[devFather[thid]] )
        {
          devHead[devInlabel[thid]] = thid;
        }
      },
      V,
      context );

  context.synchronize();

  timer.measureTime( "Head" );
  timer.setPrefix( "Queries" );

  int Q = tc.q.N;

  int *devQueries;
  CUCHECK( cudaMalloc( (void **) &devQueries, sizeof( int ) * Q * 2 ) );
  CUCHECK( cudaMemcpy( devQueries, tc.q.tab.data(), sizeof( int ) * Q * 2, cudaMemcpyHostToDevice ) );

  int *devAnswers;
  CUCHECK( cudaMalloc( (void **) &devAnswers, sizeof( int ) * Q ) );

  timer.measureTime( "Allocs and copy" );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        int x = devQueries[thid * 2];
        int y = devQueries[thid * 2 + 1];

        if ( devInlabel[x] == devInlabel[y] )
        {
          devAnswers[thid] = devLevel[x] < devLevel[y] ? x : y;
          return;
        }
        int i = 31 - __clz( devInlabel[x] ^ devInlabel[y] );

        int common = devAscendant[x] & devAscendant[y];
        common = ( ( common >> i ) << i );

        int j = __ffs( common ) - 1;

        int inlabelZ = ( devInlabel[y] >> ( j ) ) << ( j );
        inlabelZ |= ( 1 << j );

        int suspects[2];

        for ( int a = 0; a < 2; a++ )
        {
          int tmpX;
          if ( a == 0 )
            tmpX = x;
          else
            tmpX = y;

          if ( devInlabel[tmpX] == inlabelZ )
          {
            suspects[a] = tmpX;
          }
          else
          {
            int k = 31 - __clz( devAscendant[tmpX] & ( ( 1 << j ) - 1 ) );

            int inlabelW = ( devInlabel[tmpX] >> ( k ) ) << ( k );
            inlabelW |= ( 1 << k );

            int w = devHead[inlabelW];
            suspects[a] = devFather[w];
          }
        }

        if ( devLevel[suspects[0]] < devLevel[suspects[1]] )
        {
          devAnswers[thid] = suspects[0];
        }
        else
        {
          devAnswers[thid] = suspects[1];
        }
      },
      Q,
      context );

  context.synchronize();

  timer.measureTime( Q );

  int *answers = (int *) malloc( sizeof( int ) * Q );

  CUCHECK( cudaMemcpy( answers, devAnswers, sizeof( int ) * Q, cudaMemcpyDeviceToHost ) );

  timer.measureTime( "Copy to Host" );
  timer.setPrefix( "Write Output" );

  if ( argc < 3 )
  {
    writeAnswersToStdOut( Q, answers );
  }
  else
  {
    writeAnswersToFile( Q, answers, argv[2] );
  }

  timer.setPrefix("");
}

__device__ int CudaGetEdgeStart( int *father, int edgeCode )
{
  if ( edgeCode % 2 )
    return edgeCode / 2;
  else
    return father[edgeCode / 2];
}

__device__ int CudaGetEdgeEnd( int *father, int edgeCode )
{
  return CudaGetEdgeStart( father, edgeCode ^ 1 );
}
__device__ bool isEdgeToFather( int edgeCode )
{
  return edgeCode % 2;
}
__device__ int CudaGetEdgeCode( int a, bool toFather )
{
  return a * 2 + toFather;
}