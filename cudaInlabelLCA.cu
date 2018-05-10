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
  Timer timer( "Parse Input" );

  cudaSetDevice( 1 );
  standard_context_t context( 0 );

  timer.measureTime( "Init cuda" );

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
  timer.setPrefix( "Preprocessing" );

  const int V = tc.tree.V;
  const int root = tc.tree.root;

  int *devSon;
  int *devNeighbour;
  int *devFather;
  int *devNextEdge;
  CUCHECK( cudaMalloc( (void **) &devSon, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devNeighbour, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devFather, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devNextEdge, sizeof( int ) * V * 2 ) );

  timer.measureTime( "Device Allocs" );

  CUCHECK( cudaMemcpy( devSon, tc.tree.son.data(), sizeof( int ) * V, cudaMemcpyHostToDevice ) );
  CUCHECK( cudaMemcpy( devNeighbour, tc.tree.neighbour.data(), sizeof( int ) * V, cudaMemcpyHostToDevice ) );
  CUCHECK( cudaMemcpy( devFather, tc.tree.father.data(), sizeof( int ) * V, cudaMemcpyHostToDevice ) );

  timer.measureTime( "Device Memcpy" );

  transform(
      [=] MGPU_DEVICE( int thid ) {
        int v = thid / 2;
        int father = devFather[v];
        if ( isEdgeToFather( thid ) )
        {
          int neighbour = devNeighbour[v];
          if ( neighbour != -1 )
            devNextEdge[thid] = CudaGetEdgeCode( neighbour, false );
          else
          {
            if ( father != -1 )
              devNextEdge[thid] = CudaGetEdgeCode( father, true );
            else
              devNextEdge[thid] = -1;
          }
        }
        else
        {
          int son = devSon[v];
          if ( son != -1 )
            devNextEdge[thid] = CudaGetEdgeCode( son, false );
          else
            devNextEdge[thid] = CudaGetEdgeCode( v, true );
        }
      },
      V * 2,
      context );
  context.synchronize();

  CUCHECK( cudaFree( devSon ) );
  CUCHECK( cudaFree( devNeighbour ) );

  int *devEdgeRank;
  CUCHECK( cudaMalloc( (void **) &devEdgeRank, sizeof( int ) * V * 2 ) );

  transform( [=] MGPU_DEVICE( int thid ) { devEdgeRank[thid] = 0; }, V * 2, context );
  context.synchronize();

  timer.measureTime( "Init devNextEdge and devEdgeRank" );

  // int threadsPerBlockX = 1024;
  // int blocksPerGridX = ( V * 2 + threadsPerBlockX - 1 ) / threadsPerBlockX;
  // CudaSimpleListRank( devEdgeRank, V * 2, devNextEdge, threadsPerBlockX, blocksPerGridX );

  CudaFastListRank( devEdgeRank, V * 2, getEdgeCode( root, 0 ), devNextEdge, context );

  CUCHECK( cudaFree( devNextEdge ) );

  timer.measureTime( "List Rank" );

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

  int *devW1;
  int *devW2;
  int *devW1Sum;
  int *devW2Sum;


  CUCHECK( cudaMalloc( (void **) &devW1, sizeof( int ) * E ) );
  CUCHECK( cudaMalloc( (void **) &devW1Sum, sizeof( int ) * E ) );
  CUCHECK( cudaMalloc( (void **) &devW2, sizeof( int ) * E ) );
  CUCHECK( cudaMalloc( (void **) &devW2Sum, sizeof( int ) * E ) );


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

  context.synchronize();

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

        int edgeRankFromFather = devEdgeRank[codeFromFather];
        devPreorder[thid] = devW1Sum[edgeRankFromFather] + 1;
        devPrePlusSize[thid] = devW1Sum[devEdgeRank[codeToFather]] + 1;
        devLevel[thid] = devW2Sum[edgeRankFromFather];
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

  context.synchronize();

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

        int inlabelX = devInlabel[x];
        int inlabelY = devInlabel[y];

        if ( inlabelX == inlabelY )
        {
          devAnswers[thid] = devLevel[x] < devLevel[y] ? x : y;
          return;
        }
        int i = 31 - __clz( inlabelX ^ inlabelY );

        int common = devAscendant[x] & devAscendant[y];
        common = ( ( common >> i ) << i );

        int j = __ffs( common ) - 1;

        int inlabelZ = ( inlabelY >> ( j ) ) << ( j );
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

            int inlabelW = ( devInlabel[tmpX] >> k ) << ( k );
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

  timer.setPrefix( "" );
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