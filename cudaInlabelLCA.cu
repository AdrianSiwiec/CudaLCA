#include <cuda_runtime.h>
#include <iostream>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>
#include "commons.h"
#include "cudaCommons.h"
#include <cudaWeiJaJaListRank.h>

#define ll long long

using namespace std;
using namespace mgpu;

__device__ int CudaGetEdgeStart( const int *__restrict__ father, int edgeCode );
__device__ int CudaGetEdgeEnd( const int *__restrict__ father, int edgeCode );
__device__ int CudaGetEdgeCode( int a, bool toFather );
__device__ bool isEdgeToFather( int edgeCode );

const int measureTimeDebug = false;


int main( int argc, char *argv[] )
{
  Timer timer( "Parse Input" );

  cudaSetDevice( 0 );
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

  int batchSize;
  if ( argc > 3 )
    batchSize = atoi( argv[3] );
  else
    batchSize = -1;

  timer.measureTime( "Read" );
  timer.setPrefix( "Preprocessing" );

  const int V = tc.tree.V;
  const int root = tc.tree.root;

  int *devFather;
  CUCHECK( cudaMalloc( (void **) &devFather, sizeof( int ) * V ) );
  CUCHECK( cudaMemcpy( devFather, tc.tree.father.data(), sizeof( int ) * V, cudaMemcpyHostToDevice ) );

  int *devSon;
  int *devNeighbour;
  int *devNextEdge;
  CUCHECK( cudaMalloc( (void **) &devSon, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devNeighbour, sizeof( int ) * V ) );

  transform(
      [] MGPU_DEVICE( int thid, int *devSon, int *devNeighbour ) {
        devSon[thid] = -1;
        devNeighbour[thid] = -1;
      },
      V,
      context,
      devSon,
      devNeighbour );

  if ( measureTimeDebug )
  {
    context.synchronize();
    timer.measureTime( "Device Allocs" );
  }

  ll *devEdges;
  CUCHECK( cudaMalloc( (void **) &devEdges, sizeof( ll ) * V ) );

  transform(
      [] MGPU_DEVICE( int thid, ll *devEdges, const int *devFather ) {
        devEdges[thid] = ( ( (ll) devFather[thid] ) << 32 ) + thid;
      },
      V,
      context,
      devEdges,
      devFather );

  mergesort( devEdges, V, [] MGPU_DEVICE( ll a, ll b ) { return a < b; }, context );

  transform(
      [] MGPU_DEVICE( int thid, const ll *devEdges, int *devNeighbour, int *devSon ) {
        ll prevEdge = devEdges[thid];
        ll myEdge = devEdges[thid + 1];
        if ( prevEdge >> 32 == myEdge >> 32 )
          devNeighbour[(int) prevEdge] = (int) myEdge;
        else
          devSon[myEdge >> 32] = (int) myEdge;
      },
      V - 1,
      context,
      devEdges,
      devNeighbour,
      devSon );

  CUCHECK( cudaFree( devEdges ) );


  CUCHECK( cudaMalloc( (void **) &devNextEdge, sizeof( int ) * V * 2 ) );
  transform(
      [] MGPU_DEVICE( int thid, const int *devFather, const int *devNeighbour, const int *devSon, int *devNextEdge ) {
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
      context,
      devFather,
      devNeighbour,
      devSon,
      devNextEdge );

  CUCHECK( cudaFree( devSon ) );
  CUCHECK( cudaFree( devNeighbour ) );

  int *devEdgeRank;
  CUCHECK( cudaMalloc( (void **) &devEdgeRank, sizeof( int ) * V * 2 ) );

  CUCHECK( cudaMemset( devEdgeRank, 0, V * 2 ) );

  if ( measureTimeDebug )
  {
    context.synchronize();
    timer.measureTime( "Init devNextEdge and devEdgeRank" );
  }

  // CudaSimpleListRank( devEdgeRank, V * 2, devNextEdge, context);
  cudaWeiJaJaListRank( devEdgeRank, V * 2, getEdgeCode( root, 0 ), devNextEdge, context );

  CUCHECK( cudaFree( devNextEdge ) );

  timer.measureTime( "List Rank" );

  int *devSortedEdges;
  int E = V * 2 - 2;

  CUCHECK( cudaMalloc( (void **) &devSortedEdges, sizeof( int ) * E ) );

  transform(
      [] MGPU_DEVICE( int thid, int V, int *devEdgeRank, int *devSortedEdges ) {
        // int edgeRank = E - devEdgeRank[thid];
        int edgeRank = devEdgeRank[thid] - 1;

        devEdgeRank[thid] = edgeRank;

        if ( edgeRank == -1 || edgeRank == V * 2 - 1 ) return;  // edges from root

        devSortedEdges[edgeRank] = thid;
      },
      V * 2,
      context,
      V,
      devEdgeRank,
      devSortedEdges );

  int *devW1;
  int *devW2;
  int *devW1Sum;
  int *devW2Sum;


  CUCHECK( cudaMalloc( (void **) &devW1, sizeof( int ) * E ) );

  cerr << sizeof( int ) * E << endl;

  CUCHECK( cudaMalloc( (void **) &devW1Sum, sizeof( int ) * E ) );
  CUCHECK( cudaMalloc( (void **) &devW2, sizeof( int ) * E ) );
  CUCHECK( cudaMalloc( (void **) &devW2Sum, sizeof( int ) * E ) );


  if ( measureTimeDebug )
  {
    context.synchronize();
    timer.measureTime( "Inlabel allocs" );
  }

  transform(
      [] MGPU_DEVICE( int thid, int *devW1, int *devW2, const int *devSortedEdges ) {
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
      context,
      devW1,
      devW2,
      devSortedEdges );


  context.synchronize();
  CUCHECK( cudaFree( devSortedEdges ) );

  scan<scan_type_inc>( devW1, E, devW1Sum, context );
  scan<scan_type_inc>( devW2, E, devW2Sum, context );
  CUCHECK( cudaFree( devW2 ) );

  if ( measureTimeDebug )
  {
    context.synchronize();
    timer.measureTime( "W1 W2 scans" );
  }

  int *devPreorder;
  int *devPrePlusSize;
  int *devLevel;

  CUCHECK( cudaMalloc( (void **) &devPreorder, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devPrePlusSize, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devLevel, sizeof( int ) * V ) );

  transform(
      [] MGPU_DEVICE( int thid,
                      int V,
                      int root,
                      int *devPreorder,
                      int *devPrePlusSize,
                      int *devLevel,
                      const int *devEdgeRank,
                      const int *devW1Sum,
                      const int *devW2Sum ) {
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
      context,
      V,
      root,
      devPreorder,
      devPrePlusSize,
      devLevel,
      devEdgeRank,
      devW1Sum,
      devW2Sum );

  CUCHECK( cudaFree( devW2Sum ) );

  if ( measureTimeDebug )
  {
    context.synchronize();
    timer.measureTime( "Pre PrePlusSize, Level" );
  }

  int *devInlabel;

  CUCHECK( cudaMalloc( (void **) &devInlabel, sizeof( int ) * V ) );

  transform(
      [] MGPU_DEVICE( int thid, int *devInlabel, const int *devPreorder, const int *devPrePlusSize ) {
        int i = 31 - __clz( ( devPreorder[thid] - 1 ) ^ ( devPrePlusSize[thid] ) );
        devInlabel[thid] = ( ( devPrePlusSize[thid] ) >> i ) << i;
      },
      V,
      context,
      devInlabel,
      devPreorder,
      devPrePlusSize );

  CUCHECK( cudaFree( devPreorder ) );
  CUCHECK( cudaFree( devPrePlusSize ) );

  transform(
      [] MGPU_DEVICE(
          int thid, int root, const int *devFather, const int *devInlabel, const int *devEdgeRank, int *devW1 ) {
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
      context,
      root,
      devFather,
      devInlabel,
      devEdgeRank,
      devW1 );

  scan<scan_type_inc>( devW1, E, devW1Sum, context );
  CUCHECK( cudaFree( devW1 ) );

  int l = 31 - __builtin_clz( V );

  int *devAscendant;
  CUCHECK( cudaMalloc( (void **) &devAscendant, sizeof( int ) * V ) );

  transform(
      [] MGPU_DEVICE( int thid, int root, int l, int *devAscendant, const int *devW1Sum, const int *devEdgeRank ) {
        if ( thid == root )
        {
          devAscendant[thid] = ( 1 << l );
          return;
        }
        devAscendant[thid] = ( 1 << l ) + devW1Sum[devEdgeRank[CudaGetEdgeCode( thid, 0 )]];
      },
      V,
      context,
      root,
      l,
      devAscendant,
      devW1Sum,
      devEdgeRank );

  CUCHECK( cudaFree( devEdgeRank ) );
  CUCHECK( cudaFree( devW1Sum ) );

  if ( measureTimeDebug )
  {
    context.synchronize();
    timer.measureTime( "Ascendant scan and calculation" );
  }

  int *devHead;
  CUCHECK( cudaMalloc( (void **) &devHead, sizeof( int ) * ( V + 1 ) ) );

  transform(
      [] MGPU_DEVICE( int thid, int root, const int *devInlabel, const int *devFather, int *devHead ) {
        if ( thid == root || devInlabel[thid] != devInlabel[devFather[thid]] )
        {
          devHead[devInlabel[thid]] = thid;
        }
      },
      V,
      context,
      root,
      devInlabel,
      devFather,
      devHead );

  context.synchronize();

  if ( measureTimeDebug ) timer.measureTime( "Head" );

  timer.setPrefix( "Queries" );

  int Q = tc.q.N;

  if ( batchSize == -1 ) batchSize = Q;

  int *devQueries;
  CUCHECK( cudaMalloc( (void **) &devQueries, sizeof( int ) * batchSize * 2 ) );

  int *answers = (int *) malloc( sizeof( int ) * Q );
  int *devAnswers;
  CUCHECK( cudaMalloc( (void **) &devAnswers, sizeof( int ) * batchSize ) );

  for ( int qStart = 0; qStart < Q; qStart += batchSize )
  {
    int queriesToProcess = min( batchSize, Q - qStart );

    CUCHECK( cudaMemcpy(
        devQueries, tc.q.tab.data() + ( qStart * 2 ), sizeof( int ) * queriesToProcess * 2, cudaMemcpyHostToDevice ) );


    transform(
        [] MGPU_DEVICE( int thid,
                        const int *devQueries,
                        const int *devInlabel,
                        const int *devLevel,
                        const int *devAscendant,
                        const int *devFather,
                        const int *devHead,
                        int *devAnswers ) {
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
            devAnswers[thid] = suspects[0];
          else
            devAnswers[thid] = suspects[1];
        },
        queriesToProcess,
        context,
        devQueries,
        devInlabel,
        devLevel,
        devAscendant,
        devFather,
        devHead,
        devAnswers );

    CUCHECK( cudaMemcpy( answers + qStart, devAnswers, sizeof( int ) * queriesToProcess, cudaMemcpyDeviceToHost ) );
  }

  context.synchronize();

  timer.measureTime( Q );

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

__device__ int CudaGetEdgeStart( const int *__restrict__ father, int edgeCode )
{
  if ( edgeCode % 2 )
    return edgeCode / 2;
  else
    return father[edgeCode / 2];
}

__device__ int CudaGetEdgeEnd( const int *__restrict__ father, int edgeCode )
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