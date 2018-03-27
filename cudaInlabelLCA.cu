#include <cuda_runtime.h>
#include <iostream>
#include <moderngpu/transform.hxx>
#include "commons.h"
#include "cudaCommons.h"

using namespace std;
using namespace mgpu;

int main( int argc, char *argv[] )
{
  Timer timer = Timer();

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

  timer.measureTime( "Read Input" );

  EulerPath path( tc.tree );

  timer.measureTime( "Generating Euler Path" );

  const int V = tc.tree.V;
  const int E = 2 * V;

  int *devNextEdge;
  int *devEdgeRank;

  CUCHECK( cudaMalloc( (void **) &devNextEdge, sizeof( int ) * E ) );
  CUCHECK( cudaMalloc( (void **) &devEdgeRank, sizeof( int ) * E ) );

  timer.measureTime( "Cuda Allocs" );

  CUCHECK( cudaMemcpy( devNextEdge, path.next.data(), sizeof( int ) * E, cudaMemcpyHostToDevice ) );

  int threadsPerBlockX = 1024;
  int blocksPerGridX = ( E + threadsPerBlockX - 1 ) / threadsPerBlockX;

  transform( [=] MGPU_DEVICE( int thid ) { devEdgeRank[thid] = 0; }, E, context );
  context.synchronize();

  timer.measureTime( "Copy Input to Dev and Init data" );

  CudaListRank( devEdgeRank, E, devNextEdge, threadsPerBlockX, blocksPerGridX );

  timer.measureTime( "Edges List Rank" );

  int *devSortedEdges;

  CUCHECK( cudaMalloc( (void **) &devSortedEdges, sizeof( int ) * E ) );

  transform( [=] MGPU_DEVICE( int thid ) { devSortedEdges[E - devEdgeRank[thid] - 2] = thid; }, E, context );
  context.synchronize();

  int *sortedEdges = (int *) malloc( sizeof( int ) * E );
  CUCHECK( cudaMemcpy( sortedEdges, devSortedEdges, sizeof( int ) * E, cudaMemcpyDeviceToHost ) );

  for ( int i = 0; i < E; i++ )
  {
    cout << getEdgeStart( tc.tree, sortedEdges[i] ) << "->" << getEdgeEnd( tc.tree, sortedEdges[i] ) << endl;
  }
}