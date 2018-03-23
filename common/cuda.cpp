#include "cuda.h"
#include <iostream>

using namespace std;

void testRes( CUresult res, string msg );

CUmodule cudaInit( string filename )
{
  cuInit( 0 );

  CUdevice cuDevice;
  CUresult res = cuDeviceGet( &cuDevice, 0 );
  testRes( res, "Acquire cuDevice" );

  CUcontext cuContext;
  res = cuCtxCreate( &cuContext, 0, cuDevice );
  testRes( res, "Create context" );

  CUmodule cuModule = (CUmodule) 0;
  res = cuModuleLoad( &cuModule, filename.c_str() );
  testRes( res, "Load LCA module" );

  return cuModule;
}

CUfunction cudaGetFunction( CUmodule cuModule, string name )
{
  CUfunction ret;

  CUresult res = cuModuleGetFunction( &ret, cuModule, name.c_str() );
  testRes( res, "Load function " + name );

  return ret;
}

void cudaMemAlloc( CUdeviceptr *ptr, size_t size, string name = "" )
{
  CUresult res = cuMemAlloc( ptr, size );
  testRes( res, "Alloc" + name );
}

void cudaLaunchKernel( CUfunction function, int blocksPerGridX, int threadsPerBlockX, void *args[], string name = "" )
{
  CUresult res = cuLaunchKernel( function, blocksPerGridX, 1, 1, threadsPerBlockX, 1, 1, 0, 0, args, 0 );
  testRes( res, "Launch " + name );
  res = cuCtxSynchronize();
  testRes( res, "Sync " + name );
}

void testRes( CUresult res, string msg )
{
  if ( res != CUDA_SUCCESS )
  {
    cout << "Error when executing: \"" << msg << "\"" << endl;
    exit( 1 );
  }
}