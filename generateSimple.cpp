#include <algorithm>
#include <iostream>
#include "commons.h"

using namespace std;

int main( int argc, char* argv[] )
{
  if ( argc < 3 )
  {
    cerr << "2 args needed: V, Q. Add filename for binary output" << endl;
    exit( 1 );
  }
  int V = atoi( argv[1] );
  int Q = atoi( argv[2] );

  srand( 241342 + V + Q );

  vector<int> tab;
  tab.push_back( -1 );
  for ( int i = 1; i < V; i++ )
  {
    tab.push_back( rand() % i );
  }

  vector<int> father;
  int root;

  shuffleFathers( tab, father, root );

  ParentsTree tree( V, root, father );

  vector<int> q;
  for ( int i = 0; i < Q * 2; i++ )
  {
    q.push_back( rand() % V );
  }
  Queries queries( Q, q );

  TestCase tc( tree, queries );
  if ( argc == 4 )
    writeToFile( tc, argv[3] );
  else
    writeToStdOut( tc );
}