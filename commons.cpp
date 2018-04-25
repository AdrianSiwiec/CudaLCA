#include "commons.h"
#include <algorithm>
#include <fstream>

using namespace std;

Timer::Timer()
{
  prevTime = clock();
}

void Timer::measureTime( string msg )
{
  clock_t now = clock();
  if ( msg.size() > 0 )
  {
    cerr.precision( 3 );
    cerr << "\"" << msg << "\" took " << double( now - prevTime ) / CLOCKS_PER_SEC << "s" << endl;
  }

  prevTime = now;
}

ParentsTree::ParentsTree() : V( 0 ), root( 0 ), father( vector<int>() ) {}
ParentsTree::ParentsTree( int V, int root, const vector<int> &father ) : V( V ), root( root ), father( father ) {}
ParentsTree::ParentsTree( ifstream &in )
{
  in.read( (char *) &V, sizeof( int ) );
  in.read( (char *) &root, sizeof( int ) );

  father.resize( V );
  in.read( (char *) father.data(), sizeof( int ) * V );
}
void ParentsTree::writeToStream( ofstream &out )
{
  out.write( (char *) &V, sizeof( int ) );
  out.write( (char *) &root, sizeof( int ) );
  out.write( (char *) father.data(), sizeof( int ) * V );
}

Queries::Queries() : N( 0 ), tab( vector<int>() ) {}
Queries::Queries( int N, const vector<int> &tab ) : N( N ), tab( tab ) {}
Queries::Queries( ifstream &in )
{
  in.read( (char *) &N, sizeof( int ) );

  tab.resize( N * 2 );
  in.read( (char *) tab.data(), sizeof( int ) * N * 2 );
}
void Queries::writeToStream( ofstream &out )
{
  out.write( (char *) &N, sizeof( int ) );
  out.write( (char *) tab.data(), sizeof( int ) * N * 2 );
}

TestCase::TestCase() : tree( ParentsTree() ), q( Queries() ) {}
TestCase::TestCase( const ParentsTree &tree, const Queries &q ) : tree( tree ), q( q ) {}
TestCase::TestCase( ifstream &in ) : tree( in ), q( in ) {}
void TestCase::writeToStream( ofstream &out )
{
  tree.writeToStream( out );
  q.writeToStream( out );
}

int getEdgeCode( int a, bool toFather )
{
  return a * 2 + toFather;
}
int getEdgeStart( ParentsTree &tree, int edgeCode )
{
  if ( edgeCode % 2 )
    return edgeCode / 2;
  else
    return tree.father[edgeCode / 2];
}
int getEdgeEnd( ParentsTree &tree, int edgeCode )
{
  return getEdgeStart( tree, edgeCode ^ 1 );
}

NextEdgeTree::NextEdgeTree() : firstEdge( 0 ), next( vector<int>() ) {}
NextEdgeTree::NextEdgeTree( int firstEdge, const vector<int> &next ) : firstEdge( firstEdge ), next( next ) {}
NextEdgeTree::NextEdgeTree( ParentsTree &tree )
{
  next.resize( tree.V * 2, -1 );

  vector<int> lastEdges( tree.V, -1 );
  vector<int> firstEdges( tree.V, -1 );

  for ( int i = 0; i < tree.V; i++ )
  {
    int father = tree.father[i];
    if ( father == -1 ) continue;

    if ( lastEdges[father] == -1 )
    {
      firstEdges[father] = getEdgeCode( i, 0 );
    }
    else
    {
      next[lastEdges[father]] = getEdgeCode( i, 0 );
    }

    lastEdges[father] = getEdgeCode( i, 1 );
  }


  for ( int i = 0; i < tree.V; i++ )
  {
    int father = tree.father[i];
    if ( father == -1 )
    {
      firstEdge = firstEdges[i];
    }
    if ( firstEdges[i] == -1 )
    {
      next[getEdgeCode( i, 0 )] = getEdgeCode( i, 1 );
    }
    else
    {
      next[getEdgeCode( i, 0 )] = firstEdges[i];
      next[lastEdges[i]] = getEdgeCode( i, 1 );
    }
  }
}

void writeToFile( TestCase &tc, const char *filename )
{
  ofstream out( filename, ios::binary );
  tc.writeToStream( out );
}
void writeToStdOut( TestCase &tc )
{
  cout << tc.tree.V << " ";
  cout << tc.tree.root << endl;
  for ( int i = 0; i < tc.tree.V; i++ )
  {
    cout << tc.tree.father[i] << " ";
  }
  cout << endl << tc.q.N << endl;
  for ( int i = 0; i < tc.q.N * 2; i++ )
  {
    cout << tc.q.tab[i] << " ";
  }
  cout << endl;
}

TestCase readFromFile( const char *filename )
{
  ifstream in( filename, ios::binary );
  return TestCase( in );
}
TestCase readFromStdIn()
{
  TestCase tc;
  cin >> tc.tree.V >> tc.tree.root;
  tc.tree.father.resize( tc.tree.V );
  for ( int i = 0; i < tc.tree.V; i++ )
  {
    cin >> tc.tree.father[i];
  }
  cin >> tc.q.N;
  tc.q.tab.resize( tc.q.N * 2 );
  for ( int i = 0; i < tc.q.N * 2; i++ )
  {
    cin >> tc.q.tab[i];
  }
  return tc;
}

void writeAnswersToStdOut( int Q, int *ans )
{
  for ( int i = 0; i < Q; i++ )
  {
    cout << ans[i] << endl;
  }
}
void writeAnswersToFile( int Q, int *ans, const char *filename )
{
  ofstream out( filename, ios::binary );
  out.write( (char *) ans, sizeof( int ) * Q );
}

void shuffleFathers( vector<int> &in, vector<int> &out, int &root )
{
  int V = in.size();
  vector<int> shuffle;
  for ( int i = 0; i < V; i++ )
  {
    shuffle.push_back( i );
  }
  random_shuffle( shuffle.begin(), shuffle.end() );

  vector<int> newPos;
  newPos.resize( V );
  for ( int i = 0; i < V; i++ )
  {
    newPos[shuffle[i]] = i;
  }

  out.clear();
  for ( int i = 0; i < V; i++ )
  {
    if ( shuffle[i] == 0 )
    {
      root = i;
    }
    out.push_back( in[shuffle[i]] == -1 ? -1 : newPos[in[shuffle[i]]] );
  }
}