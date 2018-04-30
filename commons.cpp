#include "commons.h"
#include <algorithm>
#include <fstream>

using namespace std;

Timer::Timer( string prefix ) : prefix( prefix )
{
  prevTime = clock();
  prefixTime = clock();
}

void Timer::setPrefix( string prefix )
{
  double time = resetTimer( prefixTime );

  cerr.precision( 3 );
  cerr << this->prefix << "," << time << ","
       << "Whole" << endl;

  this->prefix = prefix;
}
void Timer::measureTime( string msg )
{
  double time = resetTimer( prevTime );

  cerr.precision( 3 );
  cerr << prefix << "," << time << "," << msg << endl;
}
void Timer::measureTime( int i )
{
  measureTime( to_string( i ) );
}
double Timer::resetTimer( clock_t &timer )
{
  clock_t now = clock();
  double res = double( now - timer ) / CLOCKS_PER_SEC;

  timer = now;

  return res;
}


ParentsTree::ParentsTree() : V( 0 ), root( 0 ), father( vector<int>() ), sons( 0 ) {}
ParentsTree::ParentsTree( int V, int root, const vector<int> &father ) : V( V ), root( root ), father( father )
{
  sons = new vector<int>[V];
  for ( int i = 0; i < V; i++ )
  {
    if ( father[i] != -1 )
    {
      sons[father[i]].push_back( i );
    }
  }
}
ParentsTree::ParentsTree( ifstream &in )
{
  in.read( (char *) &V, sizeof( int ) );
  in.read( (char *) &root, sizeof( int ) );

  father.resize( V );
  in.read( (char *) father.data(), sizeof( int ) * V );

  sons = new vector<int>[V];

  for ( int i = 0; i < V; i++ )
  {
    int sonsCounter;
    in.read( (char *) &sonsCounter, sizeof( int ) );
    if ( sonsCounter > 0 )
    {
      sons[i].resize( sonsCounter );
      in.read( (char *) sons[i].data(), sizeof( int ) * sonsCounter );
    }
  }
}
void ParentsTree::writeToStream( ofstream &out )
{
  out.write( (char *) &V, sizeof( int ) );
  out.write( (char *) &root, sizeof( int ) );
  out.write( (char *) father.data(), sizeof( int ) * V );
  for ( int i = 0; i < V; i++ )
  {
    int size = sons[i].size();
    out.write( (char *) &size, sizeof( int ) );
    if ( sons[i].size() > 0 ) out.write( (char *) sons[i].data(), sizeof( int ) * sons[i].size() );
  }
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
      firstEdge = getEdgeCode( i, 0 );
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
  cout << endl;
  for ( int i = 0; i < tc.tree.V; i++ )
  {
    cout << tc.tree.sons[i].size() << " ";
    for ( int j = 0; j < tc.tree.sons[i].size(); j++ )
    {
      cout << tc.tree.sons[i][j] << " ";
    }
    cout << endl;
  }
  cout << tc.q.N << endl;
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
  ParentsTree tree;
  cin >> tree.V >> tree.root;

  tree.father.resize( tree.V );
  for ( int i = 0; i < tree.V; i++ )
  {
    cin >> tree.father[i];
  }

  tree.sons = new vector<int>[tree.V];
  for ( int i = 0; i < tree.V; i++ )
  {
    int tmpSize;
    cin >> tmpSize;
    tree.sons[i].resize( tmpSize );
    for ( int j = 0; j < tmpSize; j++ )
    {
      cin >> tree.sons[i][j];
    }
  }

  int N;
  cin >> N;
  vector<int> q( N * 2 );

  for ( int i = 0; i < N * 2; i++ )
  {
    cin >> q[i];
  }
  return TestCase( tree, Queries( N, q ) );
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