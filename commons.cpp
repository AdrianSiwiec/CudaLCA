#include "commons.h"
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