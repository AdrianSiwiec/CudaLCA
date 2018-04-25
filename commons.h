#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

struct Timer
{
  clock_t prevTime;

  Timer();
  void measureTime( string msg = "" );
};

struct ParentsTree
{
  int V;
  int root;
  vector<int> father;

  ParentsTree();
  ParentsTree( int V, int root, const vector<int> &father );
  ParentsTree( ifstream &in );

  void writeToStream( ofstream &out );
};

struct Queries
{
  int N;
  vector<int> tab;

  Queries();
  Queries( int N, const vector<int> &tab );
  Queries( ifstream &in );

  void writeToStream( ofstream &out );
};

struct TestCase
{
  ParentsTree tree;
  Queries q;

  TestCase();
  TestCase( const ParentsTree &tree, const Queries &q );
  TestCase( ifstream &in );

  void writeToStream( ofstream &out );
};

int getEdgeCode( int a, bool toFather );  // edge father[a]->a or a->father[a]
int getEdgeStart( ParentsTree &tree, int edgeCode );
int getEdgeEnd( ParentsTree &tree, int edgeCode );

struct NextEdgeTree
{
  int firstEdge;
  vector<int> next;

  NextEdgeTree();
  NextEdgeTree( int firstEdge, const vector<int> &next );
  NextEdgeTree( ParentsTree &tree );
};

void writeToFile( TestCase &tc, const char *filename );
void writeToStdOut( TestCase &tc );
TestCase readFromFile( const char *filename );
TestCase readFromStdIn();

void writeAnswersToFile( int Q, int *ans, const char *filename );
void writeAnswersToStdOut( int Q, int *ans );

void shuffleFathers( vector<int> &in, vector<int> &out, int &root );