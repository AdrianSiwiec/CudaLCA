#include <iostream>
#include <vector>
#include "commons.h"

using namespace std;

void dfsRmq( int v );
void initIntervalTree( int n, int power );
int rmqMin( int p, int q, int skad, int dokad, int gdzie );

int INF;

int *rmqTab;
int *dfsEulerPath;  // by preorder
int *rmqPos;        // by preorder
int *preorder;
int *reversePreorder;
int preCounter;
int eulerCounter;

ParentsTree *tree;

int main( int argc, char *argv[] )
{
  Timer timer( "Parse Input" );

  TestCase tc;
  if ( argc == 1 )
  {
    tc = readFromStdIn();
  }
  else
  {
    tc = readFromFile( argv[1] );
  }

  timer.measureTime();
  timer.setPrefix( "Preprocessing" );

  int V = tc.tree.V;
  int root = tc.tree.root;
  INF = V + 10;

  int rmqTabSize = V * 2 - 1;
  int treePower = 1;  // for interval tree
  while ( treePower < rmqTabSize )
    treePower *= 2;

  rmqTab = new int[treePower * 2];
  preorder = new int[V];
  reversePreorder = new int[V];
  rmqPos = new int[V];


  dfsEulerPath = rmqTab + treePower - 1;
  tree = &tc.tree;
  preCounter = 0;

  timer.measureTime( "Allocs" );

  //   cout << tree->sons[0].size() << endl;

  dfsRmq( root );

  //   for ( int i = 0; i < V; i++ )
  //   {
  //     cout << preorder[i] << " ";
  //   }
  //   cout << endl;
  //   for ( int i = 0; i < rmqTabSize; i++ )
  //   {
  //     cout << dfsEulerPath[i] << " " << endl;
  //   }

  initIntervalTree( rmqTabSize, treePower );

  timer.measureTime( "Dfs" );
  timer.setPrefix( "Queries" );

  vector<int> answers( tc.q.N );

  for ( int i = 0; i < tc.q.N; i++ )
  {
    int p = rmqPos[preorder[tc.q.tab[i * 2]]];
    int q = rmqPos[preorder[tc.q.tab[i * 2 + 1]]];

    if ( p > q ) swap( p, q );

    answers[i] = reversePreorder[rmqMin( p, q, 0, treePower - 1, 0 )];
  }

  timer.measureTime( tc.q.N );
  timer.setPrefix( "Write Output" );

  if ( argc < 3 )
  {
    writeAnswersToStdOut( tc.q.N, answers.data() );
  }
  else
  {
    writeAnswersToFile( tc.q.N, answers.data(), argv[2] );
  }

  timer.measureTime();
}
void dfsRmq( int v )
{
  preorder[v] = preCounter;
  reversePreorder[preCounter] = v;

  dfsEulerPath[eulerCounter] = preCounter;
  rmqPos[preCounter] = eulerCounter;

  preCounter++;
  eulerCounter++;

  for ( int son = tree->son[v]; son != -1; son = tree->neighbour[son] )
  {
    dfsRmq( son );
    dfsEulerPath[eulerCounter] = preorder[v];
    eulerCounter++;
  }
}
void initIntervalTree( int n, int power )
{
  for ( int i = power + n - 1; i < power * 2 - 1; i++ )
  {
    rmqTab[i] = INF;
  }

  for ( int i = power - 2; i >= 0; i-- )
  {
    rmqTab[i] = min( rmqTab[i * 2 + 1], rmqTab[i * 2 + 2] );
  }
}
int rmqMin( int p, int q, int skad, int dokad, int gdzie )
{
  if ( p == skad && q == dokad ) return rmqTab[gdzie];
  if ( p > ( skad + dokad ) / 2 ) return rmqMin( p, q, ( skad + dokad ) / 2 + 1, dokad, gdzie * 2 + 2 );
  if ( q <= ( skad + dokad ) / 2 ) return rmqMin( p, q, skad, ( skad + dokad ) / 2, gdzie * 2 + 1 );

  return min( rmqMin( p, ( skad + dokad ) / 2, skad, ( skad + dokad ) / 2, gdzie * 2 + 1 ),
              rmqMin( ( skad + dokad ) / 2 + 1, q, ( skad + dokad ) / 2 + 1, dokad, gdzie * 2 + 2 ) );
}