#include <iostream>
#include <vector>
#include "../common/time.cpp"

using namespace std;

void dfs( int i );

vector<int> *G;
int *depth;
int *father;
int *answers;
int *queries;

int main()
{
  ios_base::sync_with_stdio( 0 );

  measureTime();

  int V;
  cin >> V;

  G = new vector<int>[V];
  depth = new int[V];
  father = new int[V];

  for ( int i = 1; i < V; i++ )
  {
    int tmp;
    cin >> tmp;
    father[i] = tmp;
    G[tmp].push_back( i );
  }

  measureTime( "Read Input" );

  depth[0] = 0;
  dfs( 0 );

  measureTime( "Preprocessing" );

  //   for ( int i = 0; i < V; i++ )
  //   {
  //     cout << i << ": " << depth[i] << endl;
  //   }

  int Q;
  cin >> Q;

  queries = new int[2 * Q];
  answers = new int[Q];

  for ( int i = 0; i < Q; i++ )
  {
    cin >> queries[i * 2] >> queries[i * 2 + 1];
  }

  measureTime( "Read Queries" );

  for ( int i = 0; i < Q; i++ )
  {
    int p = queries[i * 2];
    int q = queries[i * 2 + 1];
    while ( depth[p] != depth[q] )
    {
      if ( depth[p] > depth[q] )
        p = father[p];
      else
        q = father[q];
    }

    while ( p != q )
    {
      p = father[p];
      q = father[q];
    }

    answers[i] = p;
  }

  measureTime( "Calculate Queries" );

  for ( int i = 0; i < Q; i++ )
  {
    cout << answers[i] << endl;
  }

  measureTime( "Output Queries" );
}

void dfs( int i )
{
  for ( int a = 0; a < G[i].size(); a++ )
  {
    if ( depth[G[i][a]] == 0 )
    {
      depth[G[i][a]] = depth[i] + 1;
      dfs( G[i][a] );
    }
  }
}
