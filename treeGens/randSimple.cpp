#include <iostream>

using namespace std;

int main()
{
  srand( time( 0 ) );

  int V = 10000000;
  int Q = 10000000;

  cout << V << endl;
  for ( int i = 1; i < V; i++ )
  {
    cout << rand() % i << endl;
  }

  cout << Q << endl;
  for ( int i = 0; i < Q; i++ )
  {
    cout << rand() % V << " " << rand() % V << endl;
  }
}