#include <iostream>
using namespace std;

static clock_t prevTime;

void measureTime( string msg = "" )
{
  clock_t now = clock();
  if ( msg.size() > 0 )
  {
    cerr.precision( 3 );
    cerr << "\"" << msg << "\" took " << double( now - prevTime ) / CLOCKS_PER_SEC << "s" << endl;
  }
  prevTime = now;
}