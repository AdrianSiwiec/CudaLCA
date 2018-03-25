#include <iostream>
using namespace std;

struct Timer
{
  clock_t prevTime;

  Timer();
  void measureTime( string msg = "" );
};