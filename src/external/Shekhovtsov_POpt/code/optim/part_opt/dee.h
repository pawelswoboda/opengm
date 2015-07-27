#ifndef dee_h
#define dee_h

#include "dynamic/num_array.h"

using namespace dynamic;

long long dee(const num_array<int,2>  & E, const num_array<int,2> & f1, const num_array<int,3> & f2, num_array<int,2> & X, num_array<int,2> & P,double eps=0);
long long dee2(const num_array<int,2>  & E, const num_array<int,2> & f1, const num_array<int,3> & f2, num_array<int,2> & X, num_array<int,2> & P,double eps=0);

#endif