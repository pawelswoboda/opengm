#ifndef stream_graph_h
#define stream_graph_h

#include "optim/graph/mgraph.h"

//!Create stream graph from G
/*! levels contains nodes ordered in their level number  
two nodes are in same level iff there is no directed path between them
level 0 -- nodes with no incoming arcs
level k = all incoming arcs are only from levels up to k-1

nn contain number of nodes in each level
ordering within level is undefined

levels_inv maps G node index -> (level index,level node index)
*/
class stream_graph{
public:
	dynamic::fixed_array1<int> levels;// list of node indices in order of increasing layers
	dynamic::fixed_array1<int> nn;
	dynamic::fixed_array1<int> level_start; // index in levels where each level starts
	dynamic::fixed_array1<exttype::mint2> levels_inv;
public:
	void init(datastruct::mgraph & G);
};

#endif