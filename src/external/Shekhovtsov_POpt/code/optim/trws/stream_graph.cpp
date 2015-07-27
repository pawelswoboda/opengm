#include "stream_graph.h"



void stream_graph::init(datastruct::mgraph & G){
	//calculate layers			
	levels.reserve(G.nV());
	nn.reserve(G.nV()+1);
	int n = 0;

	dynamic::fixed_array1<int> labels(G.nV()); //which layer each vertex is assigned to
	dynamic::fixed_array1<int> node_labels(G.nV());

	//find sources
	for(int v=0;v<G.nV();++v){
		node_labels[v] = G.in[v].size();
		if(G.in[v].size()==0){
			levels.push_back(v);
			labels[v] = 0;
			++n;
		};
	};
	nn.push_back(n);
	int beg = 0;
	while(levels.size()<G.nV()){
		n = 0;
		//for each node in the preceeding level deactivate outcoming edges
		for(int j=0;j<nn.back();++j){
			int v = levels[beg+j];
			for(int k=0;k<G.out[v].size();++k){
				int e = G.out[v][k];
				int u = G.E[e][1];
				if(--node_labels[u]==0){
					levels.push_back(u);
					labels[u] = nn.size();
					++n;
				};
			};
		};
		beg+=nn.back();
		nn.push_back(n);
	};

	//!todo: might want to do ordering within levels in here

	//build level_starts
	level_start.resize(nn.size());
	beg = 0;
	for (int l = 0; l < nn.size(); ++l){
		level_start[l] = beg;
		beg += nn[l];
	};

	//build levels inverse index
	levels_inv.resize(G.nV());
	beg = 0;
	for(int i=0;i<nn.size();++i){
		for(int j=0;j<nn[i];++j){
			int v = levels[beg];
			levels_inv[v] = exttype::mint2(i,j);
			++beg;
		};
	};
};