#include "emaxflow.h"
#include "maxflow/graph.h"
//
int inline u(int s,int i,int K){
	return s*(K-1)+(i-1);
};
//
long long emaxflow(const num_array<int,2>  & E, const num_array<int,2> & f1, const num_array<int,3> & f2, num_array<int,1> & x){
	int K = f1.size()[0];
	int nV = f1.size()[1];
	int nE = E.size()[1];

	//maxflow graph construction

	int nW = nV*(K-1);
	int nA = (K-1)*(K-1)*nE + nV*(K-2);

	typedef int tcap;
	typedef long long tflow;
	typedef Graph<tcap,tcap,tflow> tgraph;
	tgraph * g = new tgraph(nW,nA);
	g->add_node(nW);

	for(int e=0;e<nE;++e){
		for(int i=1;i<K;++i){
			for(int j=1;j<K;++j){
				tcap c = f2[mint3(i,j,e)]+f2[mint3(i-1,j-1,e)]-f2[mint3(i-1,j,e)]-f2[mint3(i,j-1,e)];
				int s = u(E[mint2(0,e)],i,K);
				int t = u(E[mint2(1,e)],j,K);
				if(c>0){
					char S[1024];
					sprintf(S,"negative capacity at edge %i=(%i,%i)",e,s,t);
					throw debug_exception(S);
				};
				if(c==0)continue;
				g->add_edge(s, t, -c,-c);
			};
		};
		g->flow += 2*f2[mint3(0,0,e)];
	};

	for(int s=0;s<nV;++s){
		for(int i=2;i<K;++i){
			int uu = u(s,i,K);
			g->add_edge(u(s,i-1,K), u(s,i,K), 1<<29,0);
		};
	};

	for(int s=0;s<nV;++s){
		for(int i=1;i<K;++i){
			int uu = u(s,i,K);
			tcap a = f1[mint2(i,s)]-f1[mint2(i-1,s)];
			g->add_tweights(uu,2*a,0);
		};
		g->flow += 2*f1[mint2(0,s)];
	};

	for(int e=0;e<nE;++e){
		int s = E[mint2(0,e)];
		int t = E[mint2(1,e)];
		for(int i=1;i<K;++i){
			tcap a = f2[mint3(i,K-1,e)]-f2[mint3(i-1,K-1,e)]+(f2[mint3(i,0,e)]-f2[mint3(i-1,0,e)]);
			g->add_tweights(u(s,i,K),a,0);
		};
		for(int j=1;j<K;++j){
			tcap b = f2[mint3(K-1,j,e)]-f2[mint3(K-1,j-1,e)]+(f2[mint3(0,j,e)]-f2[mint3(0,j-1,e)]);
			g->add_tweights(u(t,j,K),b,0);
		};
	};

	tflow flow = g -> maxflow();
	// write out optimal labeling
	for(int s=0;s<nV;++s){
		int l=0;
		for(int i=1;i<K;++i){
			l+= (g->what_segment(u(s,i,K),tgraph::SOURCE) == tgraph::SINK);
			//l+= !(g->what_segment(u(s,i,K),tgraph::SOURCE) == tgraph::SOURCE);
		};
		x[s] = l;
	};
	delete g;
	//
	return flow/2;
};