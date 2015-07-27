//#include "dynamic/block_allocator.h"

#include "mex/mex_io.h"
#include "dynamic/num_array.h"
//#include "emaxflow.h"
#include "dee.h"
#include "maxflow/graph.h"
//
using namespace dynamic;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	/*
	x = mexFunction(E,f1,f2,X,y)
	Input:
	E -- [2 x nE] int32 -- list of edges
	f1 -- [K x nV] int32 -- unary costs
	f2 -- [K x K x nE] int32 -- pairwise costs
	X -- [K x nV] int32 -- unary mask of alive labels
	y -- [1 x nV] int32 -- test labeling1 for the aux problem
	z -- [1 x nV] int32 -- test labeling2 for the aux problem
	Output:
	//X - [K x nV] int32 -- unary mask of alive labels
	x -- [1 x nV] int32 -- subset reduction
	*/

	using namespace exttype;

	//mexargs::MexLogStream log("log/output.txt",false);
	//debug::stream.attach(&log);
	//mexargs::MexLogStream err("log/errors.log",true);
	//debug::errstream.attach(&err);

	if(nrhs != 6){
		mexErrMsgTxt("[X] = part_opt_elim_mex(E,f1,f2,X,y,z) -- 5 input arguments expected");
	};

	mx_array<int,2>  E(prhs[0]);
	mx_array<int,2> f1(prhs[1]);
	mx_array<int,3> f2(prhs[2]);
	mx_array<int,2> X(prhs[3]);
	mx_array<int,1> y(prhs[4]);
	mx_array<int,1> z(prhs[5]);

	int K = f1.size()[0];
	int nV = f1.size()[1];
	int nE = E.size()[1];

	//intermediate data
	//
	num_array<int,1> g1(nV);
	num_array<int,1> a0_ts(nE);
	num_array<int,1> a0_st(nE);
	num_array<int,1> g_d(nE);
	//

	//maxflow graph construction

	typedef int tcap;
	typedef long long tflow;
	typedef Graph<tcap,tcap,tflow> tgraph;
	tgraph * g = new tgraph(nV,nE);
	g->add_node(nV);
	num_array<tgraph::arc*,1> arcs(nE);

	for(int s=0;s<nV;++s){
		// 1 represents labeling z
		g1(s) = (f1(z(s),s)-f1(y(s),s))*2;
		g->add_tweights(s,g1(s),0);
	};
	for(int e = 0;e<nE;++e){
		int s = E(0,e);
		int t = E(1,e);
		//
		int a_st = 1e8; // a_st
		for(int j=0;j<K;++j){
			if(j==z(t) || X(j,t)==0)continue;
			a_st = std::min(a_st,( f2(z(s),j,e) - f2(y(s),j,e) ));
		};
		if(a_st==1e8)a_st=0;
		a0_st(e) = a_st;
		//
		int a_ts = 1e8; // a_ts
		for(int i=0;i<K;++i){
			if(i==z(s) || X(i,s)==0)continue;
			a_ts = std::min(a_ts,( f2(i,z(t),e)-f2(i,y(t),e) ));
		};
		if(a_ts==1e8)a_ts=0;
		a0_ts(e) = a_ts;
		//
		g_d(e) = (f2(z(s),z(t),e)-f2(y(s),y(t),e));
		int cap = std::max(0,a_ts+a_st-g_d(e));
		arcs(e) = g->arc_last;
		g->add_edge(s, t, cap,cap);
		g->add_tweights(s,2*a_st-cap,0);
		g->add_tweights(t,2*a_ts-cap,0);
	};

//	for(int s=0;s<nV;++s){
//		g1(s) = g->t
//	};

	mx_array<int,1> xx(nV);
	xx << 0;
	mx_array<int,1> x(nV);
	//
	/*
	mexPrintf("f=%i\n",flow);
	for(int s=0;s<nV;++s){
		//lower cut, closest to sink
		x[s] = (g->what_segment(s,tgraph::SOURCE) == tgraph::SINK);
	};
	plhs[0] = x.get_mxArray_andDie();
	delete g;
	return;
	*/
	//
	bool reuse = false;
	tflow flow;
	int it = 0;
	do{
		flow = g -> maxflow(reuse);
		//mexPrintf("f=%i\n",flow);
		//
		for(int s=0;s<nV;++s){
			//lower cut, closest to sink
			x[s] = (g->what_segment(s,tgraph::SOURCE) == tgraph::SINK);
		};
		//
		//if(flow==0){
		//	for(int s=0;s<nV;++s){
		//		if(x(s)){
		//			mexPrintf("error1\n");
		//			return;
		//		};
		//	};
		//};
		//shrink
		bool shrink=false;
		for(int s=0;s<nV;++s){
			if(x(s)){
				g->add_tweights(s,-g1(s),0);
				g1(s) = 0;
				if(!xx(s)){
					shrink = true;
				};
				xx(s) = 1;
			};
		};
		if((flow<0 && !shrink) || (flow==0 && shrink)){
			mexPrintf("error, wrong shrink\n");
			return;
		};

		//
		//if(it==1){
		//	plhs[0] = xx.get_mxArray_andDie();
		//	delete g;
		//	return;
		//}

		for(int e=0;e<nE;++e){
			int s = E(0,e);
			int t = E(1,e);
			if(x(s) || x(t)){
				int o_a_ts = a0_ts(e);
				int o_a_st = a0_st(e);
				int o_cap = std::max(0,o_a_ts+o_a_st-g_d(e));

				if(xx(s))a0_st(e) = 0;
				if(xx(t))a0_ts(e) = 0;
				if(xx(s) && xx(t)){// let z'_s:= y_s and z'_t:=y_t
					g_d(e) = 0;
					// already have: a_st = a_ts = 0;
				}else{
					if(xx(s)){//let z'_s:=y_s, z'_t = z_t 
						//g_d(e) = (f2(z(s),z(t),e)-f2(z(s),y(t),e));
						g_d(e) = (f2(y(s),z(t),e)-f2(y(s),y(t),e));
						//recompute a_ts
						int a_ts = 1e8; // a_ts
						for(int i=0;i<K;++i){
							if(i==y(s) || X(i,s)==0)continue;
							a_ts = std::min(a_ts,( f2(i,z(t),e)-f2(i,y(t),e) ));
						};
						if(a_ts==1e8)a_ts=0;
						a0_ts(e) = a_ts;
					};
					if(xx(t)){//let z'_s:=z_s, z'_t:=y_t
						//g_d(e) = (f2(z(s),z(t),e)-f2(y(s),z(t),e));
						g_d(e) = (f2(z(s),y(t),e)-f2(y(s),y(t),e));
						//recompute a_st
						int a_st = 1e8; // a_st
						for(int j=0;j<K;++j){
							if(j==y(t) || X(j,t)==0)continue;
							a_st = std::min(a_st,( f2(z(s),j,e) - f2(y(s),j,e) ));
						};
						if(a_st==1e8)a_st=0;
						a0_st(e) = a_st;
					};
				};
				//
				int a_ts = a0_ts(e);
				int a_st = a0_st(e);
				int cap = std::max(0,a_ts+a_st-g_d(e));
				//g->add_edge(s, t, cap-o_cap,cap-o_cap);
				int cc = g->get_rcap(arcs(e))-o_cap+cap;
				int rcc = g->get_rcap(arcs(e)->sister)-o_cap+cap;
				int ds = (2*a_st-cap)-(2*o_a_st-o_cap);
				int dt = (2*a_ts-cap)-(2*o_a_ts-o_cap);
				//if(cc<0 && rcc<0){
				//	mexPrintf("error\n");
				//	return;
				//};
				if(cc<0){
					dt += cc;
					rcc+= cc;
					ds -= cc;
					//g->flow+=cc;
					cc = 0;
				};
				if(rcc<0){
					ds += rcc;
					cc += rcc;
					dt -= rcc;
					//g->flow+=rcc;
					rcc = 0;
				};
				g->set_rcap(arcs(e),cc);
				g->set_rcap(arcs(e)->sister,rcc);
				//
				g->add_tweights(s,ds,0);
				g->add_tweights(t,dt,0);
				//
				g->mark_node(s);
				g->mark_node(t);
			};
		};
		reuse = true;
		/*if(++it>100){
			mexPrintf("Error\n");
			plhs[0] = xx.get_mxArray_andDie();
			return;
		};*/
	}while(flow<0);
	//
	delete g;

	plhs[0] = xx.get_mxArray_andDie();
};