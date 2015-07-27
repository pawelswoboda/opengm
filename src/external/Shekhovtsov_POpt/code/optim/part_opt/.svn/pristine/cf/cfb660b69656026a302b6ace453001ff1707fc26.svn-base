////////////////////////////////////////////////////////////
// Submodular energy minimization via maxflow
// Graph construction [H. Ishikawa] [D. Sschlesinger]
// Maxflow implementation by [Y. Boykov V. Kolmogor], see maxflow-v3.0/README.TXT
//
// Safe bound on the maximal weight (we use int32 type) is
// 2^25/(d+1) (?) per edge/vertex, where d is vertex degree
//
// AS
////////////////////////////////////////////////////////////

//#include "dynamic/block_allocator.h"
//
//#include "maxflow/graph.h"
#include "mex/mex_io.h"
#include "debug/performance.h"
#include "qpbo-v1.3/QPBO.h"

int inline u(int s,int i,int K){
	return s*(K-1)+(i-1);
};

void err_function(char * s){
	std::string S = std::string("Error:") + s;
	mexErrMsgTxt(S.c_str());
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	/*
	[LB X P] = mexFunction(E,f1,f2)
	Input:
	E -- [2 x nE] int32 -- list of edges
	f1 -- [K x nV] int32 -- unary costs
	f2 -- [K x K x nE] int32 -- pairwise costs
	mode = {0,1,2} -- no probing, probing 1, probing 2
	Output:
	LB - lower bound
	X - [K x nV] int32 -- unary mask of alive labels
	P - [K x nV] int32 -- improving mapping
	*/

	using namespace exttype;

	//mexargs::MexLogStream log("log/output.txt",false);
	//debug::stream.attach(&log);
	//mexargs::MexLogStream err("log/errors.log",true);
	//debug::errstream.attach(&err);

	if(nrhs != 4){
		mexErrMsgTxt("[LB X P] = part_opt_MQPBO_mex(E,f1,f2,ops) -- 4 input arguments expected");
	};

	//debug::stream << "chk1\n";
	mx_array<int,2>  E(prhs[0]);
	//debug::stream << "chk1\n";
	mx_array<int,2> f1(prhs[1]);
	//debug::stream << "chk1\n";
	mx_array<int,3> f2(prhs[2]);
	//debug::stream << "chk1\n";
	mx_array<int,1> mode(prhs[3]);

	int K = f1.size()[0];
	int nV = f1.size()[1];
	int nE = E.size()[1];

	int nW = nV*(K-1);
	int nA = (K-1)*(K-1)*nE + nV*(K-2);
	debug::PerformanceCounter construct_t;

	typedef int tcap;
	typedef long long tflow;
	typedef QPBO<tcap> tgraph;

	debug::stream << "Need memory: " << 4 * nA*sizeof(tgraph::Arc)/1024/1024 << "Mb \n";

	//typedef Graph<tcap,tcap,tflow> tgraph;
	//tgraph * g = new tgraph(nW,nA);
	tgraph * q = new tgraph(nW, nA,&err_function);
	tflow f0 = 0;
	q->AddNode(nW);
	int *mapping = new int[q->GetNodeNum()];

	for(int e=0;e<nE;++e){
		for(int i=1;i<K;++i){
			for(int j=1;j<K;++j){
				tcap c = f2[mint3(i,j,e)]+f2[mint3(i-1,j-1,e)]-f2[mint3(i-1,j,e)]-f2[mint3(i,j-1,e)];
				int s = u(E[mint2(0,e)],i,K);
				int t = u(E[mint2(1,e)],j,K);
				/*
				if(c>0){
				char S[1024];
				sprintf(S,"negative capacity at edge %i=(%i,%i)",e,s,t);
				throw debug_exception(S);
				};
				*/
				if(c==0)continue;
				//g->add_edge(s, t, -c,-c);
				q->AddPairwiseTerm(s,t,0,-c,-c,0);
			};
		};
		f0 += 2*f2[mint3(0,0,e)];
	};

	for(int s=0;s<nV;++s){
		for(int i=2;i<K;++i){
			int uu = u(s,i,K);
			//g->add_edge(u(s,i-1,K), u(s,i,K), 1<<29,0);
			q->AddPairwiseTerm(u(s,i-1,K), u(s,i,K), 0,1<<25,0,0);
		};
	};

	for(int s=0;s<nV;++s){
		for(int i=1;i<K;++i){
			int uu = u(s,i,K);
			tcap a = f1[mint2(i,s)]-f1[mint2(i-1,s)];
			q->AddUnaryTerm(uu, 0, 2*a);
			//g->add_tweights(uu,2*a,0);
		};
		f0 += 2*f1[mint2(0,s)];
	};

	for(int e=0;e<nE;++e){
		int s = E[mint2(0,e)];
		int t = E[mint2(1,e)];
		for(int i=1;i<K;++i){
			tcap a = f2[mint3(i,K-1,e)]-f2[mint3(i-1,K-1,e)]+(f2[mint3(i,0,e)]-f2[mint3(i-1,0,e)]);
			//g->add_tweights(u(s,i,K),a,0);
			q->AddUnaryTerm(u(s,i,K), 0, a);
		};
		for(int j=1;j<K;++j){
			tcap b = f2[mint3(K-1,j,e)]-f2[mint3(K-1,j-1,e)]+(f2[mint3(0,j,e)]-f2[mint3(0,j-1,e)]);
			//g->add_tweights(u(t,j,K),b,0);
			q->AddUnaryTerm(u(t,j,K), 0, b);
		};
	};

	construct_t.stop();
	debug::PerformanceCounter solve_t;

	//tflow flow = g -> maxflow();
	if(mode[0]==0){
		q->Solve();
	}else{
		q->MergeParallelEdges();
		q->Solve();
		tgraph::ProbeOptions ops;
		ops.weak_persistencies = 1;
		ops.dilation = 1;
		//int *tmp_mapping = new int[q->GetNodeNum()];
		for (int i = 0; i < q->GetNodeNum(); i++) {
			mapping[i] = i * 2;
		//	tmp_mapping[i] = i * 2;
		};
		q->Probe(mapping,ops);
	};
	q->ComputeWeakPersistencies();

	//int x = q->GetLabel(0);
	//int y = q->GetLabel(1);
	solve_t.stop();

	//mexPrintf("\n");
	//mexPrintf("flow: %i construct: %3.3fs solve: %3.3fs\n",int(g->flow),construct_t.time(),solve_t.time());

	mx_array<int,1> x(nV);
	mx_array<int,2> X1(mint2(K,nV));
	mx_array<int,2> P(mint2(K,nV));
	X1 << 1; //all alive
	for(int s=0;s<nV;++s){
		int l1=0;
		int l2=K-1;
		for(int i=1;i<K;++i){
			//l+= (g->what_segment(u(s,i,K)) == tgraph::SINK);
			int a;
			if(mode[0]==0){
				a = q->GetLabel(u(s,i,K));
			}else{
				a = q->GetLabel(mapping[u(s,i,K)]/2);
				if(a>=0){
					a = (a+mapping[u(s,i,K)]) % 2;
				};
			};
			//if(a==1)++l;
			if(a==1){
				l1 = std::max(l1,i);
				X1(i-1,s) = 0;
			};
			if(a==0){
				X1(i,s) = 0;
				l2 = std::min(l2,i-1);
			};
		};
		for(int i=0;i<K;++i){
			P(i,s) = i;
			if(i<l1)P(i,s) = l1;
			if(i>l2)P(i,s) = l2;
		};
	};

	if(nlhs>0){
		mx_array<double,1> mxF(1);
		mxF[0] = ((double)(q->ComputeTwiceLowerBound()/2.0+f0))/2.0;
		plhs[0] = mxF.get_mxArray_andDie();
	};
	if(nlhs>1){
		plhs[1] = X1.get_mxArray_andDie();
	};
	if(nlhs>2){
		plhs[2] = P.get_mxArray_andDie();
	};
	delete	q;
	delete mapping;
	//debug::stream.detach();
	//debug::errstream.detach();
};
