//#include "dynamic/block_allocator.h"

#include "mex/mex_io.h"
#include "dynamic/num_array.h"
#include "emaxflow.h"
#include "dee.h"
//
using namespace dynamic;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	/*
	X = mexFunction(E,f1,f2,X)
	Input:
	E -- [2 x nE] int32 -- list of edges
	f0 -- int32 -- constant term
	f1 -- [K x nV] int32 -- unary costs
	f2 -- [K x K x nE] int32 -- pairwise costs
	y  -- [1 x nV] int32 -- labeling
	Output:
		modifies f0,f1,f2
	*/
	using namespace exttype;

	//mexargs::MexLogStream log("log/output.txt",false);
	//debug::stream.attach(&log);
	//mexargs::MexLogStream err("log/errors.log",true);
	//debug::errstream.attach(&err);

	if(nrhs != 5){
		mexErrMsgTxt("[X] = part_opt_IK_mex(E,f0,f1,f2,y) -- 5 input arguments expected");
	};
	//
	mx_array<int,2>  E(prhs[0]);
	mx_array<int,1> F0(prhs[1]);
	int & f0 = F0(0);
	mx_array<int,2> f1(prhs[2]);
	mx_array<int,3> f2(prhs[3]);
	mx_array<int,1> y(prhs[4]);
	//
	int K = f1.size()[0];
	int nV = f1.size()[1];
	int nE = E.size()[1];
	//
	//
	//
	for(int e=0;e<nE;++e){
		int s = E(0,e);
		int t = E(1,e);
		int f2yy = f2(y[s],y[t],e);
		for(int i=0;i<K;++i){
			f1(i,s) = f1(i,s) + f2(i,y[t],e) - f2yy;
		};
		for(int j=0;j<K;++j){
			f1(j,t) = f1(j,t) + f2(y[s],j,e) - f2yy;
		};
		for(int i=0;i<K;++i){
			for(int j=0;j<K;++j){
				if(i==y[s] || j==y[t])continue;
				f2(i,j,e) = f2(i,j,e)-f2(i,y[t],e)-f2(y[s],j,e)+f2yy;
			};
		};
		for(int i=0;i<K;++i){
			f2(i,y[t],e) = 0;
			f2(y[s],i,e) = 0;
		};
		f0 = f0 + f2yy;
	};
	//reparametrize such that f1(0,s) = 0
	for(int s=0;s<nV;++s){
		int f1y = f1(y[s],s);
		for(int i=0;i<K;++i){
			f1(i,s) -= f1y;
		};
		f0 = f0 + f1y;
	};
	/*
	if(nlhs>0){
		plhs[0] = X1.get_mxArray_andDie();
	};
	if(nlhs>1){
		plhs[1] = P.get_mxArray_andDie();
	};
	*/
	//
	//debug::stream.detach();
	//debug::errstream.detach();
};
