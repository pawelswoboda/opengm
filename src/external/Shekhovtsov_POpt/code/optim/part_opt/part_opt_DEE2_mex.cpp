//#include "dynamic/block_allocator.h"

#include "mex/mex_io.h"
#include "dynamic/num_array.h"
#include "emaxflow.h"
#include "dee.h"
//
using namespace dynamic;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	/*
	[X P] = mexFunction(E,f1,f2,X,eps)
	Input:
	E -- [2 x nE] int32 -- list of edges
	f1 -- [K x nV] int32 -- unary costs
	f2 -- [K x K x nE] int32 -- pairwise costs
	X -- [K x nV] int32 -- unary mask of alive labels
	eps -- [1 x 1] double -- 0 if weak partial optimality, > 0 if strong with threshold eps
	Output:
	X - [K x nV] int32 -- unary mask of alive labels
	P - [K x nV] int32 -- improving mapping
	*/

	using namespace exttype;

	//mexargs::MexLogStream log("log/output.txt",false);
	//debug::stream.attach(&log);
	//mexargs::MexLogStream err("log/errors.log",true);
	//debug::errstream.attach(&err);

	if(nrhs !=4 && nrhs !=5){
		mexErrMsgTxt("[X P] = part_opt_dee2(E,f1,f2,X, eps=0) -- 4(5) input arguments expected");
	};
	//
	mx_array<int,2>  E(prhs[0]);
	mx_array<int,2> f1(prhs[1]);
	mx_array<int,3> f2(prhs[2]);
	mx_array<int,2> X(prhs[3]);
	double eps = 0;
	if(nrhs>=5){
		mx_array<double,1> m_eps(prhs[4]);
		eps = m_eps[0];
	};
	//
	int K = f1.size()[0];
	int nV = f1.size()[1];
	int nE = E.size()[1];
	//
	mx_array<int,2> X1(mint2(K,nV));
	X1 << X;
	//
	mx_array<int,2> P(mint2(K,nV));
	//
	dee2(E,f1,f2,X1,P,eps);
	//
	if(nlhs>0){
		plhs[0] = X1.get_mxArray_andDie();
	};
	if(nlhs>1){
		plhs[1] = P.get_mxArray_andDie();
	};
	//
	//debug::stream.detach();
	//debug::errstream.detach();
};