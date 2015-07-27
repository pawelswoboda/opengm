//#include "dynamic/block_allocator.h"

#include "mex/mex_io.h"
#include "dynamic/num_array.h"
#include "emaxflow.h"
#include "dee.h"
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
	X - [K x nV] int32 -- unary mask of alive labels
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

	num_array<int,2> g1(mint2(2,nV));
	num_array<int,3> g2(mint3(2,2,nE));

	for(int v=0;v<nV;++v){
		//if(X(y(v),v)==0){
		//	for(int l=0;l<K;++l){
		//		if(X(l,v)==1){
		//			y(v)=l;
		//			break;
		//		};
		//	};
		//};
		g1(0,v) = f1(y[v],v);// 0 represents labeling y
		g1(1,v) = f1(z[v],v);// 1 represents labeling z
	};

	for(int e = 0;e<nE;++e){
		int s = E(0,e);
		int t = E(1,e);
		int a = 0; //f2(y(s),y(t),e);
		
		int c = 1e10;
		for(int j=0;j<K;++j){
			if(j==z(t) || X(j,t)==0)continue;
			c = std::min(c,f2(z(s),j,e) - f2(y(s),j,e));
		};
		int b = 1e10;
		for(int i=0;i<K;++i){
			if(i==z(s) || X(i,s)==0)continue;
			b = std::min(b,f2(i,z(t),e)-f2(i,y(t),e));
		};
		int d = b+c-a;
		d = std::min(b+c-a,f2(z(s),z(t),e)+std::min(b-f2(y(s),z(t),e),c-f2(z(s),y(t),e)));

		if(b==1e10 && c==1e10){// y(s) and y(t) are decided
			a = 0;
			b = 0;
			c = 0;
			d = 0;
		};
		if(b<1e10 && c==1e10){// y(s) is the only remained label in s
			c = 0;
			d = b-a;
		};
		if(b==1e10 && c<1e10){// y(t) is the only remained label in t
			b = 0;
			d = c-a;
		};
		//int b = 0;
		//int c = 0;
		//int d = a;
		g2(0,0,e) = a;
		g2(0,1,e) = b;
		g2(1,0,e) = c;
		g2(1,1,e) = d;
	};

	num_array<int,1> x(y.size());

	emaxflow(E,g1,g2,x);

	mx_array<int,2> X1(mint2(K,nV));
	X1 << X;

	for(int v=0;v<nV;++v){
		if(x[v]==0){// optimal labeling choses y in g
			if(z(v)!=y(v)){
				X1(z(v),v) = 0;
			};
		};
	};

	//dee(E,f1,f2,X1);

	plhs[0] = X1.get_mxArray_andDie();

	//debug::stream.detach();
	//debug::errstream.detach();
};