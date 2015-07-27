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
	y -- [1 x nV] int32 -- test labeling for the aux problem
	Output:
	X - [K x nV] int32 -- unary mask of alive labels
	P - [K x nV] int32 -- imroving mapping
	*/

	using namespace exttype;

	//mexargs::MexLogStream log("log/output.txt",false);
	//debug::stream.attach(&log);
	//mexargs::MexLogStream err("log/errors.log",true);
	//debug::errstream.attach(&err);

	if(nrhs != 5){
		mexErrMsgTxt("[X P] = part_opt_IK_mex(E,f1,f2,X,y) -- 5 input arguments expected");
	};

	mx_array<int,2>  E(prhs[0]);
	mx_array<int,2> _f1(prhs[1]);
	mx_array<int,3> _f2(prhs[2]);
	num_array<int,2> f1; f1 = _f1;
	num_array<int,3> f2; f2 = _f2;
	mx_array<int,2> X(prhs[3]);
	mx_array<int,1> y(prhs[4]);

	int K = f1.size()[0];
	int nV = f1.size()[1];
	int nE = E.size()[1];

	num_array<int,2> g1(mint2(2,nV));
	num_array<int,3> g2(mint3(2,2,nE));


	for(int v=0;v<nV;++v){
		if(X(y(v),v)==0){
			for(int l=0;l<K;++l){
				if(X(l,v)==1){
					y(v)=l;
					break;
				};
			};
		};
	};

	// reparametrize such that f2(y,j,st) = f2(i,y,st) = 0 forall i,j
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
	};

	// reparametrize such that f1(0,s) = 0
	//for(int s=0;s<nV;++s){
	//	int f1y = f1(y[s],s);
	//	for(int i=0;i<K;++i){
	//		f1(i,s) -= f1y;
	//	};
	//};

	for(int v=0;v<nV;++v){
		//if(X(y(v),v)==0){
		//	for(int l=0;l<K;++l){
		//		if(X(l,v)==1){
		//			y(v)=l;
		//			break;
		//		};
		//	};
		//};
		g1(0,v) = f1(y[v],v);
		g1(1,v) = 1e10;
		for(int l=0;l<K;++l){
			if(l==y[v] || X(l,v)==0)continue;
			g1(1,v) = std::min(g1(1,v),f1(l,v));
		};
	};

	for(int e = 0;e<nE;++e){
		int s = E(0,e);
		int t = E(1,e);
		int a = f2(y(s),y(t),e);

		int b = 1e10;
		for(int j=0;j<K;++j){
			if(j==y(t) || X(j,t)==0)continue;
			b = std::min(b,f2(y(s),j,e));
		};
		int c = 1e10;
		for(int i=0;i<K;++i){
			if(i==y(s) || X(i,s)==0)continue;
			c = std::min(c,f2(i,y(t),e));
		};
		int d = b+c-a;
		for(int j=0;j<K;++j){
			if(j==y(t) || X(j,t)==0)continue;
			int r = b-f2(y(s),j,e);
			int * p1 = &f2(0,j,e);
			int * p2 = &f2(0,y(t),e);
			for(int i=0;i<K;++i){
				if(i==y(s) || X(i,s)==0)continue;
				d = std::min(d,p1[i]+std::min(r,c-p2[i]));
			};
		};
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
    mx_array<int,2> P(mint2(K,nV));

	for(int v=0;v<nV;++v){
		if(x[v]==0){// optimal labeling choses y in g
			for(int l=0;l<K;++l){
			    P(l,v) = y(v);
				if(l==y(v))continue;
				X1(l,v) = 0;
			};
		}else{
		    for(int l=0;l<K;++l)P(l,v) = l;
		};
	};

	//dee(E,f1,f2,X1);

	plhs[0] = X1.get_mxArray_andDie();
	plhs[1] = P.get_mxArray_andDie();

	//debug::stream.detach();
	//debug::errstream.detach();
};
