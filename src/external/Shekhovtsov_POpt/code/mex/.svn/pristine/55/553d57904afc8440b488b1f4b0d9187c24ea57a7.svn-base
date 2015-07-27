#include "num_array.h"
#include "mex_io.h"
#include "debug/msvcdebug.h"
#include "debug/logs.h"

using namespace exttype;

void my_processing_function(mx_array<float,4> & x){
	for(int i=0;i<x.size()[3];++i){
		for(int j=0;j<x.size()[3];++j){
			for(int k=0;k<x.size()[1];++k){
				for(int l=0;l<x.size()[0];++l){
					x(l,k,j,i)+=1;
				};
			};
		};
	};
	for(itern<4> ii(x.size());ii.allowed();++ii){
		x[ii]+=1;
	};
};

int main(){

	using namespace dynamic;

	//create 3D double array of size 5 x 5 x 5
	
	array<double,3> A(int3(5,5,5));
	A(0,0,0) = 1;
	A[1] = 2;
	A[int3(1,2,3)] = 3;

	array<float,3> B = A;
	mx_array<float,3> C = B/B.min().first;

	mxArray * p = C.get_mxArray();

	debug::stream<<"Testing exception\n";
	try{
		// this throws exception -- p is of float type
		mx_array<double,3> D(p);
	}catch(const std::exception & e){
		debug::stream<<"\tException caugth: " << e.what()<<"\n";
	};
	try{
		// this throws exception -- p is at least 3D
		mx_array<float,2> D(p);
	}catch(const std::exception & e){
		debug::stream<<"\tException caugth: " << e.what()<<"\n";
	};

	//this is OK
	mx_array<float,4> D(p);

	D+=2;// this will change data associated to p

	my_processing_function(D);
	
	return 0;
};