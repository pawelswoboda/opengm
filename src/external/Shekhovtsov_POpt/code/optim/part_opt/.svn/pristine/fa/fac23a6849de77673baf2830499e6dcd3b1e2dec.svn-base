//#include "dynamic/block_allocator.h"

#include "mex/mex_io.h"
#include "mex/mex_log.h"
#include "dynamic/num_array.h"
#include "part_opt_TRWS.h"
#include "debug/performance.h"
#include "opengm_read.h"

//#include <omp.h>

//
using namespace dynamic;
using namespace exttype;

void mexFunction_protect(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	/*
	[y P M phi] = mexFunction(E,f1,f2,X,M,y,options)
	[y P M phi] = mexFunction(file_name,y,options)
	Input 1:
	0) E -- [2 x nE] int32 -- list of edges
	1) f1 -- [K x nV] double -- unary costs
	2) f2 -- [K x K x nE] double -- pairwise costs
	3) X -- [K x nV] int32 -- unary mask of alive labels (reserved, currently no effect)
	4) M -- [K x nE] double -- backward messages
	5) y -- [1 x nV] int32 -- test labeling
	6) options struct double -- get available fields from an output
	Input 2:
	file_name -- opengm model
	M, y, options -- as above
	Output:
	y - integer labeling found by TRW-S in the initialization phase
	P - [K x nV] int32 -- imroving mapping
	M - [K x nE] double -- current messages
	statistics - times for different parts of computation
	hist - history of the lower bound, etc
	burn [K x nV x nC] - double - progress of prunning: 
		layer 0: iteration # when prunned, 
		layer 1: margin when prunned
		layer 2: prunning condition: pixel / cut / WTA
		layer 3: sensetivity
	*/

	energy_auto<d_type> * f = 0;
	mx_array<double, 2> M;
	mx_array<int, 1> y;
	mx_struct ops;
	alg_po_trws * alg = new alg_po_trws();
	//
	if (nrhs <= 4){ // this is Input 2, read model from file
		mx_string fname(prhs[0]);
		f = opengm_read<d_type>(fname, "gm"); // , f);
		if (nrhs >= 2){
			M = mx_array<double, 2>(prhs[1]);
		};
		if (nrhs >= 3){
			y = mx_array<int, 1>(prhs[2]);
		};
		if (nrhs >= 4){
			ops = mx_struct(prhs[3]);
		};
		//
		//alg->ops.local_min_tol = 0; // -1e-1*f->tolerance;
	} else{ // this is Input 1, model from matlab
		f = new energy_auto<d_type>();
		mx_array<int, 2> E(prhs[0]);
		mx_array<double, 2> f1(prhs[1]);
		mx_array<double, 3> f2(prhs[2]);
		mx_array<int, 2> X(prhs[3]);
		if (nrhs >= 5){
			M = mx_array<double, 2>(prhs[4]);
		};
		if (nrhs >= 6){
			y = mx_array<int, 1>(prhs[5]);
		};
		if (nrhs >= 7){
			ops = mx_struct(prhs[6]);
		};
		//
		int K = f1.size()[0];
		int nV = f1.size()[1];
		int nE = E.size()[1];
		//
		f->set_nV(nV);
		f->set_E(E);
		f->K << K;
		f->maxK = K;
		for (int s = 0; s < nV; ++s){
			//f->f1[s].resize(K);
			//f->f1[s] << f1.subdim<1>(s);
			f->set_f1(s,f1.subdim<1>(s));
		};
		for (int e = 0; e < nE; ++e){
			dynamic::num_array<double, 2> df2; df2 = f2.subdim<2>(e);
			f->set_f2(e, df2);
		};
		f->init();
		f->report();
	};

	//ops["conv_tol"] = 1e-2*f->tolerance;
	//ops["gap_tol"] = 1e-1*f->tolerance;

	// allocate outputs
	mx_array<int, 2> P(mint2(f->maxK, f->nV()));
	mx_array<double, 2> M_out(mint2(f->maxK, f->nE()));
	mx_array<double, 2> hist;
	mx_array<double, 3> burn;

	burn.resize(exttype::mint3(f->maxK,f->nV(),4));

	alg->set_E(f);
	debug::PerformanceCounter c1;
	c1.start();
	alg->ops << ops;
	alg->init();
	alg->burn = &burn;
	//alg->ops.read(ops);
	alg->set_M(M);
	alg->set_P(P);
	//alg->set_X(X1);
	debug::PerformanceCounter c2;
	//alg->ops.it_batch = 1;
	//alg->alg_trws::ops->it_batch = 1;
	// do not do initial iterations if have y and M
	if (M.is_empty()){
		alg->run_converge();
	} else{//some quick initialization pass
		debug::stream << "Accepting provided reparametrization.\n";
		alg->ops["max_it"] = 1;
		alg->run_converge();
	};
	hist = alg->hist;
	debug::stream << "Starting PO. Init iterations: " << alg->total_it << " Init time:" << c2.time() << "\n";
	alg->get_M(M_out);
	mx_array<int, 2> y_out(mint2(1, alg->best_x.size()));
	//
	// partial optimality stuff
	//alg->ops.it_batch = 5;
	//int maxit = 50;
	//alg->ops.n_sensetivity = 1;
	//alg->ops.weak_po = false;
	if (y.length() > 0){
		intf yy; yy.resize(y.length());
		yy << y;
		alg->prove_optimality(yy);
	} else{
		alg->prove_optimality();
	};
	c1.stop();
	// after ICM step too
	y_out << alg->y;
	mx_array<double, 2> statistics; statistics.resize(mint2(1, 1));
	statistics[0] = c1.time();
	debug::stream << "computation time: " << statistics[0] << "\n";
	//
	if (nlhs >= 1){
		plhs[0] = y_out.get_mxArray_andDie();
	};
	if (nlhs >= 2){
		plhs[1] = P.get_mxArray_andDie();
	};
	if (nlhs >= 3){
		plhs[2] = M_out.get_mxArray_andDie();
	};
	if (nlhs >= 4){
		plhs[3] = statistics.get_mxArray_andDie();
	};
	if (nlhs >= 5){
		plhs[4] = hist.get_mxArray_andDie();
	};
	if (nlhs >= 6){
		plhs[5] = burn.get_mxArray_andDie();
	};
	delete alg;
	delete f;
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	mexargs::MexLogStream log("log/output.txt", false);
	debug::stream.attach(&log);
	mexargs::MexLogStream err("log/errors.log", true);
	debug::errstream.attach(&err);

	/*
	if (omp_in_parallel()){
		debug::stream << "woah!";
		omp_set_nested(true);
	};
	*/

	try{
		mexFunction_protect(nlhs, plhs, nrhs, prhs);
	} catch (std::exception & e){
		debug::stream << "!Exception:"<<e.what()<<"\n";
	};

	memserver::get_global()->clean_garbage();
	//memserver::get_global()->~block_allocator();
	debug::stream << "mem at exit: " << memserver::get_global()->mem_used() << " bytes\n";
	//
	debug::stream << "end of output.\n";
	//omp_set_num_threads(1);
	debug::stream.detach();
	debug::errstream.detach();
};
