#include "part_opt_TRWS.h"

#include "debug/msvcdebug.h"
#include "dynamic/dynamic.h"
#include "optim/graph/mgraph.h"
#include "debug/performance.h"



void test_rand(int rand_inst, int ptype, options & ops, int & nit, double & elim, double & time){
	using exttype::mint2;
	using exttype::mint3;
	using exttype::mint4;

	datastruct::mgraph G;
	int M = 320;
	int N = 240;
	int K = 12;
#ifdef _DEBUG
	//M = 20;
	//N = 20;
	//K = 9;
	N = 4;
	M = 4;
	K = 9;
#endif
	//N = 90;
	//M = 90;
	//K = 12;

	G.create_grid<2, 0>(mint2(M, N));
	G.edge_index();
	int nV = G.nV();
	int nE = G.nE();

	dynamic::num_array<double, 2> f1(mint2(K, nV));
	dynamic::num_array<double, 3> f2(mint3(K, K, nE));
	dynamic::num_array<int, 2> X(mint2(K, nV));
	dynamic::num_array<int, 2> P(mint2(K, nV));
	dynamic::num_array<int, 2> E(mint2(2, nE));

	srand(rand_inst);
	//generate random unaries
	for (int s = 0; s < nV; ++s){
		for (int k = 0; k < K; ++k){
			f1(k, s) = double(rand() % 10000)/100;
		};
	};
	if (ptype == 3){// generate random tquadratic model
		for (int e = 0; e < nE; ++e){
			//f2.subdim<2>(e) << 0;
			double gamma = (rand() % 6000)/100+1;
			double th = ((rand() % 4000 + 3000)*gamma)/100;
			for (int k1 = 0; k1 < K; ++k1){
				for (int k2 = 0; k2 < K; ++k2){
					double v = std::min(math::sqr(k1 - k2)*gamma, th);
					f2(k1, k2, e) = v;
				};
			};
		};
	};
	if (ptype == 2){// generate random tlinear model
		for (int e = 0; e < nE; ++e){
			//f2.subdim<2>(e) << 0;
			int gamma = rand() % 20 + 1;
			int th = (rand() % 2 + 2)*gamma;
			for (int k1 = 0; k1 < K; ++k1){
				for (int k2 = 0; k2 < K; ++k2){
					f2(k1, k2, e) = std::min(std::abs(k1 - k2)*gamma, th);
				};
			};
		};
	};
	if (ptype == 1){
		// generate random potts model
		for (int e = 0; e < nE; ++e){
			double  gamma = double(rand() % 5500)/100;
			f2.subdim<2>(e) << gamma;
			for (int k = 0; k < K; ++k){
				f2(k, k, e) = 0;
			};
		};
	};
	if(ptype == 0){
		// generate random full model
		for (int e = 0; e < nE; ++e){
			f2.subdim<2>(e) << 0;
			for (int k1 = 0; k1 < K; ++k1){
				for (int k2 = 0; k2 < K; ++k2){
					f2(k1, k2, e) = double(rand() % 5000)/100;
				};
			};
		};
	};
	X << 0;
	for (int e = 0; e < nE; ++e){
		E(0, e) = G.E[e][0];
		E(1, e) = G.E[e][1];
	};
	// varied size
	energy_auto<d_type> f;
	f.set_nV(nV);
	f.set_E(E);
	f.K << K;
	f.maxK = K;
	
	for (int s = 0; s < nV; ++s){
		//f.K[s] = (rand() % (K-1)) + 1;
	};

	for (int s = 0; s < nV; ++s){
		//f.f1[s].resize(f.K[s]);
		//f.f1[s] << f1.subdim<1>(s);
		num_array<double, 1> a(f.K[s]);
		a << f1.subdim<1>(s);
		f.set_f1(s,a);
	};
	for (int e = 0; e < nE; ++e){
		//f.f2(e).resize(mint2(K, K));
		//f.f2(e) << f2.subdim<2>(e);
		int s = G.E(e)[0];
		int t = G.E(e)[1];
		num_array<double, 2> a(mint2(f.K[s], f.K[t]));
		a << f2.subdim<2>(e);
		f.set_f2(e, a);
	};
	
	/*
	//part_opt_TRWS(E, f1, f2, X, P);
	energy_auto<type> f;
	f.set_nV(nV);
	f.set_E(E);
	f.K << K;
	f.maxK = K;
	for (int s = 0; s < nV; ++s){
		f.f1[s].resize(K);
		f.f1[s] << f1.subdim<1>(s);
	};
	for (int e = 0; e < nE; ++e){
		//f.f2(e).resize(mint2(K, K));
		//f.f2(e) << f2.subdim<2>(e);
		f.set_f2(e, f2.subdim<2>(e));
	};
	*/
	//debug::stream << "Recognized models: " << f.npotts << "Potts terms, " << f.nfull << " full terms\n";
	f.init();
	f.report();
	//f.tolerance = 1e-3;
	alg_po_trws alg;
	alg.ops << ops;
	//alg.ops.gap_tol = 1e-5;
	//alg.ops.sensetivity = -0.01;
	//alg.init(E, f1, f2, X, P);
	alg.set_E(&f);
	alg.init();
	//alg.check_get_col();
	//alg.set
	debug::PerformanceCounter c1;
	//alg.ops.it_batch = 1;
	//alg.alg_trws::ops->it_batch = 1;
	//alg.ops.it_batch = 5;
	alg.run_converge();
	debug::stream << "TRW-S time:" << c1.time() << "\n";
	//alg.ops.reduce_immovable = false;
	//alg.ops.fast_msg = false;
	//alg.ops.use_cut = 1;
	//alg.ops.use_pixel_cut = 1;
	//alg.ops.it_batch = 5;
	alg.prove_optimality();

	//alg.check_get_col();

	time = c1.time();
	elim = double(alg.maxelim + alg.nV - alg.nimmovable) / double(alg.maxelim) * 100;
	nit = alg.total_it;
	//debug::stream << "TEST INSTANCE " << rand_inst << " nit: " << nit << " elim:" << elim << "% / " << time << "s" "\n";
	/*
	{
	using namespace version1;
	version1::alg_po_trws alg;
	alg.ops = ops;
	//alg.init(E, f1, f2, X, P);
	alg.set_E(&f);
	debug::PerformanceCounter c1;
	alg.init();
	//alg.set
	alg.run_converge(1e-8, 100);
	if (ops.swoboda14){
	//alg.prove_optimality_swoboda14(1e-8, 20);
	} else{
	alg.prove_optimality(1e-8, 20);
	};
	time = c1.time();
	elim = double(alg.maxelim + alg.nV - alg.nimmovable) / double(alg.maxelim) * 100;
	nit = alg.total_it;
	debug::stream << "TEST INSTANCE " << rand_inst << " nit: " << nit << " elim:" << elim << "% / " << time << "s" "\n";
	};
	*/
};

int main(){

	//alg_po_trws_ops ops;
	//ops.fast_msg = true;
	options ops;
	//ops["fast_msg"] = 1;
	ops["max_CPU"] = 1;
	//ops["max_it"] = 1;
	//ops["it_batch"] = 1;
	//ops["po_max_it"] = 1;
	//ops["po_it_batch"] = 1;
	for (int inst = 1; inst <= 10; ++inst){
		for (int ptype = 0; ptype < 4; ++ptype){
#ifndef _DEBUG
			ptype = 1;
#endif
			int nit1; double elim1; double time1;
			//ops.use_cut = 1;
			//ops.local_min_tol = -0.5;
			//ops.swoboda14 = true;
			test_rand(inst, ptype, ops, nit1, elim1, time1);
			//expecting 4.61967 %
			// 100 initial it: 4.46% 51s double | 4.46615% 32s (2.3 init) floatx4 |
			debug::stream << "TYPE:"<< ptype << " TEST INSTANCE " << inst << " nit: " << nit1 << " elim:" << elim1 << "% / " << time1 << "s" "\n";
			/*
			ops.cut_type = 0;
			int nit1; double elim1;
			test_rand(inst,ops,nit1,elim1);
			ops.cut_type = 1;
			int nit2; double elim2;
			test_rand(inst, ops, nit2, elim2);
			debug::stream << "TEST INSTANCE " << inst << " nit: " << nit1 << " elim:" << elim1 << "\n";
			//if (nit1 != nit2 || elim1 != elim2){
			if (elim1 != elim2){
			debug::stream << "CUT with QPBO differs, nit: " << nit2 << " elim:" << elim2 << "\n";
			throw debug_exception("bla");
			};
			*/
#ifndef _DEBUG
			std::cin.get();
			return 0;
#endif
		};
	};

	std::cin.get();
};