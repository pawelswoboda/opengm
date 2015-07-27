#include "part_opt_TRWS.h"

#include "opengm_read.h"
#include "part_opt_TRWS1.h"
#include <conio.h>

#include "debug/msvcdebug.h"
#include "dynamic/dynamic.h"
#include "optim/graph/mgraph.h"
#include "debug/performance.h"

void test_opengm(){
	//std::string fname = "../../matlab/part_opt/datasets/color-seg/colseg-cow3.h5";
	//std::string fname = "../../matlab/part_opt/datasets/mrf-photomontage/pano-gm.h5";
	//std::string fname = "../../matlab/part_opt/datasets/mrf-stereo/ven-gm.h5";
	//std::string fname = "../../matlab/part_opt/datasets/mrf-stereo/tsu-gm.h5";
	std::string fname = "../../matlab/part_opt/datasets/color-seg-n4/pfau-small.h5";
	//
	//std::string fname = "Z:/work/dev/matlab/part_opt/PBP-bug.h5";
	// 97.0916% | 34.7012% / 27.8%
	//std::string fname = "C:/work/dev/matlab/part_opt/datasets/protein-folding/pdb1i24.h5";
	// 99.93% | 30.5795%
	//std::string fname = "../../matlab/part_opt/datasets/protein-folding/1CKK.h5";
	//std::string fname = "../../matlab/part_opt/datasets/protein-folding/1CM1.h5";
	//std::string fname = "../../matlab/part_opt/datasets/protein-folding/2BCX.h5";
	//std::string fname = "../../matlab/part_opt/datasets/protein-folding/1SY9.h5";
	//std::string fname = "C:/work/dev/matlab/part_opt/datasets/protein-folding/pdb1kwh.h5";
	//std::string fname = "../../matlab/part_opt/datasets/protein-folding/pdb1iqc.h5";
	//std::string fname = "../../matlab/part_opt/datasets/protein-folding/pdb1kgn.h5";
	// 83.2644% | 31.1208%
	//std::string fname = "C:/work/dev/matlab/part_opt/datasets/protein-folding/pdb1fmj.h5";
	energy_auto f;
	opengm_read(fname, "gm", f);
	debug::PerformanceCounter c0;
	alg_po_trws alg;
	alg.set_E(&f);
	//debug::stream << "constructor time: " << c0.time() << "s. \n";
	//alg.ops.reparam_zerotop = 0;
	//alg.ops.reduce_immovable = 0;
	alg.ops.use_cut = 1;
	alg.ops.fast_msg = 1;
	alg.ops.reduce_immovable = 1;
	alg.ops.use_pixel_cut = 1;
	//alg.ops.swoboda14 = true;
	//alg.ops.local_min_tol = 0.001;
	debug::PerformanceCounter c1;
	alg.init();
	double conv_tol = 0.01;
	alg.run_converge(conv_tol, 1000);
	if (alg.ops.swoboda14){
		/*
		alg.prove_optimality_swoboda14(1e-8, 20);
		int opt = 0;
		for (int v = 0; v < alg.nV; ++v){
		if (alg.isA(v))++opt;
		};
		debug::stream << "old PO: (" << double(opt)  <<")" << double(opt) / alg.nV * 100 << "%\n";
		*/
	} else{
		alg.ops.it_batch = 5;
		alg.prove_optimality(conv_tol, 50);
	};
	double elim = double(alg.maxelim + alg.nV - alg.nimmovable) / double(alg.maxelim) * 100;
	double time1 = c1.time();
	debug::stream << "INSTANCE " << fname << "\n nit: " << alg.total_it << " elim:" << elim << "% / " << time1 << "s" "\n";
};

void test_rand(int rand_inst, alg_po_trws_ops & ops, int & nit, double & elim, double & time){
	using exttype::mint2;
	using exttype::mint3;
	using exttype::mint4;

	datastruct::mgraph G;
	int M = 20;
	int N = 20;
	int K = 10;
	G.create_grid<2, 0>(mint2(M, N));
	G.edge_index();
	int nV = G.nV();
	int nE = G.nE();

	dynamic::num_array<int, 2> f1(mint2(K, nV));
	dynamic::num_array<double, 3> f2(mint3(K, K, nE));
	dynamic::num_array<int, 2> X(mint2(K, nV));
	dynamic::num_array<int, 2> P(mint2(K, nV));
	dynamic::num_array<int, 2> E(mint2(2, nE));

	srand(rand_inst);
	//generate random unaries
	for (int s = 0; s < nV; ++s){
		for (int k = 0; k < K; ++k){
			f1(k, s) = rand() % 100;
		};
	};
	if (1){
		// generate random potts model
		for (int e = 0; e < nE; ++e){
			f2.subdim<2>(e) << 0;
			int gamma = rand() % 70;
			for (int k = 0; k < K; ++k){
				f2(k, k, e) = -gamma;
			};
		};
	} else{
		// generate random full model
		for (int e = 0; e < nE; ++e){
			f2.subdim<2>(e) << 0;
			for (int k1 = 0; k1 < K; ++k1){
				for (int k2 = 0; k2 < K; ++k2){
					f2(k1, k2, e) = rand() % 50;
				};
			};
		};
	}
	X << 0;
	for (int e = 0; e < nE; ++e){
		E(0, e) = G.E[e][0];
		E(1, e) = G.E[e][1];
	};

	//part_opt_TRWS(E, f1, f2, X, P);
	energy_auto f;
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

	alg_po_trws alg;
	alg.ops = ops;
	//alg.init(E, f1, f2, X, P);
	alg.set_E(&f);
	debug::PerformanceCounter c1;
	alg.init();
	//alg.set
	alg.run_converge(1e-8, 100);
	//return;
	if (ops.swoboda14){
		//alg.prove_optimality_swoboda14(1e-8, 20);
	} else{
		alg.ops.it_batch = 5;
		alg.prove_optimality(1e-8, 50);
	};
	time = c1.time();
	elim = double(alg.maxelim + alg.nV - alg.nimmovable) / double(alg.maxelim) * 100;
	nit = alg.total_it;
	debug::stream << "TEST INSTANCE " << rand_inst << " nit: " << nit << " elim:" << elim << "% / " << time << "s" "\n";
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

void main(){

	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDOUT);
	int tmp;
	tmp = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	tmp = tmp | _CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF;
	_CrtSetDbgFlag(tmp);


	test_opengm();
	std::cin.get();
	return;

	alg_po_trws_ops ops;
	ops.fast_msg = true;
	for (int inst = 2; inst <= 2; ++inst){
		int nit1; double elim1; double time1;
		//ops.use_cut = 1;
		//ops.local_min_tol = -0.5;
		//ops.swoboda14 = true;
		test_rand(inst, ops, nit1, elim1, time1);
		debug::stream << "TEST INSTANCE " << inst << " nit: " << nit1 << " elim:" << elim1 << "% / " << time1 << "s" "\n";
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
	};

	std::cin.get();
};