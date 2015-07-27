#ifndef msg_alg_h
#define msg_alg_h

#include "dynamic/num_array.h"
//#include "dynamic/array_allocator.h"
#include "dynamic/fixed_array1.h"
#include "exttype/fixed_vect.h"
//#include "dynamic/fixed_array2.h"
//#include "optim/graph/mgraph.h"
//#include "optim/trws/stream_graph.h"
//#include "debug/performance.h"
//#include "energy.h"

#include <limits>
#include "dynamic/options.h"

class msg_alg;

class alg_trws_ops : public options{
public:
	NEWOPTION(double, rel_gap_tol, 0.1);
	NEWOPTION(double, rel_conv_tol, 0.01);
	NEWOPTION(int, it_batch, 10);
	NEWOPTION(int, max_it, 1000);
	NEWOPTION(int, max_CPU, 2); // parallelization
	double gap_tol;
	double conv_tol;
};

//_________________________alg_po_trws_________________________________
class alg_po_trws_ops : public alg_trws_ops{
private: // depricated
	int cut_type; // 0 - truncated, 1 -qpbo
	bool use_local_condition;
	bool reparam_zerotop; // [depricated] apply reparametrization on initial energy
	bool swoboda14; // use method of Swoboda 2014 CVPR, for comparison
public:
	NEWOPTION(bool, use_cut, true);
	NEWOPTION(bool, use_pixel_cut, true);
	NEWOPTION(bool, reduce_immovable, true); // use reduction, making equal immovable nodes
	NEWOPTION(bool, fast_msg, true); // use fast message passing
	NEWOPTION(bool, weak_po, false); // weak or strong perssitency, default strong
	NEWOPTION(double, sensetivity, 0);
	NEWOPTION(double, local_min_tol, 0);
	NEWOPTION(int, n_sensetivity, 1);
	NEWOPTION(int, po_it_batch, 5); // TRW-S iterations block in PO
	NEWOPTION(int, po_max_it, 50);
	/*
	//
	bool use_cut;
	bool use_pixel_cut;
	bool reduce_immovable; 
	double local_min_tol;
	bool fast_msg; 
	bool weak_po; 
	double sensetivity;
	int n_sensetivity;
	int po_it_batch; 
	int po_max_it;
public:
	void read(options & ops);
	void write(options & ops);
	alg_po_trws_ops(){
		//use the same defaults as in the read
		read(options());
	};
	*/
};


class msg_alg{
public:
	//int nV;
	//int nE;
	//int maxK;
public: // initialization
	/*
	void set_E(energy * E){
		this->E = E;
		nV = E->G.nV();
		nE = E->G.nE();
		maxK = E->maxK;
	};
	int K(int v){ return E->K[v]; };
	*/
public: // interface
	virtual double cost(const dynamic::intf & x)const = 0;
	virtual void get_M(dynamic::num_array<double, 2> & M) = 0;//! get messages as array
	virtual void set_M(const dynamic::num_array<double, 2> & M) = 0; //! set messages
	virtual void run_converge(double target_E = -std::numeric_limits<double>::infinity()) = 0; //! run for at most 10*niter iterations or precision eps or until energy below target_E is achieved
public: //output statistics
	double LB;
	double margin;
	dynamic::intf current_x;
	dynamic::intf best_x;
	double current_E;
	double best_E;
	double dPhi;
	int total_it;
	dynamic::num_array<double, 2> hist;
	enum{ iterations, precision, target_energy, zero_gap, error} exit_reason;
	msg_alg(){
		total_it = 0;
		best_E = std::numeric_limits<double>::infinity();
	};
};

#endif