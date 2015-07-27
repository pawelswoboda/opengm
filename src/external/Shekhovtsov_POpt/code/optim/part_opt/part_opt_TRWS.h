#ifndef part_opt_trws_h
#define part_opt_trws_h

#include <queue>


#include "dynamic/num_array.h"
#include "dynamic/array_allocator.h"
#include "dynamic/fixed_array1.h"
#include "dynamic/fixed_array2.h"
#include "exttype/fixed_vect.h"
#include "optim/graph/mgraph.h"
#include "optim/trws/stream_graph.h"
#include "debug/performance.h"

#include "energy.h"
#include "trws_machine.h"

using namespace dynamic;

#define FINF std::numeric_limits<VALUE_TYPE>::infinity();
//typedef sse_float_4 vectorizer;
//typedef d_type type;
//typedef d_vectorizer vectorizer;
typedef trws_machine<d_type, d_vectorizer> alg_trws;
typedef energy<d_type> tenergy;
//
//_________________________alg_po_trws_________________________________
//! class for maximum persistency with message passing solver
class alg_po_trws : public alg_trws{
public:
	typedef alg_trws parent;
	//typedef tenergy::t_f2 t_f2;
	typedef alg_trws::t_f2 t_f2;
public:
	tenergy * E0; // initial input energy
	//energy F; // manipulated energy
	//dynamic::fixed_array1<t_f2*> ff2;
public://data
	fixed_array1<po_mask> UU; // lists of immovable nodes
	std::queue<node_info*> dee_q;
	//
	num_array<int, 2> P; // improving mapping
	//num_array<int, 2> X; // alive labels
	num_array<double, 3> * burn;
public: // options
	alg_po_trws_ops ops;
	intf y; // labeling to which we will reduce
private://internal and temp vars
	tvect v1;
	tvect v2;
	bool f2_replaced;
	intf y1;
private:
	static intf y0;
	int dee_mark;
	int dee_mark_strong;
public://statistics
	int nimmovable;
	//int total_it;
	int maxelim;
	int po_it;
private:
	virtual t_f2 * construct_f2(int e, aallocator * al)override;
protected:
	void finish();
	void cut(intf & x);
	bool cut_tr(char * caller);
	bool flat_cuts();
	bool apply_cut(intf & x, char * caller);
	void set_y(intf & y);
	void rebuild_energy();
	void reduce_immovable();
	void expansion_move();
	double dee_delta(node_info & v, int k);
	double icm_delta(node_info & v, int k);
	void icm();
	int mark_dee();
	void mark_WTA();
	void unmark();
	void queue_neib(node_info & v);
	typedef enum { INIT = 0, WTA = 1, CUT = 2, PXCUT = 3} prunner;
	void mark_immovable(node_info & v, int k, prunner who);
	void report_po_iter(int total_mark1, int total_mark2, std::string caller);
	void init_aux();
	bool check_zero_gap();
	//
	std::pair<double,int> marg(int s);
	bool remove_condition(int s);
	bool stop_condition();
public://input
	alg_po_trws();
	~alg_po_trws(){
		cleanup();
	};
	void cleanup();
	void set_E(tenergy * E){
		this->E0 = E;
		nV = E->G.nV();
		nE = E->G.nE();
		maxK = E->maxK;
	};
	void init(){
		parent::init(E0);
		maxelim = E0->K.sum() - nV;
		init1();
		ops.print();
	};
	void init1();
	//void set_X(num_array<int, 2> & X){
	//	this->X.set_ref(X);
	//};
	void set_P(num_array<int, 2> & P){
		this->P.set_ref(P);
	};
public://run
	//void prove_optimality(double eps, int maxit);
	void prove_optimality(intf & y = y0);
public: //performance
	debug::PerformanceCounter po_cuts;
	debug::PerformanceCounter po_msgs;
	debug::PerformanceCounter po_node_cuts;
	debug::PerformanceCounter po_total;
	debug::PerformanceCounter po_rebuild;
	/*
public: //implementation of Swoboda-14 method, comparison
	bool isA(int s);// is s in the set A
	bool isB(int s);// is s in the (inner) boundary of A
	void prove_optimality_swoboda14(double eps, int maxit);
	void rebuild_energy_swoboda14();
	*/
};

#endif