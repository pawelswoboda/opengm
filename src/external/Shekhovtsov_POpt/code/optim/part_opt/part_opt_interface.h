#ifndef part_opt_interface_h
#define part_opt_interface_h

template<typename type> class energy_auto;
class alg_po_trws;

#include <string>
#include "dynamic/options.h"

template<typename type>
class part_opt_interface{
public:
	typedef float compute_type;
	energy_auto<compute_type> * energy;
	alg_po_trws * alg;
	options * ops;
public:
	std::string instance_name;
public:
	part_opt_interface();
	~part_opt_interface();
protected:
	int nV;
	int nE;
	int maxK;
protected: // convert energy
	void energy_read_start(int nV);
	// loop 0
	void energy_read_vertex(int v, int K);
	void energy_read_edge(int u, int v);
	void energy_init0();
	// loop 1
	void energy_read_f1(int v, int K, double * pval);
	void energy_read_f2(int u, int v, int K1, int K2, double * pval);
	void energy_init1();
protected: // run algorithm
	void clear_alg();
	void alg_run();
	bool is_alive(int v, int k);
public:
	double time_init;
	double time_total;
	double time_po;
	int iter_init_trws;
	int iter_po;
	int iter_po_trws;
};

#endif