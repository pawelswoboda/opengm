#ifndef trws_machine_h
#define trws_machine_h

#include "dynamic/num_array.h"
#include "dynamic/array_allocator.h"
#include "dynamic/fixed_array1.h"
#include "dynamic/fixed_array2.h"
#include "exttype/fixed_vect.h"
#include "optim/graph/mgraph.h"
#include "optim/trws/stream_graph.h"
#include "debug/performance.h"
#include "energy.h"

#include "msg_alg.h"
#include "vectorizers.h"
#include "aallocator.h"


using namespace dynamic;

/*! type - basic type: field or double, vectorizer = type x N, implementing arithmetic operations with sse
*/
template<class type, class vectorizer> class trws_machine : public msg_alg{
public:
	typedef exttype::fixed_vect<type> tvect;
	typedef term2v<type, vectorizer> t_f2;
protected:
	struct node_info;
	struct edge_info;
	struct node_info_core;
	struct edge_info_core{
		vectorizer * msg;
		t_f2 * f2;
		node_info_core * _tail;
		node_info_core * _head;
		//
		edge_info_core():msg(0),f2(0),_tail(0),_head(0){
		};
		const node_info_core * tail()const{
			return _tail;
		};
		const node_info_core * head()const{
			return _head;
		};
		template<bool dir_fw>
		const node_info_core * dir_tail()const{
			if (dir_fw){
				return tail();
			} else{
				return head();
			};
		};
	};
	struct node_info_core{ // layout: [ node_info_core ] [edge_info_core in_1] ... [edge_info_core in_n] [edge_info_core out_1] [edge_info_core out_n]
		node_info_core * next; // next in the layer
		vectorizer * data; // data layout [ r1 ] [ f1 ] [ bw ] [ msg_out 1 , data 1] ... [ msg_out n , data n], messages can be of varied size 
		//vectorizer * data_end;
		int vK; // size in vectorizers
		int n_in;
		int n_out;
		type coeff;
		node_info * node;
		edge_info_core * __out_b;
		//
		int x;
		int y;
		//int offset_next; // unused
	private:
		friend class trws_machine;
		//vectorizer * __bw;
		//vectorizer * __f1;
		vectorizer * _bw(){ return data + vK; };
		vectorizer * _f1(){ return data + vK * 2; };
		vectorizer * const _bw()const{ return data + vK; };
		const vectorizer * _f1()const{ return data + vK * 2; };
		edge_info_core * _in_begin(){ return (edge_info_core*)(this + 1); };
		edge_info_core * _out_begin(){ return (edge_info_core*)(in_begin() + n_in); };
		void init(){
			//__bw = _bw();
			//__f1 = _f1();
			__out_b = _out_begin();
		};
	public:
		node_info_core() :x(0), y(0),next(0),data(0),node(0){};
		//vectorizer *& r1(){ return data; };
		vectorizer * const r1()const { return data; };
		//vectorizer * const & bw()const { return __bw; };
		//vectorizer * const & f1()const { return __f1; };
		vectorizer * const bw()const { return _bw(); };
		const vectorizer * f1()const { return _f1(); };
		//
		edge_info_core * in_begin(){ return (edge_info_core*)(this + 1); };
		edge_info_core * out_begin(){ return __out_b; };
		const edge_info_core * in_begin()const{ return (edge_info_core*)(this + 1); };
		edge_info_core * const & out_begin()const{ return __out_b; };
		//
		template<bool dir_fw> const edge_info_core * dir_in_begin()const{
			if (dir_fw){
				return in_begin();
			} else{
				return out_begin();
			};
		};
		template<bool dir_fw> const edge_info_core * dir_out_begin()const{
			if (dir_fw){
				return out_begin();
			} else{
				return in_begin();
			};
		};
		//
		template<bool dir_fw> int dir_n_in()const{
			if (dir_fw){
				return n_in;
			} else{
				return n_out;
			};
		};
		template<bool dir_fw> int dir_n_out()const{
			return dir_n_in<!dir_fw>();
		};
		template<bool dir_fw> const edge_info_core & dir_in(int i)const{
			if (dir_fw){
				return in_begin()[i];
			} else{
				return out_begin()[i];
			};
		};
		template<bool dir_fw> const edge_info_core & dir_out(int i)const{
			return dir_in<!dir_fw>(i);
		};
	};
	//
	struct node_info{
		int K;
		node_info_core * core;
		tvect f1;
		tvect r1;
		tvect bw;
		//dynamic::fixed_array1<vectorizer, array_allocator<vectorizer, sizeof(vectorizer)> > aligned_block;
		dynamic::fixed_array1<edge_info *> in; // incoming edges
		dynamic::fixed_array1<edge_info *> out; // outcoming edges
		bool active;
		//int x;
		// find edge_info in the incoming list (only needed when rebuilding)
		int find_in(edge_info * e){
			for (int i = 0; i < in.size(); ++i){
				if (in[i] == e)return i;
			};
			return -1;
		};
		// find edge_info in the outcoming list (only needed when rebuilding)
		int find_out(edge_info * e){
			for (int i = 0; i < out.size(); ++i){
				if (out[i] == e)return i;
			};
			return -1;
		};
	};
	struct edge_info{
		edge_info_core * core;
		node_info * head;
		node_info * tail;
		tvect msg;
		//t_f2 * _f2;
		edge_info():core(0),head(0),tail(0){};
		t_f2 *& f2(){ 
			assert(core);
			assert(core->f2);
			return core->f2; 
		};
		//t_f2 *& f2(){ return _f2; };
	};
private:
	alg_trws_ops _ops;
private:
	aallocator aligned_block;
	template<typename type2> struct line_compute_struct{
	public:
		double E;
		double LB;
		type2 dPhi;
		type2 margin;
		dynamic::fixed_vect<type, array_allocator<type, 16> > t1; // temp vars
		dynamic::fixed_vect<type, array_allocator<type, 16> > t2; // temp vars
		dynamic::fixed_vect<type, array_allocator<type, 16> > t3; // temp vars
		dynamic::fixed_vect<type, array_allocator<type, 16> > t4; // temp vars
		//dynamic::fixed_vect<type, array_allocator<type, 16> > data;
		//tvect t1, t2, t3, t4;
		line_compute_struct(){};
		line_compute_struct(int maxK){
			init(maxK);
		};
		void init(int maxK){
			E = 0;
			LB = 0;
			dPhi = 0;
			margin = INF(double);
			if (maxK == 0){
				return;
			};
			//int V = sizeof(vectorizer) / sizeof(type);
			//assert(maxK % V == 0);
			int maxKa = v_align<type, vectorizer>(maxK);
			//data.resize(maxKa * 4);
			//t1.set_ref(&data[maxKa*0], maxKa);
			//t2.set_ref(&data[maxKa*1], maxKa);
			//t3.set_ref(&data[maxKa*2], maxKa);
			//t4.set_ref(&data[maxKa*3], maxKa);
			t1.resize(maxKa);
			t2.resize(maxKa);
			t3.resize(maxKa);
			t4.resize(maxKa);
		};
		template <typename type3> void aggregate(line_compute_struct<type3> & C){
			E += C.E;
			C.E = 0;
			LB += C.LB;
			C.LB = 0;
			dPhi = std::max(dPhi, type2(C.dPhi));
			margin = std::min(margin, type2(C.margin));
		};
	};
public:
	int nV;
	int nE;
	int maxK;
	alg_trws_ops * ops;
	energy<type> * E;
	private:
	long long data_total;
	bool debug_LB;
private:
	//char * info_core; // destroy this afterwards
protected:
	dynamic::fixed_array1<node_info> nodes; //nodes
	dynamic::fixed_array1<edge_info> edges; //edges
	stream_graph SG;
protected:
	type vertex_coeff(node_info & v, type multiplier = 1);
	template<bool dir_fw, bool measure> void line_update(const node_info_core * v, const node_info_core * v_end, line_compute_struct<type> & temp);
	//template<bool dir_fw, bool measure> void line_warmup(node_info_core * v, node_info_core * v_end);
	template<bool dir_fw, bool measure, bool parallel> void run_dir();
	template<bool measure, bool parallel> void run_forward_backward();
	void init_iteration();
	void renormalize_to_y();
	void run(int niter);
	void set_f2(edge_info & ee, t_f2 * f2);
	virtual t_f2 * construct_f2(int e, aallocator * al);
	void set_y(const intf & y);
public:
	trws_machine();
	~trws_machine();
	size_t size_required();
	void init(energy<type> * E);
protected:
	void init_core();
	void destroy_core();
public:
	virtual double cost(const intf & x)const override;
	virtual void run_converge(double target_E = -INF(type)) override; //! run for at most 10*niter iterations or precision eps or until energy below target_E is achieved
public: // other interface
	virtual void get_M(num_array<double, 2> & M) override;
	virtual void set_M(const num_array<double, 2> & M) override;
public:
	void check_get_col();
private:
	int get_num_chunks()const;
	struct chunk_iterator{
		int th_id;
		trws_machine<type, vectorizer> * p;
		chunk_iterator(trws_machine<type, vectorizer> * _p):p(_p){
			restart();
		};
		chunk_iterator(const chunk_iterator & I) :p(I.p),th_id(I.th_id){
		};
		void restart(){
			th_id = 0;
		};
		bool valid()const;
		chunk_iterator & operator ++(){// increase 
			++th_id;
			return *this;
		};
		chunk_iterator next()const{
			chunk_iterator I(p);
			I.th_id = th_id;
			++I;
			return I;
		};
	};

	struct slice_iterator{
		const chunk_iterator & ci;
		int l; // layer, second
	public:
		int th_id()const{ return ci.th_id; };
	public:
		slice_iterator(const chunk_iterator & _ci) :ci(_ci){
			restart();
		};
		slice_iterator(const slice_iterator & _si, const chunk_iterator & _ci) :ci(_ci){
			l = _si.l;
		};
		void restart(){
			l = 0;
		};
		int line_begin()const;
		int line_end()const;
		node_info * first()const;
		node_info * last()const;
		bool valid()const;
		slice_iterator & operator ++(){// increase 
			++l;
			return *this;
		};
	};

	struct line_iterator{
	private:
		trws_machine<type, vectorizer> * p;
		int i; // index within layer
	public:
		const slice_iterator & si;
		int th_id()const{ return si.th_id(); };// chunk, fixed
		int l()const{ return si.l; };
	public:
		//! node index;
		int s()const; 
	public:
		line_iterator(const slice_iterator & _si) :si(_si),p(_si.ci.p){
			restart();
		};
		line_iterator(const line_iterator & _li, const slice_iterator & _si) :si(_si), p(_si.ci.p){
			i = _li.i;
		};
		void restart(){
			i = si.line_begin();
		};
		bool valid()const;

		//int line_begin_node()const{
		//};
		//int line_end_node()const{
		//};
		line_iterator next()const{
			line_iterator L(si);
			L.i = i + 1;
			return L;
		};
		line_iterator & operator ++(){
			++i;
			return *this;
		};
	};

	struct full_iterator{
		trws_machine<type, vectorizer> * p;
	public:
		chunk_iterator ci;
		slice_iterator si;
		line_iterator li;
		int s()const{
			return li.s();
		};
		full_iterator(trws_machine<type, vectorizer> * _p) :p(_p), ci(_p), si(ci), li(si){
		};
		full_iterator(const full_iterator & I) :p(I.p), ci(I.ci), si(I.si, ci), li(I.li, si){
		};
		bool valid()const{
			return li.valid();
		};
		full_iterator & operator ++(){
			while (ci.valid()){
				while (si.valid()){
					while (li.valid()){
						++li;
						if (li.valid())return *this; // found valid entry
					};
					++si;
					li.restart();
					if (li.valid())return *this; // found valid entry
				};
				++ci;
				si.restart();
				li.restart();
				if (li.valid())return *this; // found valid entry
			};
			// no more valid entries
			return *this;
		};
		full_iterator next()const{
			full_iterator I(*this);
			++I;
			return I;
		};
	};
};


#endif