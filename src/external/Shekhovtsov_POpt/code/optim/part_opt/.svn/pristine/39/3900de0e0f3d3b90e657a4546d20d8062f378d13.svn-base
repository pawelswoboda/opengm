#ifndef energy_h
#define energy_h

#include "dynamic/fixed_array1.h"
#include "dynamic/fixed_array2.h"
#include "exttype/fixed_vect.h"
#include "optim/graph/mgraph.h"
#include "geom/math.h"
#include <bitset>
#include <limits>

#include "vectorizers.h"
#include "aallocator.h"

//#define VALUE_TYPE float
//#define FINF std::numeric_limits<double>::infinity()
//#define IINF std::numeric_limits<int>::max()

#define INF(type) std::numeric_limits<type>::infinity()
typedef std::bitset<512> po_mask;

//typedef float VALUE_TYPE;
typedef float d_type;
typedef sse_float_4 d_vectorizer;

//typedef float VALUE_TYPE;
//typedef float type;
//typedef float vectorizer;

//typedef double VALUE_TYPE;
//typedef double type;
//typedef double vectorizer;


//int size_align(int size);
//void check_align(int size);
//exttype::mint2 size_align(const exttype::mint2 & size);
template<typename type, typename vectorizer>int v_align(int size); // aligns size measured in types to integer # of vectorizers

using namespace exttype;
using namespace dynamic;

// itroduce energy with virtual pairwise functions, currently a virtual function returns full pw term
// abstract pw function:
// needed capablities: 
// min_sum, min_sum_t
// get_row, get_column, get_full
// min_sum_mask, min_sum_mask_t
//typedef sse_float_4 vectorizer;
//typedef sse_vect_scalar<VALUE_TYPE,4> tvect;
//typedef exttype::fixed_vect<float> tvect;


template<typename type>
class term2{
public:
	typedef exttype::fixed_vect<type> tvect;
	typedef term2<type> tthis;
public:
	virtual ~term2(){};
	virtual void min_sum(tvect & a, tvect & r)const = 0;
	virtual void min_sum_t(tvect & a, tvect & r)const{
		return this->min_sum(a, r);
	};
	virtual type operator()(int i, int j) const = 0;
	virtual void get_row(int i, tvect & r) const = 0;
	virtual void get_col(int j, tvect & r) const = 0;
public: // some compatibility stuff
	virtual int count(int i) const = 0;
public: // methods for partial optimality
	virtual void reduce(po_mask & U_s, int y_s, po_mask & U_t, int y_t){};//defaults to empty
	virtual void rebuild(po_mask & U_s, int y_s, po_mask & U_t, int y_t){};//defaults to empty
public:
	virtual tthis * copy(aallocator * al)const = 0;
	//virtual void * data_ptr(){return 0; };
};

template<typename ttype, typename tvectorizer>
class term2v : public term2<ttype>{
public:
	typedef ttype type;
	typedef tvectorizer vectorizer;
	typedef typename term2<type>::tvect tvect;
public: // optimized interface:
	virtual void get_col_e(const int vK1, int j0, vectorizer * message)const = 0;
	virtual void min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const{
		// this is to check correctness, otherwise nested virtual function is not good
		// default to min_sum regular vector implementation
		int V = sizeof(vectorizer) / sizeof(type);
		int K1 = vK1*V;
		tvect a; a.set_ref(source, K1);
		tvect r; r.set_ref(message, K1);
		this->min_sum(a, r);
	};
	virtual void min_sum_et(const int vK1, vectorizer * source, vectorizer * message)const{
		// default to min_sum_t regular vector implementation
		int V = sizeof(vectorizer) / sizeof(type);
		int K1 = vK1*V;
		tvect a; a.set_ref(source, K1);
		tvect r; r.set_ref(message, K1);
		this->min_sum_t(a, r);
	};
};

/*
template<typename ttype, typename tvectorizer>
class term2_vcore{
public:
	typedef ttype type;
	typedef tvectorizer vectorizer;
public:
	term2<type> * src; // pointer to source term2
	//typedef typename term2<type>::tvect tvect;
public: // optimized interface:
	//! pass message in forward direction: minimize over first index
	virtual void min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const;
	//! pass message in backward direction 
	virtual void min_sum_et(const int vK1, vectorizer * source, vectorizer * message)const;
	//! callback for allocating data and initialization
	virtual void init(aallocator & al, bool mem_valid);
};
*/

//_______________________term2v_matrix______________________________

template<typename type, typename vectorizer>
class term2v_matrix : public term2v<type, vectorizer>, public num_array<type, 2, array_allocator<type, 16> > {//public exttype::ivector<fixed_matrix<type>, dynamic::fixed_array2<type> >{
private:
	typedef num_array<type, 2, array_allocator<type, 16> > parent;
	typedef typename term2v<type, vectorizer>::tvect tvect;
	typedef term2v_matrix<type, vectorizer> tthis;
public:
	// virtual void * data_ptr()override{return this->begin(); };
public:
	using parent::begin;
	//empty constructor
	term2v_matrix(){};
	//copy constructor
	term2v_matrix(const term2v_matrix & X, aallocator * al);
	//construct empty matrix of required size
	term2v_matrix(const mint2 & sz, aallocator * al);
	// construct from a num_array
	template<typename type2> explicit term2v_matrix(const num_array<type2, 2> & m);

	virtual tthis * copy(aallocator * al)const override{
		return al->allocate<tthis>(*this);
	};
public: // implementing interfaces
	virtual type operator()(int i, int j) const override{
		return parent::operator()(i, j);
	};
	type & operator()(int i, int j){
		return parent::operator()(i, j);
	};
public:
	virtual void min_sum(tvect& src, tvect & dest)const override;
	virtual void min_sum_t(tvect& src, tvect& dest)const override;
	virtual void min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const override;
	virtual void min_sum_et(const int vK1, vectorizer * source, vectorizer * message)const override;
	virtual void get_row(int i, tvect & r) const override{
		for (int j = 0; j < r.count(); ++j){
			r[j] = parent::operator()(i,j);
		};
	};
	virtual void get_col(int j, tvect & r) const override{
		assert(r.count() <= count(0));
		for (int i = 0; i < r.count(); ++i){
			r[i] = parent::operator()(i, j);
		};
	};
	virtual void get_col_e(const int vK1, int j0, vectorizer * message)const override{
		const vectorizer * pdata = (const vectorizer *)begin(0, j0);
		for (int i = 0; i < vK1; ++i){
			message[i] = pdata[i];
		};
	};
	virtual int count(int i) const override{
		return parent::size()[i];
	};
};

//_______________________term2v_diff______________________________

template<typename type, typename vectorizer>
class term2v_diff : public term2v<type, vectorizer>{
private:
	typedef typename term2v<type, vectorizer>::tvect tvect;
	typedef term2v_diff<type, vectorizer> tthis;
	tvect diags; // assume matrix has constant diagonals
	tvect rdiags; // the same in the reverse order
	mint2 size; // aligned size
public:
	//virtual void * data_ptr()override{ return diags.begin(); };
public: // implementing interfaces
	virtual type operator()(int i, int j) const override{
		return diags(i - j + size[1] - 1);
	};
	type & operator()(int i, int j){
		return diags(i - j + size[1] - 1);
	};
public:
	term2v_diff(const term2v_diff & X, aallocator * al){
		size = X.size;
		type * v1 = al->allocate_a<type>(X.diags.size());
		type * v2 = al->allocate_a<type>(X.rdiags.size());
		if (v1){
			assert(diags.empty());
			diags.set_ref(v1, X.diags.size());
			diags << X.diags;
		};
		if (v2){
			assert(rdiags.empty());
			rdiags.set_ref(v2, X.rdiags.size());
			rdiags << X.rdiags;
		};
	};
	virtual tthis * copy(aallocator * al)const override{
		return al->allocate<tthis>(*this);
	};
public:
	template<typename type2> explicit term2v_diff(const num_array<type2, 2> & m);
	virtual void min_sum(tvect& src, tvect & dest)const override;
	virtual void min_sum_t(tvect& src, tvect& dest)const override;
	virtual void min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const override;
	virtual void min_sum_et(const int vK1, vectorizer * source, vectorizer * message)const override;
	virtual void get_row(int i, tvect & r) const override{
		for (int j = 0; j < size[1]; ++j){
			r[j] = term2v_diff::operator()(i, j);
		};
	};
	virtual void get_col(int j, tvect & r) const override{
		for (int i = 0; i < size[0]; ++i){
			r[i] = term2v_diff::operator()(i, j);
		};
	};
	virtual void get_col_e(const int vK1, int j0, vectorizer * message)const override;
	virtual int count(int i) const override{
		return size[i];
	};
};

//_______________________term2v_matrix_po_____________________________
template<typename type, typename vectorizer>
class term2v_matrix_po : public term2v_matrix<type, vectorizer>{
private:
	typedef term2v_matrix<type, vectorizer> parent;
public:
	const term2v<type,vectorizer> * source;
public:
	term2v_matrix_po(){};
	~term2v_matrix_po(){
	};
	// construct from any term2v
	term2v_matrix_po(const term2v<type, vectorizer> & m, aallocator * al) : parent(mint2(m.count(0),m.count(1)), al){
		source = &m;
	};
	/*
	term2v_matrix_po(term2v<type,vectorizer> * m, int K1, int K2){
		init(m,K1,K2);
	};
	void init(term2v<type, vectorizer> * m, int K1, int K2);
	*/
	virtual void reduce(po_mask & U_s, int y_s, po_mask & U_t, int y_t) override;
	virtual void rebuild(po_mask & U_s, int y_s, po_mask & U_t, int y_t) override;
};


//_______________________term2v_potts_______________________________
template<typename type, typename vectorizer = type>
class term2v_potts : public term2v<type,vectorizer>{
	typedef typename term2v<type, vectorizer>::tvect tvect;
	typedef term2v_potts<type,vectorizer> tthis;
public:
	type gamma;
	mint2 size; // aligned size
public:
	template<typename type2> term2v_potts(const dynamic::num_array<type2, 2> & _f2);
	term2v_potts(mint2 sz, type gamma){
		if (gamma < 0)throw debug_exception("Will break down with negative gamma");
		this->gamma = gamma;
		this->size = sz;
	};
	term2v_potts(){};
	term2v_potts(const tthis & a, aallocator * al = 0);
	virtual tthis * copy(aallocator * al)const override;
	//
	virtual void min_sum(tvect & a, tvect & r)const override;
	virtual void min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const;
	virtual void min_sum_et(const int vK1, vectorizer * source, vectorizer * message)const{
		term2v_potts::min_sum_e(vK1,source, message);
	};
	virtual type operator()(int i, int j)const override{
		if (i == j){
			//return type(-gamma);
			return 0;
		} else{
			//return 0;
			return gamma;
		};
	};
	virtual void get_row(int i, tvect & r) const override{
		//r << 0;
		//r[i] = -gamma;
		r << gamma;
		r[i] = 0;
	};
	virtual void get_col(int i, tvect & r) const override{
		r << gamma;
		r[i] = 0;
		//r << 0;
		//r[i] = -gamma;
	};
	virtual void get_col_e(const int vK1, int j0, vectorizer * message)const override{
		for (int i = 0; i < vK1; ++i){
			message[i] = gamma;
		};
		((type*)message)[j0] = 0;
	};
	virtual int count(int i) const override{
		return size[i];
	};
};

//_______________________term2v_tlinear_______________________________
template<typename type, typename vectorizer>
class term2v_tlinear : public term2v<type, vectorizer>{
	// pairwise function min(gamma*|i-j|,th);
	typedef typename term2v<type, vectorizer>::tvect tvect;
	typedef term2v_tlinear<type, vectorizer> tthis;
public:
	type gamma;
	type th;
	mint2 size; // aligned size
public:
	term2v_tlinear(mint2 sz, type gamma, type th){
		if (gamma < 0)throw debug_exception("Will break down with negative gamma - non-convex");
		this->gamma = gamma;
		this->th = th;
		this->size = sz;
	};
	//! copy constructor
	term2v_tlinear(const term2v_tlinear & X, aallocator * al = 0){
		this->gamma = X.gamma;
		this->th = X.th;
		this->size = X.size;
	};
	virtual tthis * copy(aallocator * al)const override{
		return al->allocate<tthis>(*this);
	};
	template<typename type2> term2v_tlinear(const dynamic::num_array<type2, 2> & _f2);
	virtual void min_sum(tvect & a, tvect & r)const override;
	virtual void min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const;
	__forceinline virtual void min_sum_et(const int vK1, vectorizer * source, vectorizer * message)const{
		term2v_tlinear::min_sum_e(vK1, source, message);
	};
	virtual type operator()(int i, int j)const override{
		return std::min(gamma*abs(i-j),th);
	};
	virtual void get_row(int i, tvect & r) const override{
		for (int j = 0; j < size[1]; ++j){
			r[j] = term2v_tlinear::operator()(i, j);
		};
	};
	virtual void get_col(int j, tvect & r) const override{
		for (int i = 0; i < size[0]; ++i){
			r[i] = term2v_tlinear::operator()(i, j);
		};
	};
	virtual void get_col_e(const int vK1, int j0, vectorizer * message)const override{
		int i = 0;
		type * m = (type*)message;
		type v = gamma*j0;
		for (; i <= j0; ++i){
			m[i] = std::min(v, th);
			v -= gamma;
		};
		v  = gamma;
		for (; i < size[0]; ++i){
			m[i] = std::min(v, th);
			v += gamma;
		};
	};
	virtual int count(int i) const override{
		return size[i];
	};
};

//_______________________term2v_tquadratic_______________________________
template<typename type, typename vectorizer>
class term2v_tquadratic : public term2v<type, vectorizer>{
	// pairwise function min(gamma*|i-j|,th);
	typedef typename term2v<type, vectorizer>::tvect tvect;
	typedef term2v_tquadratic<type, vectorizer> tthis;
	struct q_roof{
		int i;  // when entered
		type a; // value a[i]
		int x; // when becomes active
	};
public:
	type gamma;
	type th;
	mint2 size; // aligned size
public:
	//! Try construct from a matrix and throw if not truncated quadratic
	template<typename type2> term2v_tquadratic(const num_array<type2,2> & matrix, mint2 sza);
	term2v_tquadratic(mint2 sz, type gamma, type th){
		if (gamma < 0)throw debug_exception("Will break down with negative gamma - non-convex");
		this->gamma = gamma;
		this->th = th;
		this->size = sz;
	};
	term2v_tquadratic(const tthis & X, aallocator * al){
		this->gamma = X.gamma;
		this->th = X.th;
		this->size = X.size;
	};
	virtual tthis * copy(aallocator * al)const override{
		return al->allocate<tthis>(*this);
	};
	virtual void min_sum(tvect & a, tvect & r)const override;
	virtual void min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const;
	__forceinline virtual void min_sum_et(const int vK1, vectorizer * source, vectorizer * message)const{
		term2v_tquadratic::min_sum_e(vK1, source, message);
	};
	virtual type operator()(int i, int j)const override{
		return std::min(gamma*math::sqr(i - j), th);
	};
	virtual void get_row(int i, tvect & r) const override{
		for (int j = 0; j < size[1]; ++j){
			r[j] = term2v_tquadratic::operator()(i, j);
		};
	};
	virtual void get_col(int j, tvect & r) const override{
		for (int i = 0; i < size[0]; ++i){
			r[i] = term2v_tquadratic::operator()(i, j);
		};
	};
	virtual void get_col_e(const int vK1, int j0, vectorizer * message)const override{
		type *m = (type*)message;
		for (int i = 0; i < size[0]; ++i){
			m[i] = term2v_tquadratic::operator()(i, j0);
		};
	};
	virtual int count(int i) const override{
		return size[i];
	};
};


/*
//_______________________term2_potts_po____________________________
template<typename type, typename vectorizer>
class term2_potts_po : public term2_potts<type,vectorizer>{
	typedef term2_potts<type, vectorizer> parent;
public:
	type beta_st;
	type beta_ts;
	type c;
	po_mask * U_s;
	po_mask * U_t;
	int y_s;
	int y_t;
public: //_DEBUG
#ifdef _DEBUG
	term2_matrix_po<type> ref;
#endif
public:
#ifdef _DEBUG
	term2_potts_po(term2_potts * src, int K1, int K2):term2_potts(mint2(K1,K2),src->gamma),ref(src,K1,K2){
	};
#else
	term2_potts_po(term2_potts * src, int K1, int K2) :term2_potts(mint2(K1, K2), src->gamma){
	};
#endif
	virtual void reduce(po_mask & U_s, int y_s, po_mask & U_t, int y_t) override{
		// nothing to do
	};
	virtual void rebuild(po_mask & U_s, int y_s, po_mask & U_t, int y_t) override;
	type delta_st(int i)const;
	type delta_ts(int j)const;
	virtual type operator()(int i, int j)const;

	void min_sum(tvect & a, tvect & r, po_mask * U_s, po_mask * U_t, int y_s, int y_t, type beta_st, type beta_ts);
	virtual void min_sum(tvect & a, tvect & r);
	virtual void min_sum_t(tvect & a, tvect & r) override;
};
*/

//____________________________________________________________
/*
template<class base> class term2v_po_reduced;
template<class base>
class term2v_po_reduced_constructor : public term2v < typename base::type, typename base::vectorizer > {
public:
	typedef typename base::type type;
	typedef typename base::vectorizer vectorizer;
	typedef typename term2v<type, vectorizer>::tvect tvect;
	typedef term2v_po_reduced<base> tobject;
	base & src;
public:
	term2v_po_reduced_constructor(base & Src) :base(Src){};
	virtual tobject * copy(aallocator * al)const override;
};
*/

//_______________________term2v_po_rduced_____________________
template<class base>
class term2v_po_reduced : public term2v<typename base::type, typename base::vectorizer>{
public:
	typedef base parent;
	typedef typename base::type type;
	typedef typename base::vectorizer vectorizer;
	typedef typename term2v<type, vectorizer>::tvect tvect;
	typedef term2v_po_reduced<base> tthis;
	base _src;
	base & src(){return _src;};
	const base & src()const{ return _src; };
private:
	type c;
	vectorizer * _delta_st;
	vectorizer * _delta_ts;
	//dynamic::fixed_array1<vectorizer, array_allocator<vectorizer, 16> > block;
	int vK1;
	int vK2;
	//int vK1()const{ return base::count(0); };
	//int vK2()const{ return base::count(1); };
public:
	//virtual void * data_ptr()override{return _delta_st; };
	//virtual void * data_ptr()override{ return &src; };
public: //additional data
	tvect delta_st;
	tvect delta_ts;
	po_mask * U_s;
	po_mask * U_t;
#ifdef _DEBUG
	int y_s;
	int y_t;
	term2v_matrix_po<type,vectorizer> ref;
#endif
public:
	// copy constructor from base term2
	term2v_po_reduced(const base & Src, aallocator * al);
	// copy constructor from self
	term2v_po_reduced(const tthis & X, aallocator * al);
	void init(aallocator * al);
	virtual tthis * copy(aallocator * al)const override{
		return al->allocate<tthis>(*this);
	};
public:
	virtual void reduce(po_mask & U_s, int y_s, po_mask & U_t, int y_t) override;
	virtual void rebuild(po_mask & U_s, int y_s, po_mask & U_t, int y_t) override;
public:
	virtual type operator()(int i, int j)const;
	type operator()(int i, int j, bool i_inU, bool j_inU)const;
	virtual void min_sum(tvect & a, tvect & r)const override;
	virtual void min_sum_t(tvect & a, tvect & r)const override;
public:
	template <bool dir_fw>
	void t_min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const;
	virtual void min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const override;
	virtual void min_sum_et(const int vK1, vectorizer * source, vectorizer * message)const override;
private:
	void min_sum(tvect & a, tvect & r, po_mask * U_s, po_mask * U_t, const tvect & delta_st, const tvect & detla_ts)const;
	void check(tvect & a, tvect & r, tvect & r1, po_mask * U_s, po_mask * U_t, const tvect & delta_st, const tvect & detla_ts, int y_s, int y_t)const;
	virtual void get_row(int i, tvect & r)const override;
	virtual void get_col(int j, tvect & r)const override;
	virtual void get_col_e(const int vK1, int j0, vectorizer * message)const override;
	virtual int count(int i)const override{ return src().base::count(i); };
};

//_______________________energy______________________________
template <class type>
class energy{
public:
	typedef typename term2<type>::tvect t_f1;
	//typedef term2<type> t_f2;
	typedef term2v<type, d_vectorizer> t_f2;
public:
	datastruct::mgraph G;
	intf K;
	int maxK;
	dynamic::fixed_array1<t_f1> f1;
	double tolerance;
public:
	virtual t_f2 & f2(int e) = 0;
	double cost(const intf & x);
	virtual void set_nE(int nE){
	};
	virtual void set_nV(int nV){
		G._nV = nV;
		f1.resize(nV);
		K.resize(nV);
	};
	void set_E(dynamic::num_array<int, 2> E){
		int nE = E.size()[1];
		G.E.resize(nE);
		for (int e = 0; e < nE; ++ e){
			G.E[e][0] = E(0, e);
			G.E[e][1] = E(1, e);
		};
		set_nE(nE);
		G.edge_index();
	};
	int nV()const{
		return G.nV();
	};
	int nE()const{
		return G.nE();
	};
public:
	virtual ~energy(){};
public:
};


//____________________energy_auto_________________________________________

template<typename type>
class energy_auto : public energy<type>{
public:
	typedef typename energy<type>::t_f1 t_f1;
	typedef typename energy<type>::t_f2 t_f2;
public:
	dynamic::fixed_array1<t_f2*> F2;
public:
	int nfull;
	int npotts;
	int ntlinear;
	int ndiff;
	int ntquadratic;
	double maxf;
	double delta;
	double mult;
public:
	energy_auto();
	virtual void set_nE(int nE){
		this->G.E.resize(nE);
		this->F2.resize(nE);
	};
	virtual t_f2 & f2(int e){
		return *(this->F2[e]);
	};
	void cleanup();
	virtual ~energy_auto(){
		cleanup();
	};
	//bool test_potts(const dynamic::num_array<double, 2> & _f2);
	//bool test_tlinear(const dynamic::num_array<double, 2> & _f2);
	void set_f2(int e, const dynamic::num_array<double, 2> & _f2);
	void set_f1(int v, const dynamic::num_array<double, 1> & _f1);
public:
	void init();
	void report();
};


#endif