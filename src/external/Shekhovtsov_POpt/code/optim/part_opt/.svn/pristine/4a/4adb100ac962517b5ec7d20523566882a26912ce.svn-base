#include "energy.h"
//#include "unroller.h"

#include <deque>
#include <cmath>

using namespace exttype;
using namespace dynamic;
using namespace custom_new;


/*
int size_align(int size){
return ((size + 3) / 4) * 4; // aligned to multiple of 4
};

void check_align(int size){
if (size % 4 != 0){
throw debug_exception("bad");
};
};

mint2 size_align(const mint2 & size){
return ((size + 3) / 4) * 4; // aligned to multiple of 4
};
*/

template<typename type, typename vectorizer>
int v_align(int size){
	int V = sizeof(vectorizer) / sizeof(type);
	return ((size + V - 1) / V) * V; // aligned to multiple of V
};

/*
//______________________term2_vcore__________________________________
template<typename type, typename vectorizer>
void term2_vcore<type,vectorizer>::min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const{
	int V = sizeof(vectorizer) / sizeof(type);
	int K1 = vK1*V;
	tvect a; a.set_ref(source, K1);
	tvect r; r.set_ref(message, K1);
	src->min_sum(a, r);
};

template<typename type, typename vectorizer>
void term2_vcore<type, vectorizer>::min_sum_et(const int vK1, vectorizer * source, vectorizer * message)const{
	int V = sizeof(vectorizer) / sizeof(type);
	int K1 = vK1*V;
	tvect a; a.set_ref(source, K1);
	tvect r; r.set_ref(message, K1);
	this->min_sum_t(a, r);
};
*/


//_______________________term2v_matrix________________________________
template<typename type, typename vectorizer>
term2v_matrix<type, vectorizer>::term2v_matrix(const term2v_matrix & X, aallocator * al){
	// put matrix on allocator
	type * data = al->allocate_a<type>(X.length());
	assert(this->is_empty());
	this->set_ref(data,X.size());
	// copy data
	if (data){
		(*this) << X;
	};
};

template<typename type, typename vectorizer>
term2v_matrix<type, vectorizer>::term2v_matrix(const mint2 & sz, aallocator * al){
	int K1a = v_align<type, vectorizer>(sz[0]);
	int K2a = v_align<type, vectorizer>(sz[1]);
	type * data = al->allocate_a<type>(K1a*K2a);
	assert(this->is_empty());
	this->set_ref(data, mint2(K1a, K2a));
	if (data){
		(*this) << 0;
	};
};

template<typename type, typename vectorizer>
template<typename type2> 
term2v_matrix<type, vectorizer>::term2v_matrix(const num_array<type2, 2> & m){
	const int V = sizeof(vectorizer) / sizeof(type);
	mint2 sz = mint2(v_align<type, vectorizer>(m.size()[0]), v_align<type, vectorizer>(m.size()[1]));
	this->resize(sz);
	int K1 = m.size()[0];
	int K2 = m.size()[1];
	(*this) << 0;
	//copy from num_array
	for (int i = 0; i < K1; ++i){
		for (int j = 0; j < K2; ++j){
			type v1 = (type)m(i, j);
			(*this)(i, j) = v1;
		};
	};
};


template<typename type, typename vectorizer>
void term2v_matrix<type, vectorizer>::min_sum(tvect& Src, tvect & Dest)const{
	const type * pdata = this->begin();
	int stride = this->stride(0, 1);
	for (int j = 0; j < Dest.size(); ++j){
		Dest[j] = Src[0] + pdata[j*stride];
	};
	for (int i = 1; i < Src.size(); ++i){
		type const * pdata = this->begin(i, 0);
		//++pdata;
		type s = Src[i];
		for (int j = 0; j < Dest.size(); ++j){
			Dest[j] = std::min(Dest[j], s + pdata[j*stride]);
		};
	};
	return;
};

template<typename type, typename vectorizer>
void term2v_matrix<type, vectorizer>::min_sum_t(tvect& Src, tvect & Dest)const{
	const type * pdata = this->begin();
	for (int i = 0; i < Dest.size(); ++i){
		Dest[i] = Src[0] + pdata[i];
	};
	for (int j = 1; j < Src.size(); ++j){
		const type * pdata = this->begin(0, j);
		type s = Src[j];
		for (int i = 0; i < Dest.size(); ++i){
			Dest[i] = std::min(Dest[i], s + pdata[i]);
		};
		//Unroller<4>::UnrollerP::step(Dest.size(), [&](int i){
		//	Dest[i] = std::min(Dest[i], s + pdata[i]);
		//});
	};
};

template<typename type, typename vectorizer>
void term2v_matrix<type, vectorizer>::min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const{
	//return term2v<type, vectorizer>::min_sum_e(vK1, source, message);
	//
	const int K2 = this->size()[1];
	const vectorizer * pdata = (const vectorizer*)this->begin();
	const vectorizer * source_end = source + vK1;
	type * pm = (type*)message;
	for (int j = 0; j < K2; ++j){// along message entries
		vectorizer * ps = source;
		vectorizer m = *pdata + *ps;
		++pdata;
		++ps;
		//for (int vi = vK1 - 2; vi >= 0; --vi){
		for (;ps<source_end;){
			v_min(m, *pdata + *ps);
			++pdata;
			++ps;
		};
		v_min(m);
		*pm = v_first(m);
		++pm;
	};
	return;

	/*
	int V = sizeof(vectorizer) / sizeof(type);
	int vK2 = size()[1] / V;
	for (int vj = 0; vj < vK2; ++vj){
		for (int j = 0; j < V; ++j){
			const vectorizer * data = (const vectorizer*)this->begin(0, vj*V + j);
			//vectorizer v;
			vectorizer m = source[0] + data[0];
			for (int vi = 1; vi < vK1; ++vi){
				vectorizer a = source[vi] + data[vi];
				v_min(m, a);
			};
			v_min(m);
			((type*)message)[vj*V + j] = v_first(m); // *(type*)(&m);
		};
	};
	*/
};

template<typename type, typename vectorizer>
void term2v_matrix<type, vectorizer>::min_sum_et(const int vK1, vectorizer * source, vectorizer * message)const{
	//return term2v<type, vectorizer>::min_sum_et(vK1, source, message);
	//
	
	const int K1 = this->size()[1];
	const int V = sizeof(vectorizer) / sizeof(type);
	const int vK2 = this->size()[0] / V;

	const vectorizer * pdata = (const vectorizer*)this->begin();
	const type * ps = (type*)(source);
	
	const vectorizer a(*ps);
	vectorizer * pm = message;
	for (int vj = 0; vj < vK2; ++vj){// along message entries
		*pm = *pdata + a;
		++pm;
		++pdata;
	};
	++ps;
	for (int vi = 1; vi < K1; ++vi){//along source entries, starting from 2nd row
		const vectorizer a(*ps);
		vectorizer * pm = message;
		for (int vj = vK2-1; vj >= 0 ; --vj){//along message entries
			v_min(*pm, *pdata + a);
			++pdata;
			++pm;
		};
		++ps;
	};
	return;
	
	/*
	// 2535604.0328
	// 2535604.0328
	const int V = sizeof(vectorizer) / sizeof(type);
	const int vK2 = size()[0] / V;
	for (int vj = 0; vj < vK2; ++vj){
		message[vj] = INF(type); // copy INF to all entries of message[vj]
	};
	int stride1 = this->stride(0, 1) / V;
	for (int vi = 0; vi < vK1; ++vi){
		type * a = (type*)&(source[vi]);
		const vectorizer * data0 = (const vectorizer*)this->begin(0, vi*V);
		for (int vj = 0; vj < vK2; ++vj){
			const vectorizer * data = data0;
			vectorizer m = message[vj];
			for (int i = 0; i < V; ++i){ // this loop can be unfolded by compiler
				vectorizer A(a[i]); // copy a[i] to all entries of A
				vectorizer v = A + *data;
				v_min(m, v);
				data += stride1;
			};
			message[vj] = m;
			++data0;
		};
	};
	*/
};


//_______________________term2v_diff________________________________
template<typename type, typename vectorizer>
template<typename type2> term2v_diff<type, vectorizer>::term2v_diff(const num_array<type2, 2> & m){
	const int V = sizeof(vectorizer) / sizeof(type);
	size = mint2(v_align<type, vectorizer>(m.size()[0]), v_align<type, vectorizer>(m.size()[1]));
	//if (m.size()[0] != m.size[1])throw quiet_exception("not square");
	int K1 = m.size()[0];
	int K2 = m.size()[1];
	//if (K1 != K2)throw quiet_exception("diff: not square"); // only accept square matrices here
	if (K1 < 2 || K2 < 2) throw quiet_exception("diff: too small"); // too small to bother
	int aK1 = this->size[0];
	int aK2 = this->size[1];
	diags.resize(aK1 + aK2 - 1);
	rdiags.resize(aK1 + aK2 - 1);
	diags << 0; // INF(type);
	rdiags << 0; // INF(type);
	int j = 0;
	for (int i = 0; i < K1; ++i){
		diags[i + aK2 - 1 - j] = (type)m(i, j);
		rdiags[j + aK1 - 1 - i] = (type)m(i, j);
	};
	int i = 0;
	for (int j = 0; j < K2; ++j){
		diags[i + aK2 - 1 - j] = (type)m(i, j);
		rdiags[j + aK1 - 1 - i] = (type)m(i, j);
	};
	//verify
	for (int i = 0; i < K1; ++i){
		for (int j = 0; j < K2; ++j){
			type v1 = (type)m(i, j);
			type v2 = (*this)(i, j);
			if (std::abs(v1-v2) > 1e-4){
				throw quiet_exception("diff: does not fit");
			};
		};
	};
};

template<typename type, typename vectorizer>
void term2v_diff<type, vectorizer>::min_sum(tvect& Src, tvect & Dest)const{
	for (int j = 0; j < this->size[1]; ++j){
		type m = INF(type);
		for (int i = 0; i < this->size[0]; ++i){
			m = std::min(m, Src[i] + term2v_diff::operator()(i, j));
		};
		Dest[j] = m;
	};
};

template<typename type, typename vectorizer>
void term2v_diff<type, vectorizer>::min_sum_t(tvect& Src, tvect & Dest)const{
	for (int j = 0; j < this->size[0]; ++j){
		type m = INF(type);
		for (int i = 0; i < this->size[1]; ++i){
			m = std::min(m, Src[i] + term2v_diff::operator()(j, i));
		};
		Dest[j] = m;
	};
};

template<typename type, typename vectorizer>
void term2v_diff<type, vectorizer>::min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const{
	//return term2v<type, vectorizer>::min_sum_e(vK1, source, message);
	//
	//int K1 = size()[0];
	//const int V = sizeof(vectorizer) / sizeof(type);
	//struct unaligned{
	//	type a[V];
	//};
	//typedef type unaligned[V];
	const int K2 = this->size[1];
	const vectorizer * source_end = source + vK1;
	type * pm = (type*)message;
	for (int j = 0; j < K2; ++j){// along message entries, unaligned K2
		const unaligned<type, vectorizer> * pdata = (const unaligned<type, vectorizer> *)&diags[K2 - 1 - j];
		vectorizer * ps = source;
		vectorizer m  = *pdata;
		m += *ps;
		++pdata;
		++ps;
		for (; ps<source_end;){
			vectorizer a = *pdata;
			v_min(m, a + *ps);
			++pdata;
			++ps;
		};
		v_min(m);
		*pm = v_first(m);
		assert(*pm > -INF(type));
		++pm;
	};
};

template<typename type, typename vectorizer>
void term2v_diff<type, vectorizer>::min_sum_et(const int vK1, vectorizer * source, vectorizer * message)const{
	//return term2v<type, vectorizer>::min_sum_et(vK1, source, message);
	const int V = sizeof(vectorizer) / sizeof(type);
	//typedef type unaligned[V];
	//struct unaligned{
//		type a[V];
	//};
	const int K2 = this->size[0];
	const vectorizer * source_end = source + vK1;
	type * pm = (type*)message;
	for (int j = 0; j < K2; ++j){// along message entries, unaligned K2
		const unaligned<type, vectorizer> * pdata = (const unaligned<type, vectorizer>*)&rdiags[K2 - 1 - j];
		vectorizer * ps = source;
		vectorizer m = *pdata;
		m += *ps;
		++pdata;
		++ps;
		for (; ps<source_end;){
			vectorizer a = *pdata;
			v_min(m, a + *ps);
			++pdata;
			++ps;
		};
		v_min(m);
		*pm = v_first(m);
		++pm;
	};
};

template<typename type, typename vectorizer>
void term2v_diff<type, vectorizer>::get_col_e(const int vK1, int j0, vectorizer * message)const{
	typedef unaligned<type, vectorizer> tu;
	const tu * up = (const tu*)&diags(0 - j0 + size[1] - 1);
	//tu * um = (tu*)message;
	//debug::stream << size[0] << "x" << size[1] << "\n";
	for (int i = 0; i < vK1; ++i){
		message[i] = up[i];
	};
};


//_______________________term2v_matrix_po______________________________

/*
template<typename type, typename vectorizer>
void term2v_matrix_po<type, vectorizer>::init(term2v<type,vectorizer> * m, int K1, int K2){
	int K1a = v_align<type, vectorizer>(K1);
	int K2a = v_align<type, vectorizer>(K2);
	source = m;
	this->resize(mint2(K1a, K2a));
	(*this) << 0;
};
*/

template<typename type, typename vectorizer>
void term2v_matrix_po<type, vectorizer>::reduce(po_mask & U_s, int y_s, po_mask & U_t, int y_t){
	int K1 = parent::size()[0];
	int K2 = parent::size()[1];
	// reduce energy
	assert(U_s[y_s]);
	assert(U_t[y_t]);
	for (int k1 = 0; k1 < K1; ++k1){
		type U_min1 = std::numeric_limits<type>::max();
		if (!U_s[k1]){
			for (int k2 = 0; k2 < K2; ++k2){
				if (U_t[k2]){
					U_min1 = std::min(U_min1, (*this)(k1, k2));
				};
			};
			for (int k2 = 0; k2 < K2; ++k2){
				if (U_t[k2])(*this)(k1, k2) = U_min1;
			};
		};
	};
	for (int k2 = 0; k2 < K2; ++k2){
		type U_min2 = std::numeric_limits<type>::max();
		if (!U_t[k2]){
			for (int k1 = 0; k1 < K1; ++k1){
				if (U_s[k1]){
					U_min2 = std::min(U_min2, (*this)(k1, k2));
				};
			};
			for (int k1 = 0; k1 < K1; ++k1){
				if (U_s[k1]){
					(*this)(k1, k2) = U_min2;
				};
			};
		};
	};
	for (int k1 = 0; k1 < K1; ++k1){
		if (U_s[k1])continue;
		for (int k2 = 0; k2 < K2; ++k2){
			if (U_t[k2])continue;
			type d = (*this)(k1, k2);
			type b = (*this)(y_s, k2);
			type c = (*this)(k1, y_t);
			type a = (*this)(y_s, y_t);
			assert(std::abs(a) < 1e-4);
			//type delta = b + c - d - a;
			//if (delta < 0){
			//	(*this)(k1, k2) = b + c - a;
			//};
			(*this)(k1, k2) = std::min(b + c, d);
			assert((*this)(k1, k2) > -1e10);
		};
	};
};

template<typename type, typename vectorizer>
void term2v_matrix_po<type, vectorizer>::rebuild(po_mask & U_s, int y_s, po_mask & U_t, int y_t){
	int K1 = parent::size()[0];
	int K2 = parent::size()[1];
	//(*this) << INF(type);
	//int K1 = source->count(0);
	//int K1 = source->count(1);
	for (int k1 = 0; k1 < K1; ++k1){
		for (int k2 = 0; k2 < K2; ++k2){
			if (U_s[k1]){
				if (U_t[k2]){ // [1,1] - both immovable
					(*this)(k1, k2) = 0;
				} else{ // [1 0]
					(*this)(k1, k2) = (*source)(k1, k2) - (*source)(k1, y_t);
				};
			} else{
				if (U_t[k2]){ // [0,1]
					(*this)(k1, k2) = (*source)(k1, k2) - (*source)(y_s, k2);
				} else{ // [0 0]
					(*this)(k1, k2) = (*source)(k1, k2) - (*source)(y_s, y_t);
				};
			};
			assert((*this)(k1, k2) > -1e10);
			assert((*this)(k1, k2) == INF(type) || (*this)(k1, k2) < 1e20);
		};
	};
};

//___________________________term2v_potts__________________________________________

template<typename type, typename vectorizer>
term2v_potts<type, vectorizer>::term2v_potts(const term2v_potts & a, aallocator * al){
	this->gamma = a.gamma;
	this->size = a.size;
	// nothing to allocate additionally
};

template<typename type, typename vectorizer>
term2v_potts<type, vectorizer> * term2v_potts<type, vectorizer>::copy(aallocator * al)const{
	return al->allocate<tthis>(*this); // copy-initialized from *this
};

template<typename type, typename vectorizer>
template<typename type2>
term2v_potts<type, vectorizer>::term2v_potts(const dynamic::num_array<type2, 2> & _f2){
	int K1 = _f2.size()[0];
	int K2 = _f2.size()[1];
	this->size = mint2(v_align<type, vectorizer>(_f2.size()[0]), v_align<type, vectorizer>(_f2.size()[1]));
	if (K1 != K2)throw quiet_exception("potts: not square"); // only accept square matrices here
	if (K1 < 2 || K2 < 2) throw quiet_exception("potts: too small"); // too small to bother
	this->gamma = (type)_f2(1, 0); // try this or it is not Potts
	// verift whether it fits
	for (int i = 0; i < K1; ++i){
		for (int j = 0; j < K2; ++j){
			if (std::abs(_f2(i, j) - type2((*this)(i, j)))>1e-5)throw quiet_exception("does not fit");
		};
	};
	return;
	//term2v_potts<type,vectorizer> a1(dynamic::num_array<double, 2>());
	//term2v_potts<type,vectorizer> a2(dynamic::num_array<float, 2>());
	// passed
};

template<typename type, typename vectorizer>
void term2v_potts<type, vectorizer>::min_sum(tvect & a, tvect & r)const{
	//assert(a.count() == r.count());
	type m = a.min().first + gamma;
	for (int i = 0; i < r.count(); ++i){
		r[i] = std::min(a[i], m);
	};
};

template<typename type, typename vectorizer>
__forceinline void term2v_potts<type, vectorizer>::min_sum_e(const int vK1, vectorizer * source, vectorizer * const message)const{
	//_mm_prefetch((char *)message, _MM_HINT_T0);
	vectorizer m(INF(type));
	const vectorizer * ps = source;
	for (int i = 0; i < vK1; ++i){
		v_min(m, ps[i]);
	};
	v_min(m);
	m += gamma;
	for (int i = 0; i < vK1 ; ++i){
		vectorizer v = ps[i];
		v_min(v, m);
		message[i] = v;
	};

	/*
	for (int i = vK1 - 1; i >= 0; --i){
		vectorizer v = *ps;
		v_min(v, m);
		*pm = v;
		++ps;
		++pm;
	};
	*/

	/*
	// pass 1 find minimum
	vectorizer m = source[0];
	for (int i = 1; i < vK1; ++i){
		vectorizer v = source[i];
		v_min(m, v);
	};
	v_min(m);
	// pass 2 threshold
	vectorizer vgamma(gamma);
	for (int i = 0; i < vK1; ++i){
		vectorizer v = source[i];
		v -= vgamma;
		v_min(v, m);
		message[i] = v;
	};
	*/
};


//___________________________term2v_tlinear__________________________________________
template<typename type, typename vectorizer>
template<typename type2>
term2v_tlinear<type, vectorizer>::term2v_tlinear(const dynamic::num_array<type2, 2> & m){
	// energy in the form  min(|i-j|*gamma,th)
	int K1 = m.size()[0];
	int K2 = m.size()[1];
	this->size = mint2(v_align<type, vectorizer>(m.size()[0]), v_align<type, vectorizer>(m.size()[1]));
	if (K1 != K2)throw quiet_exception("tlinear: not square"); // only accept square matrices here
	if (K1 < 2 || K2 < 2) throw quiet_exception("tlinear: too small"); // too small to bother
	//
	this->th = (type)m(K1 - 1, 0); // this must be the threshold or the threshold has no effect
	this->gamma = (type)m(1, 0); // this must be the slope
	// verift whether it fits
	for (int i = 0; i < K1; ++i){
		for (int j = 0; j < K2; ++j){
			if (std::abs(m(i, j) - type2((*this)(i, j)))>1e-5)throw quiet_exception("tlinear: does not fit");
		};
	};
	// passed
};


template<typename type, typename vectorizer>
__forceinline void term2v_tlinear<type, vectorizer>::min_sum(tvect & a, tvect & r)const{
	// forward pass: find min and min-conv with max(0,(i-j)*gamma)
	int K = a.size();
	type roof = a[0]+gamma;
	r[0] = a[0];
	type m1 = a[0];
	for (int i = 1; i < K; ++i){
		if (a[i] < roof){
			roof = a[i];
		};
		if (a[i] < m1){
			m1 = a[i];
		};
		r[i] = roof;
		roof += gamma;
	};
	m1 += th;
	// backward pass: min-conv with max(0,-(i-j)*gamma) and min with m1
	roof = r[K-1] + gamma;
	for (int i = K - 1; i >= 0; --i){
		if (r[i] < roof){
			roof = r[i];
		} else{
			r[i] = roof;
		};
		if (m1 < r[i]){
			r[i] = m1;
		};
		roof += gamma;
	};
};
template<typename type, typename vectorizer>
__forceinline void term2v_tlinear<type, vectorizer>::min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const{
	const int V = sizeof(vectorizer) / sizeof(type);
	/*
	{
	int K1 = vK1*V;
	tvect a; a.set_ref(source, K1);
	tvect r; r.set_ref(message, K1);
	term2v_tlinear::min_sum(a, r);
	return;
	};
	*/
	// todo: can still expand last iteration
	// forward pass: find min and min-conv with max(0,(i-j)*gamma)
	vectorizer gamma_ramp;
	for (int i = 0; i < V; ++i){
		((type*)&gamma_ramp)[i] = (i + 1)*gamma;
	};
	vectorizer m1(INF(type));
	vectorizer roof(INF(type));
	vectorizer * ps = source;
	vectorizer * pm = message;
	for (int i = vK1 - 1; i >= 0; --i){
		vectorizer A = *ps;
		v_min(m1, A);
		A -= gamma_ramp;
		v_cum_min(A);
		v_min(A, roof);
		A += gamma_ramp;
		*pm = A;
		roof = v_broadcast_last(A);
		++ps;
		++pm;
	};
	v_min(m1);
	m1 += th;
	// backward pass: min-conv with max(0,-(i-j)*gamma) and min with m1
	roof = INF(type);
	--pm;
	gamma_ramp = v_flip(gamma_ramp);
	for (int i = vK1 - 1; i >= 0; --i){
		vectorizer A = *pm;
		A -= gamma_ramp;
		v_rcum_min(A);
		v_min(A, roof);
		A += gamma_ramp;
		v_min(A, m1);
		*pm = A;
		roof = v_broadcast_first(A);
		--pm;
	};

	/*
	return;
	//
	const int K = size[0];
	{
		const type * a = (type*)source;
		type * r = (type*)message;
		type roof = a[0] + gamma;
		r[0] = a[0];
		for (int i = 1; i < K; ++i){
			if (a[i] < roof){
				roof = a[i];
			};
			//r[i] = roof;
			assert(roof == r[i] || std::abs(roof - r[i]) < 1e-3);
			roof += gamma;
		};
	}
	{
		// backward pass: min-conv with max(0,-(i-j)*gamma) and min with m1
		type m1s = v_first(m1);
		type * r = (type*)message;
		type roof = r[K - 1] + gamma;
		for (int i = K - 1; i >= 0; --i){
			if (r[i] < roof){
				roof = r[i];
			} else{
				r[i] = roof;
			};
			if (m1s < r[i]){
				r[i] = m1s;
			};
			roof += gamma;
		};
	};
	*/
};

//___________________________term2v_tquadratic__________________________________________
template<typename type, typename vectorizer>
template<typename type2>
term2v_tquadratic<type, vectorizer>::term2v_tquadratic(const num_array<type2, 2> & m, mint2 sza){
	if (m.size()[0] != m.size()[1]) throw quiet_exception("not square");
	int K = m.size()[0];
	// if m is truncated quadratic, read threshold from m[0,K-1]
	this->th = type(m(0, K - 1)); // threshold must be this or it has no effect
	// gamma from m[0,K-1]
	this->gamma = type(m(0, 1));
	this->size = sza;
	// verift whether it fits
	for (int i = 0; i < K; ++i){
		for (int j = 0; j < K; ++j){
			if (std::abs(m(i, j) - type2((*this)(i, j)))>1e-5)throw quiet_exception("does not fit");
		};
	};
};

template<typename type, typename vectorizer>
__forceinline void term2v_tquadratic<type, vectorizer>::min_sum(tvect & a, tvect & res)const{
	// forward pass: find min and min-conv with max(0,(i-j)*gamma)^2
	int K = a.size();
	// need a temp stack for roofs: [i - when entered, a = a[i], x - when becomes a roof]
	tvect temp(K);
	//reference
#ifdef _DEBUG
	tvect ref(K);
	for (int i = 0; i < K; ++i){
		type m = INF(type);
		for (int j = 0; j < K; ++j){
			m = std::min(m, a[j] + this->operator()(i,j));
		};
		ref[i] = m;
	};
#endif
	//debug::stream <<"a=  "<< a << "\n";
	//debug::stream <<"ref="<< ref << "\n";

	std::deque<q_roof> stack;
	//stack.reserve(K);
	stack.push_back(q_roof());
	q_roof * r = &stack.back();
	
	type * source;
	type * target;
	type m1 = a[0]; // this just computes min for thresholding

	for (int forward = 1; forward >= 0; --forward){
		if (forward){
			source = a.begin();
			target = temp.begin();	
		} else{
			source = temp.begin();
			target = res.begin();
		};
		r->i = 0;
		r->a = source[0];
		r->x = 0;

		type v = source[0];
		if (!forward){
			v = std::min(v, m1); // thresholded
		};
		target[K - 1] = v;
		//
		for (int i = 1; i < K; ++i){
			// calculate intersectios with the top roof until we get larger x
			type _a = source[i];
			if (forward && _a < m1){
				m1 = source[i];
			};
			int x;
			do{
				r = &stack.back();
				x = (int)ceil(((_a - r->a) / (gamma*(i - r->i)) + i + r->i) / 2); // where new quadric enters
				if (x <= r->x){// we are better, remove the top roof
					stack.pop_back();
					if (stack.empty()){
						x = i; // enters now
						break;
					};
				} else{
					break;
				};
			} while (true);
			if (x < K){// add the new roof on top, only if it is ever usefull
				stack.push_back(q_roof());
				r = &stack.back();
				r->i = i;
				r->a = _a;
				r->x = x;
			};
			// check if the next roof activates
			if (stack.size() > 1){
				int x1 = stack[1].x;
				if (i >= x1){ //activates
					stack.pop_front();
				};
			};
			// get the value of the active roof
			r = &stack.front();
			type v = math::sqr(i - r->i)*gamma + r->a;
			if (!forward){
				v = std::min(v, m1); // thresholded
			};
			target[K - 1 - i] = v; //write in reverse order for the second loop. a double flip in total
		};
		if (forward){
			m1 += th;
		};
		/*
		if (forward){
			debug::stream <<"res1=" << res << "\n";
		} else{
			debug::stream <<"res2="<< a << "\n";
		};
		*/
	};
#ifdef _DEBUG
	for (int i = 0; i < K; ++i){
		if (std::abs(res[i] - ref[i]) > 1e-4){
			debug::stream << res << "\n";
			debug::stream << ref << "\n";
			//debug::stream << res << "\n";
			int b = 0;
		};
	};
#endif
};

template<typename type, typename vectorizer>
__forceinline void term2v_tquadratic<type, vectorizer>::min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const{
	const int V = sizeof(vectorizer) / sizeof(type);
	{
	int K1 = vK1*V;
	tvect a; a.set_ref(source, K1);
	tvect r; r.set_ref(message, K1);
	term2v_tquadratic::min_sum(a, r);
	return;
	};
};


//_______________________term2_po_rduced_____________________
/*
template<class base>
term2v_po_reduced<base> * term2v_po_reduced_constructor<base>::copy(aallocator * al)const{
	return al->allocate<term2v_po_reduced<base> >(src); // copy-allocated from source term
};

//template<typename type, typename vectorizer>
template<class base>
term2v_po_reduced<base>::term2v_po_reduced(base & Src) :src(Src), vK1(0), vK2(0){
#ifdef _DEBUG
	ref.init(&src, src.count(0), src.count(1));
#endif
	int V = sizeof(vectorizer) / sizeof(type);
	int K1a = v_align<type, vectorizer>(src.count(0));
	int K2a = v_align<type, vectorizer>(src.count(1));
	vK1 = K1a / V;
	vK2 = K2a / V;
	block.resize(K1a + K2a);
	_delta_st = &block[0];
	_delta_ts = &block[K1a];
	for (int i = 0; i < block.size(); ++i){
		block[i] = 0;
	};
	delta_st.set_ref(_delta_st, src.count(0));
	delta_ts.set_ref(_delta_ts, src.count(1));
};
*/

template<class base>
#ifdef _DEBUG
term2v_po_reduced<base>::term2v_po_reduced(const base & Src, aallocator * al) :_src(Src, al), ref(Src, al){
#else
term2v_po_reduced<base>::term2v_po_reduced(const base & Src, aallocator * al) :_src(Src, al){
#endif
	// leave uninitialized
	init(al);
};
template<class base>
void term2v_po_reduced<base>::init(aallocator * al){
	U_s = 0;
	U_t = 0;
	int V = sizeof(vectorizer) / sizeof(type);
	int K1a = v_align<type, vectorizer>(count(0));
	int K2a = v_align<type, vectorizer>(count(1));
	vK1 = K1a / V;
	vK2 = K2a / V;
	//block.resize(K1a + K2a);
	//
	_delta_st = al->allocate_a<vectorizer>(vK1);
	_delta_ts = al->allocate_a<vectorizer>(vK2);
	//
	if (_delta_st){
		assert(delta_st.empty());
		delta_st.set_ref(_delta_st, count(0));
		delta_st << 0;
	};
	if (_delta_ts){
		assert(delta_ts.empty());
		delta_ts.set_ref(_delta_ts, count(1));
		delta_ts << 0;
	};
};

template<class base>
#ifdef _DEBUG
term2v_po_reduced<base>::term2v_po_reduced(const term2v_po_reduced<base> & X, aallocator * al) :_src(X._src, al),ref(X.src(),al){
#else
term2v_po_reduced<base>::term2v_po_reduced(const term2v_po_reduced<base> & X, aallocator * al) :_src(X._src, al){
#endif
// :parent(X.src(), al){ // go on the allocator with parent class
	//ref.init(&src(), src().count(0), src().count(1));
	init(al);
};

#define Delta_st (dir_fw? _delta_st: _delta_ts)
#define Delta_ts (dir_fw? _delta_ts: _delta_st)
#define dir_vK2 (dir_fw? vK2: vK1)

//template<typename type, typename vectorizer>
//template<typename type, typename vectorizer, template<typename, typename> class base >
template<class base>
template<bool dir_fw>
void term2v_po_reduced<base>::t_min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const{
	//int K2 = r.size();
	/*
	if (dir_fw){
	src.base::min_sum_e(vK1, source, message);
	} else{
	src.base::min_sum_et(vK1, source, message);
	};
	return;
	*/
	//_mm_prefetch((char *)&src, _MM_HINT_T0);
	//_mm_prefetch((char *)Delta_st, _MM_HINT_T0);
	_mm_prefetch((char *)_delta_st, _MM_HINT_T0);
	// message passing for original term
	if (dir_fw){
		src().base::min_sum_e(vK1, source, message);
	} else{
		src().base::min_sum_et(vK1, source, message);
	};
	// correcting min
	//_mm_prefetch((char *)Delta_ts, _MM_HINT_T0);
	vectorizer * ps = source;
	vectorizer * pd = Delta_st;
	vectorizer m1(INF(type));
	for (int i = vK1 - 1; i >= 0; --i){
		vectorizer v = *ps + *pd;
		v_min(m1, v);
		++ps;
		++pd;
	};
	v_min(m1); 
	//m1 = min(source[y],min_{i\in Y_s}(souerce[i]+delta_{st}[i]))
	// correction for reduced term
	vectorizer vc(c);
	vectorizer * pm = message;
	const vectorizer * pth = Delta_ts;
	for (int j = dir_vK2-1; j >=0; --j){
		vectorizer m = *pm;
		m -= vc;
		vectorizer th = *pth;
		th += m1;
		v_min(m, th);
		*pm = m;
		++pm;
		++pth;
	};
#ifdef _DEBUG
	if (dir_fw){
		assert(std::abs(((type*)message)[y_t] - v_first(m1)) < 1e-4);
	} else{
		assert(std::abs(((type*)message)[y_s] - v_first(m1)) < 1e-4);
	};
#endif
};

#undef Delta_st
#undef Delta_ts
#undef dir_vK2

//template<typename type, typename vectorizer>
template<class base>
void term2v_po_reduced<base>::min_sum_e(const int vK1, vectorizer * source, vectorizer * message)const{
	t_min_sum_e<true>(vK1, source, message);
#ifdef _DEBUG // check
	tvect a; a.set_ref(source, count(0));
	tvect r; r.set_ref(message, count(1));
	// reference solution
	tvect r1;
	r1.resize(r.size());
	ref.min_sum(a, r1);
	check(a, r, r1, U_s, U_t, delta_st, delta_ts, y_s, y_t);
#endif
};

//template<typename type, typename vectorizer>
template<class base>
void term2v_po_reduced<base>::min_sum_et(const int vK1, vectorizer * source, vectorizer * message)const{
	t_min_sum_e<false>(vK1, source, message);
#ifdef _DEBUG // check
	int K = delta_ts.size();
	tvect a; a.set_ref(source, count(1));
	tvect r; r.set_ref(message, count(0));
	// reference solution
	tvect r1;
	r1.resize(r.size());
	ref.min_sum_t(a, r1);
	check(a, r, r1, U_t, U_s, delta_ts, delta_st, y_t, y_s);
#endif
};


//template<typename type, typename vectorizer>
template<class base>
void term2v_po_reduced<base>::reduce(po_mask & U_s, int y_s, po_mask & U_t, int y_t){
	//nothing to do
};

//template<typename type, typename vectorizer>
template<class base>
void term2v_po_reduced<base>::rebuild(po_mask & U_s, int y_s, po_mask & U_t, int y_t){
#ifdef _DEBUG
	ref.rebuild(U_s,y_s,U_t,y_t);
	ref.reduce(U_s, y_s, U_t, y_t);
	this->y_s = y_s;
	this->y_t = y_t;
#endif
	assert(U_s[y_s]);
	assert(U_t[y_t]);
	//
	this->U_s = &U_s;
	this->U_t = &U_t;
	int K1 = src().count(0);
	int K2 = src().count(1);
	c = src().base::operator()(y_s, y_t);
	//
	tvect a;
	a.reserve(std::max(K1, K2));
	a.resize(K1);
	src().base::get_col(y_t, a);
	for (int i = 0; i < K1; ++i){
		if (U_s[i]){
			a[i] = -a[i];
		} else{
			a[i] = INF(type);
		};
	};
	src().base::min_sum(a, delta_ts);
	for (int j = 0; j < K2; ++j){
		if (U_t[j]){
			delta_ts[j] = 0;
		};
	};
	//
	a.resize(K2);
	src().base::get_row(y_s, a);
	for (int j = 0; j < K2; ++j){
		if (U_t[j]){
			a[j] = -a[j];
		} else{
			a[j] = INF(type);
		};
	};
	src().base::min_sum_t(a, delta_st);
	for (int i = 0; i < K1; ++i){
		if (U_s[i]){
			delta_st[i] = 0;
		};
	};
#ifdef _DEBUG
	for (int j = 0; j < K2; ++j){
		if (!U_t[j]){
			assert(delta_ts[j] >1e30 || delta_ts[j] == ref.operator()(y_s, j));
		};
	};
	for (int i = 0; i < K1; ++i){
		if (!U_s[i]){
			assert(delta_st[i] >1e30 || delta_st[i] == ref.operator()(i, y_t));
		};
	};
	for (int i = 0; i < K1; ++i){
		for (int j = 0; j < K2; ++j){
			type v1 = ref.operator()(i, j);
			type v2 = operator()(i, j);
			assert( std::abs(v1-v2) < 1e-4 );
		};
	};
	//for (int i = 0; i < K1; ++i){
	//	for (int j = 0; j < K2; ++j){
	//		if (U_t[j] && !U_s[i]){
	//			type v1 = src()(i, j) - src()(y_s, y_t);
	//			type v2 = delta_st(i);
	//			// would work if delta was not defined as zero on U_s
	//			assert(v1 >= v2);
	//		};
	//	};
	//};
#endif
};

//template<typename type, typename vectorizer>
template<class base>
typename base::type term2v_po_reduced<base>::operator()(int i, int j)const{
	type r = operator()(i, j, (*U_s)[i], (*U_t)[j]);
	//throw debug_exception("this must not be called");
#ifdef _DEBUG
	type r1 = ref.operator()(i,j);
	assert(r1>1e25 || std::abs(r-r1) < 1e-3);
#endif
	return r;
};

template<class base>
void term2v_po_reduced<base>::get_row(int i0, tvect & r)const{
#ifdef _DEBUG
	assert(!(*U_s)[i0] || i0== y_s);
#endif
	//performance problem, use src->get_row
	// should resolve with preprocessing then
	src().base::get_row(i0, r);
		// correcting min for the source being just a mask for index i0
	type m1 = delta_st[i0];
		// correction for reduced term
	for (int j = 0; j < tthis::count(1); ++j){
		r[j] = std::min(r[j] - c, delta_ts[j] + m1);
	};
#ifdef _DEBUG
	for (int j = 0; j < r.size(); ++j){
		assert(std::abs(r[j] - term2v_po_reduced::operator()(i0, j))<1e-4);
	};
#endif
};

template<class base>
void term2v_po_reduced<base>::get_col(int j0, tvect & r)const{
#ifdef _DEBUG
	assert(!(*U_t)[j0] || j0==y_t );
#endif
	src().base::get_col(j0, r);
	// correcting min for the source being just a mask for index i0
	type m1 = delta_ts[j0];
	// correction for reduced term
	for (int i = 0; i < tthis::count(0); ++i){
		r[i] = std::min(r[i] - c, delta_st[i] + m1);
	};
	//r[y_s] = m1;
#ifdef _DEBUG
	assert(std::abs(r[y_s] - m1) < 1e-4);
	for (int i = 0; i < tthis::count(0); ++i){
		type v1 = r[i];
		type v2 = term2v_po_reduced::operator()(i, j0);
		type v3 = ref.operator()(i, j0);
		if (!(*U_s)[i] || i == y_s){
			if (std::abs(v1 - v2) > 1e-4){ // only these are the guaranteed entries
				assert(false);
				src().base::get_col(j0, r);
				// correcting min for the source being just a mask for index i0
				type m1 = delta_ts[j0];
				// correction for reduced term
				for (int i = 0; i < tthis::count(0); ++i){
					r[i] = std::min(r[i] - c, delta_st[i] + m1);
				};
			};
		};
	};
#endif
};

template<class base>
void term2v_po_reduced<base>::get_col_e(const int vK1, int j0, vectorizer * message)const{
#ifdef _DEBUG
	assert(!(*U_t)[j0] || j0 == y_t);
#endif
	src().base::get_col_e(vK1,j0,message);
	// correcting min for the source being just a mask for index i0
	vectorizer m1(delta_ts[j0]);
	vectorizer vc(c);
	// correction for reduced term
	for (int i = 0; i < vK1; ++i){
		vectorizer m = message[i] - vc;
		v_min(m, _delta_st[i] + m1);
		message[i] = m;
	};
#ifdef _DEBUG
	assert(std::abs(((type*)message)[y_s] - v_first(m1)) < 1e-4);
#endif
};


//template<typename type, typename vectorizer>
template<class base>
typename base::type term2v_po_reduced<base>::operator()(int i, int j, bool i_inU, bool j_inU)const{
	if (i_inU){
		if (j_inU){
			return 0;
		} else{
			return delta_ts[j];
		};
	} else{
		if (j_inU){
			return delta_st[i];
		} else{
			type a = src().base::operator()(i, j) - c;
			return std::min(a, delta_st[i] + delta_ts[j]);
		};
	};
};

//template<typename type, typename vectorizer>
template<class base>
void term2v_po_reduced<base>::min_sum(tvect & a, tvect & r, po_mask * U_s, po_mask * U_t, const tvect & delta_st, const tvect & delta_ts)const{
	//assume a(i) = inf for i\notin Y_s \cup {y_s}
	//assume delta_st[y_s] = 0
	//assume delta_ts[y_t] = 0
	int K1 = a.size();
	int K2 = r.size();
	/*
	for (int i = 0; i < K1; ++i){
	if ((*U_s)[i] && i!=y_s){
	a[i] = FINF;
	};
	};
	*/
	type m1 = a[0] + delta_st[0];
	for (int i = 1; i < K1; ++i){
		m1 = std::min(m1, a[i] + delta_st[i]);
	};
	// message passing for original f
	//a[y_s] = FINF;
	src().base::min_sum(a, r); // <- todo: what about transposed?
	// correction for reduced term
	for( int j = 0; j < K2; ++j){
		r[j] = std::min(r[j] - c, delta_ts[j] + m1);
	};
	//r[y_t] = delta_ts[y_t] + m1;
	//assume r[j] for j notin Y_t \cup {y_t} has no effect
	/*
	for (int j = 0; j < K2; ++j){
	if ((*U_t)[j] && j != y_t){
	//r[j] = 0;
	r[j] = r[y_t];
	};
	};
	*/
};

//template<typename type, typename vectorizer>
template<class base>
void term2v_po_reduced<base>::check(tvect & a, tvect & r, tvect & r1, po_mask * U_s, po_mask * U_t, const tvect & delta_st, const tvect & detla_ts, int y_s, int y_t)const{
#ifdef _DEBUG
	for (int j = 0; j < r.size(); ++j){
		if (!(*U_t)[j] || j == y_t){
			if (math::abs(r1[j] - r[j]) > 1e-3){
				// print out where difference occurs:
				int K1 = a.size();
				int K2 = r.size();
				debug::stream << "U_s=[";
				for (int i = 0; i < K1; ++i){
					debug::stream << (*U_s)[i] << " ";
				};
				debug::stream << "]\nU_t=[";
				for (int i = 0; i < K2; ++i){
					debug::stream << (*U_t)[i] << " ";
				};
				debug::stream << "]\ny_s=" << y_s << " y_t=" << y_t << "\n";
				debug::stream << " a=" << a << "\n";
				debug::stream << "r1=" << r1 << "\n";
				debug::stream << " r=" << r << "\n";
				//assert(r1[j] == r[j]);
				/*
				debug::stream << "matrix=\n";
				for (int i = 0; i < K2; ++i){
					tvect a(K1);
					a << ref.subdim<1>(i);
					debug::stream << a <<"\n";
				};
				*/
				int bla = 0;
			};
		};
	};
#endif
};


//template<typename type, typename vectorizer>
template<class base>
void inline term2v_po_reduced<base>::min_sum(tvect & a, tvect & r)const{
#ifdef _DEBUG
	// reference solution
	tvect r1; 
	r1.resize(r.size());
	ref.min_sum(a, r1);
#endif
	min_sum(a, r, U_s, U_t, delta_st, delta_ts);
#ifdef _DEBUG
	check(a, r, r1, U_s, U_t, delta_st, delta_ts,y_s,y_t);
#endif
};

//template<typename type, typename vectorizer>
template<class base>
void inline term2v_po_reduced<base>::min_sum_t(tvect & a, tvect & r)const{
#ifdef _DEBUG
	// reference solution
	tvect r1; r1.resize(r.size());
	ref.min_sum_t(a, r1);
#endif
	min_sum(a, r, U_t, U_s, delta_ts, delta_st);
#ifdef _DEBUG
	check(a, r, r1, U_t, U_s, delta_ts, delta_st,y_t,y_s);
#endif
};


//_________________________________energy_______________________________________________

template<typename type>
double energy<type>::cost(const intf & x){
	double r = 0;
	for (int s = 0; s < G.nV(); ++s){
		r += f1[s][x[s]];
	};
	for (int e = 0; e < G.nE(); ++e){
		int s = G.E[e][0];
		int t = G.E[e][1];
		int x_s = x[s];
		int x_t = x[t];
		r += f2(e)(x_s, x_t);
	};
	return r;
};

//___________________________energy_auto______________________________________
template<typename type>
energy_auto<type>::energy_auto(){
	nfull = 0;
	npotts = 0;
	ntlinear = 0;
	ntquadratic = 0;
	ndiff = 0;
	maxf = -INF(type);
	delta = INF(type);
	mult = 1;
};
/*
template<typename type>
bool energy_auto<type>::test_potts(const dynamic::num_array<double, 2> & _f2){
	int K1 = _f2.size()[0];
	int K2 = _f2.size()[1];
	if (K1 != K2)return false; // only accept square matrices here
	if (K1 < 2 || K2 < 2)return false;
	for (int k1 = 0; k1 < K1; ++k1){
		for (int k2 = 0; k2 < K2; ++k2){
			if (k1 == k2){
				if (math::abs(_f2(k1, k2) - _f2(0, 0)) > 1e-8) return false;
			} else{
				if (math::abs(_f2(k1, k2) - _f2(1, 0)) > 1e-8) return false;
			};
		};
	};
	return true;
};
*/
/*
template<typename type>
bool energy_auto<type>::test_tlinear(const dynamic::num_array<double, 2> & _f2){
	// energy in the form  min(|i-j|*gamma,th)
	int K1 = _f2.size()[0];
	int K2 = _f2.size()[1];
	if (K1 != K2)return false; // only accept square matrices here
	if (K1 < 2 || K2 < 2)return false; // too small -> use matrix
	double th = _f2(K1-1, 0); // this must be the threshold or the threshold has no effect
	double gamma = _f2(1, 0); // this must be the slope
	for (int k1 = 0; k1 < K1; ++k1){
		for (int k2 = 0; k2 < K2; ++k2){
			// model value:
			double f = std::min(std::abs(k1 - k2)*gamma, th);
			if (math::abs(_f2(k1, k2) - f) > 1e-8) return false; // must follow the model
		};
	};
	return true;
};
*/

template<typename type>
void energy_auto<type>::set_f1(int v, const dynamic::num_array<double, 1> & _f1){
	int K = _f1.size();
	int Ka = v_align<type, d_vectorizer>(K); // size_align(K);
	this->f1[v].resize(Ka);
	for (int i = 0; i < Ka; ++i){
		if (i < K){
			this->f1[v][i] = type(_f1[i]);
		} else{ // aligning
			this->f1[v][i] = INF(type);
		};
	};
	maxf = std::max(maxf, _f1.maxabs().first);
	delta = std::min(delta, _f1.second_min().first - _f1.min().first);
};

template<typename type>
void energy_auto<type>::set_f2(int e, const dynamic::num_array<double, 2> & _f2){
	// statistics
	maxf = std::max(maxf, _f2.maxabs().first);
	delta = std::min(delta, _f2.second_min().first - _f2.min().first);
	//recognize and remember _f2
	int K1 = _f2.size()[0];
	int K2 = _f2.size()[1];
	// try Potts
	try{
		F2[e] = new term2v_potts<type, d_vectorizer>(_f2);
		// worked;
		++npotts;
		return;
	} catch (...){
		// did not work; go no
	};
	// try T-Linear
	try{
		F2[e] = new term2v_tlinear<type, d_vectorizer>(_f2);
		// worked;
		++ntlinear;
		return;
	} catch (...){
		// did not work; go no
	};

	/*
	
	if (test_potts(_f2)){ // construct potts term
		F2[e] = new term2v_potts<type, vectorizer>(sza, type(_f2(1, 0) - _f2(0, 0)));
		++npotts;
		return;
	};
	if (test_tlinear(_f2)){ // construct t_linear terms
		type th = type(_f2(K1 - 1, 0)); // this must be the threshold or the threshold has no effect
		type gamma = type(_f2(1, 0)); // this must be the slope
		F2[e] = new term2v_tlinear<type, vectorizer>(sza, gamma,th);
		++ntlinear;
		return;
	};
	*/
	// try diff
	try{
		F2[e] = new term2v_diff<type, d_vectorizer>(_f2);
		// worked;
		++ndiff;
		return;
	} catch (...){
		// did not work; go no
	};
	/*
	// try truncated quadratic
	try{
		F2[e] = new term2v_tquadratic<type, vectorizer>(_f2, sza);
		// worked;
		++ntquadratic;
		return;
	} catch (...){
		// did not work; go no
	};
	*/
	//default to full matrix
	/*
	mint2 sza(v_align<type, d_vectorizer>(K1), v_align<type, d_vectorizer>(K2));
	term2v_matrix<type, d_vectorizer> * pf2 = new term2v_matrix<type, d_vectorizer>();
	F2[e] = pf2;
	pf2->resize(sza);
	//(*pf2) << INF(type);
	for (int k2 = 0; k2 < sza[1]; ++k2){
		for (int k1 = 0; k1 < sza[0]; ++k1){
			if (k1 < K1 && k2 < K2){
				(*pf2)(k1, k2) = type(_f2(k1, k2));
			} else{
				(*pf2)(k1, k2) = 0;
			};
		};
	};
	*/
	F2[e] = new term2v_matrix<type, d_vectorizer>(_f2);
	++nfull;
};

template<typename type>
void energy_auto<type>::init(){
	double d = std::numeric_limits<type>::digits10;
	// take digits-3 down from maxf
	double delta1 = std::max(delta, maxf / pow(10, d - 3));
	mult = 1 / pow(10, floor(log10(delta1))); // now according to new delta
	this->tolerance = 1.0 / mult;
}

template<typename type>
void energy_auto<type>::report(){
	debug::stream << "Recognized models: " << npotts << "Potts terms, " << ntlinear << "T-Linear terms,\n " 
		<< ntquadratic << "T-Quadratic terms, " << ndiff << "Diff terms, " << nfull << " Full terms\n";
	debug::stream << "Max data value: " << maxf << "\n";
	debug::stream << "Min data delta: " << delta << "\n";
	debug::stream << "Selecting data toolerance: " << 1.0 / mult << "\n";
};

template<typename type>
void energy_auto<type>::cleanup(){
	for (int e = 0; e < F2.size(); ++e){
		delete F2[e];
	};
};

//template class term2v_po_reduced < float, float > ;
//template class term2v_po_reduced < float, sse_float_4 > ;

template term2v_potts<double, double>::term2v_potts(const num_array<double, 2> &);

template int v_align<float, float>(int size);
template int v_align<double, double>(int size);
template int v_align<float, sse_float_4>(int size);

template class term2v_po_reduced <term2v_potts<float, float> >;
template class term2v_po_reduced <term2v_potts<double, double> >;
template class term2v_po_reduced <term2v_potts<float, sse_float_4> >;

template class term2v_po_reduced <term2v_tlinear<float, float> >;
template class term2v_po_reduced <term2v_tlinear<double, double> >;
template class term2v_po_reduced <term2v_tlinear<float, sse_float_4> >;

template class term2v_po_reduced <term2v_tquadratic<float, float> >;
template class term2v_po_reduced <term2v_tquadratic<double, double> >;
template class term2v_po_reduced <term2v_tquadratic<float, sse_float_4> >;

template class term2v_po_reduced <term2v_diff<float, float> >;
template class term2v_po_reduced <term2v_diff<double, double> >;
template class term2v_po_reduced <term2v_diff<float, sse_float_4> >;

template class term2v_po_reduced <term2v_matrix<float, float> >;
template class term2v_po_reduced <term2v_matrix<double, double> >;
template class term2v_po_reduced <term2v_matrix<float, sse_float_4> > ;

template class term2v_matrix_po < float, float > ;
template class term2v_matrix_po < double, double >;
template class term2v_matrix_po < float, sse_float_4 > ;

template class term2v_matrix < float, float >;
template class term2v_matrix < double, double >;
template class term2v_matrix < float, sse_float_4 >;


template class term2v_potts < double >;

//template class energy_auto <double, double>;
template class energy < d_type >;
template class energy_auto < d_type >;
//template class energy_auto < double >;

//template class energy_auto <float>;
/*
void bla(){
	energy_auto < type > E;
	intf z(0);
	E.cost(z);
};
*/