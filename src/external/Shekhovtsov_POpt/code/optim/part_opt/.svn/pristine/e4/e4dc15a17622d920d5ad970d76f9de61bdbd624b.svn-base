#ifndef vectorizers1_h
#define vectorizers1_h

#include "defs.h"

#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2



/*vectorizer required operations:
vectorizer & operator = (const vectorizer & x)
vectorizer & operator = (const type & f)
vectorizer & operator += (const vectorizer & x)
vectorizer & operator -= (const vectorizer & x)
vectorizer & operatoer *= (const type & f)
vectorizer operator + (const vectorizer & x) const
vectorizer operator - (const vectorizer & x) const
void v_min(vectorizer & a, const vectorizer & b); // a = min(a,b)
std::pair<type v, int i> s_min(vectorizer & a); // put min of a in all elements of a, return value and position
type v_first(vectorizer & a); // return first element in a

*/

// float

__forceinline void v_min(float & x, const float & y){
	if (y < x){ x = y; };
};

__forceinline void v_max(float & x, const float & y){
	if (y > x){ x = y; };
};

__forceinline void v_min(float & x){
};

__forceinline float v_first(float & x){
	return x;
};

__forceinline float v_broadcast_last(float & x){
	return x;
};

__forceinline float v_broadcast_first(float & x){
	return x;
};

__forceinline float v_flip(float & x){
	return x;
};

__forceinline void v_cum_min(float & x){
};

__forceinline void v_rcum_min(float & x){
};

__forceinline float v_load(const float * p){
	return *p;
};

// double

__forceinline void v_min(double & x, const double & y){
	if (y < x){ x = y; };
};

__forceinline void v_max(double & x, const double & y){
	if (y > x){ x = y; };
};

__forceinline void v_min(double & x){
};

__forceinline double v_first(double & x){
	return x;
};

__forceinline double v_broadcast_last(double & x){
	return x;
};

__forceinline double v_broadcast_first(double & x){
	return x;
};

__forceinline double v_flip(double & x){
	return x;
};

__forceinline void v_cum_min(double & x){
};

__forceinline void v_rcum_min(double & x){
};

__forceinline double v_load(const double * p){
	return *p;
};

// float x 4

struct sse_float_4{
public:
	typedef float type;
	typedef sse_float_4 vectorizer;
public:
	__m128 a;
public:
	sse_float_4(const vectorizer & x){ // copy constructor
		a = x.a; // very important
	};

	explicit sse_float_4(const type & f){
		a = _mm_set1_ps(f);
	};

	__forceinline vectorizer & operator = (const vectorizer & x){
		a = x.a;
		return *this;
	};
	__forceinline vectorizer & operator = (const type & f){
		a = _mm_set1_ps(f);
		return *this;
	};

	sse_float_4(){
	};

	__forceinline vectorizer & operator += (const vectorizer & x){
		a = _mm_add_ps(a, x.a);
		return *this;
	};
	__forceinline vectorizer & operator += (const type & f){
		__m128 b = _mm_load1_ps(&f);
		a = _mm_add_ps(a, b);
		return *this;
	};
	__forceinline vectorizer & operator -= (const vectorizer & x){
		a = _mm_sub_ps(a, x.a);
		return *this;
	};
	__forceinline vectorizer & operator -= (const type & f){
		__m128 b = _mm_load1_ps(&f);
		a = _mm_sub_ps(a, b);
		return *this;
	};
	__forceinline vectorizer operator - (){
		vectorizer b(0);
		b -= (*this);
		return b;
	};
	__forceinline vectorizer & operator *= (const type & f){
		__m128 b = _mm_load1_ps(&f);
		a = _mm_mul_ps(a, b);
		return *this;
	};
	__forceinline vectorizer & operator *= (const vectorizer & x){
		a = _mm_mul_ps(a, x.a);
		return *this;
	};
	__forceinline vectorizer operator + (const vectorizer & x)const{
		vectorizer r;
		r.a = _mm_add_ps(a, x.a);
		return r;
	};
	__forceinline vectorizer operator - (const vectorizer & x)const{
		vectorizer r;
		r.a = _mm_sub_ps(a, x.a);
		return r;
	};
};

__forceinline void v_min(sse_float_4 & x, const sse_float_4 & y){
	x.a = _mm_min_ps(x.a, y.a);
};

__forceinline void v_max(sse_float_4 & x, const sse_float_4 & y){
	x.a = _mm_max_ps(x.a, y.a);
};

__forceinline void v_min(sse_float_4 & x){
	__m128 & b = x.a;
	__m128 w = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1));
	w = _mm_min_ps(b, w); // min(0,2) | min(1,3) | min(2,0) | min(3,1)
	b = _mm_shuffle_ps(w, w, _MM_SHUFFLE(1, 0, 3, 2));
	b = _mm_min_ps(b, w); // min in all
};

#define RSHUFFLE(ai,aj,bi,bj) _MM_SHUFFLE(bj,bi,aj,ai)

__forceinline void v_cum_min(sse_float_4 & x){
	// A B C D
	// A A C C   shuffle 1
	// | | | |
	// A E C F,  E = min(A,B), F = min(C,D)
	// A E E E   shuffle 2
	// | | | |
	// A E G H,  G = min(A,B,C), H = min(A,B,C,D)
	__m128 & a = x.a;
	__m128 w = _mm_shuffle_ps(a, a, RSHUFFLE(0, 0, 2, 2));
	w = _mm_min_ps(a, w);
	a = _mm_shuffle_ps(w, w, RSHUFFLE(0, 1, 1, 1));
	a = _mm_min_ps(a, w);
};

__forceinline void v_rcum_min(sse_float_4 & x){
	// A B C D
	// B B D D   shuffle 1
	// | | | |
	// E B F D,  E = min(A,B), F = min(C,D)
	// F F F D   shuffle 2
	// | | | |
	// G H F D,  G = min(A,B,C,D), H = min(B,C,D)
	__m128 & a = x.a;
	__m128 w = _mm_shuffle_ps(a, a, RSHUFFLE(1, 1, 3, 3));
	w = _mm_min_ps(a, w);
	a = _mm_shuffle_ps(w, w, RSHUFFLE(2, 2, 2, 3));
	a = _mm_min_ps(a, w);
};

__forceinline sse_float_4 v_broadcast_last(sse_float_4 & x){
	// A B C D
	// D D D D
	sse_float_4 r;
	r.a = _mm_shuffle_ps(x.a, x.a, RSHUFFLE(3, 3, 3, 3));
	return r;
};

__forceinline sse_float_4 v_broadcast_first(sse_float_4 & x){
	// A B C D
	// A A A A
	sse_float_4 r;
	r.a = _mm_shuffle_ps(x.a, x.a, RSHUFFLE(0, 0, 0, 0));
	return r;
};

__forceinline sse_float_4 v_flip(sse_float_4 & x){
	// A B C D
	// D C B A
	sse_float_4 r;
	r.a = _mm_shuffle_ps(x.a, x.a, RSHUFFLE(3, 2, 1, 0));
	return r;
};

__forceinline float v_first(sse_float_4 & x){
	return _mm_cvtss_f32(x.a);
};

/*
__forceinline sse_float_4 v_load(const sse_float_4 * p){
	sse_float_4 r;
	r.a = _mm_load_ps((const float*)p);
};
*/

//__forceinline template<typename type, typename vectorizer> int v_length(sse_float_4 & x){
//	return 4;
//};


template<typename type, typename vectorizer> struct unaligned_base{
public:
	type a[sizeof(vectorizer) / sizeof(type)];
	static int V(){
		return sizeof(vectorizer) / sizeof(type);
	};
protected:
	unaligned_base(){
	};
public:
	operator vectorizer()const{
		const type * pa = a;
		vectorizer v;
		type * pv = &v;
		for (int i = 0; i < V(); ++i){
			pv[i] = pa[i];
		};
		return v;
	};
	unaligned_base(const unaligned_base & b){
		(*this) = b;
	};
	unaligned_base(const vectorizer & b){
		(*this) = b;
	};
	void operator = (const unaligned_base & b){
		type * pa = a;
		const type * pb = b.a;
		for (int i = 0; i < V(); ++i){
			pa[i] = pb[i];
		};
	};
	void operator = (const vectorizer & v){
		type * pa = a;
		const type * pb = (const type *)&v;
		for (int i = 0; i < V(); ++i){
			pa[i] = pb[i];
		};
	};
};

template<typename type, typename vectorizer> struct unaligned : public unaligned_base < type, vectorizer > {
public:
	typedef unaligned_base < type, vectorizer > parent;
public:
	unaligned(const unaligned & b):parent(b){
	};
	unaligned(const vectorizer & b) :parent(b){
	};
};

template<> struct unaligned<float, sse_float_4> : public unaligned_base < float, sse_float_4 >{
public:
	typedef float type;
	typedef sse_float_4 vectorizer;
	typedef unaligned_base < type, vectorizer > parent;
public:

	operator vectorizer ()const{
		vectorizer v;
		v.a = _mm_loadu_ps(a);
		return v;
	};

	unaligned(const unaligned & b):parent(b){
	};

	unaligned(const vectorizer & v){
		_mm_storeu_ps(a, v.a);
	};
};

#endif