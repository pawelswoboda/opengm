#ifndef vectorizers2_h
#define vectorizers2_h

#include "defs.h"

#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <immintrin.h> //AVX

//_____________________ double x 4 ________________________________

struct sse_double_4{
public:
	typedef double type;
	typedef sse_double_4 vectorizer;
public:
	__m256d a;
public:
	__forceinline vectorizer & operator = (const vectorizer & x){
		a = x.a;
		return *this;
	};
	__forceinline vectorizer & operator = (const type & f){
		//a = _mm256_broadcast_sd(&f);
		a = _mm256_set1_pd(f);
		return *this;
	};
	sse_double_4(){
	};
	explicit sse_double_4(const type & f){
		a = _mm256_set1_pd(f);
	};
	__forceinline vectorizer & operator += (const vectorizer & x){
		a = _mm256_add_pd(a, x.a);
		return *this;
	};
	__forceinline vectorizer & operator += (const type & f){
		__m256d b = _mm256_broadcast_sd(&f);
		a = _mm256_add_pd(a, b);
		return *this;
	};
	__forceinline vectorizer & operator -= (const vectorizer & x){
		a = _mm256_sub_pd(a, x.a);
		return *this;
	};
	__forceinline vectorizer & operator -= (const type & f){
		__m256d b = _mm256_broadcast_sd(&f);
		a = _mm256_sub_pd(a, b);
		return *this;
	};
	__forceinline vectorizer & operator *= (const type & f){
		__m256d b = _mm256_broadcast_sd(&f);
		a = _mm256_mul_pd(a, b);
		return *this;
	};
	__forceinline vectorizer operator + (const vectorizer & x){
		vectorizer r;
		r.a = _mm256_add_pd(a, x.a);
		return r;
	};
	__forceinline vectorizer operator - (const vectorizer & x){
		vectorizer r;
		r.a = _mm256_sub_pd(a, x.a);
		return r;
	};
};

__forceinline void v_min(sse_double_4 & x, const sse_double_4 & y){
	x.a = _mm256_min_pd(x.a, y.a);
};

#define BINARY(b3,b2,b1,b0) (b3<<3 | b2 << 2 | b1 << 1 | b0)

__forceinline void v_min(sse_double_4 & x){
	__m256d & b = x.a;
	__m256d w = _mm256_permute2f128_pd(b, b, 1); // elements [2 3 0 1]
	w = _mm256_min_pd(b, w); // min(0,2) | min(1,3) | min(2,0) | min(3,1)
	b = _mm256_permute_pd(w, BINARY(0,1,0,1) );
	b = _mm256_min_pd(b, w);
};

__forceinline double v_first(sse_double_4 & x){
	double r;
	_mm256_storeu_pd(&r, x.a);
	return r;
};

#endif