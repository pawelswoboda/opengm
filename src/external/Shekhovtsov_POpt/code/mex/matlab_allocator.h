#ifndef matlab_allocator_h
#define matlab_allocator_h

#include <mex.h>

//________________________matlab_allocator________________________
//! Implementation of std::allocator with mxCalloc
template<class T> class matlab_allocator{
public:
	typedef mwSize size_type;
	typedef mwSize difference_type;
	typedef T *pointer;
	typedef const T *const_pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef T value_type;
	pointer address(reference x) const{return &x;};
	const_pointer address(const_reference x) const{return &x;};
public:
	//default constructor
	//default copy constructor
	//default operator =
public:
	pointer allocate(size_type n, const void *hint=0){
		if(n==0)throw debug_exception("Invalid to allocate 0 objects.");
		void * P = mxCalloc(n,sizeof(T));
		return (T*)P;
	};

	void deallocate(pointer x, size_type n){
		if(n==0)throw debug_exception("Invalid to deallocate 0 objects.");
		if(x==0)throw debug_exception("Invalid pointer.");
		mxFree(x);
	};

	void construct(pointer p){
		new ((void *)p) T();
	};

	void construct(pointer p, const T & val){
		new ((void *)p) T(val);
	};

	void destroy(pointer p){
		p->~T();
	};
public:
	template<class U> struct rebind{
		typedef matlab_allocator<U> other;
	};
};
#endif