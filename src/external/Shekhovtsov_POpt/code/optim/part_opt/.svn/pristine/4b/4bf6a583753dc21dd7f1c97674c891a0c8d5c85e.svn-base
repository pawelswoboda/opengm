#ifndef aallocator_h
#define aallocator_h

#include "defs.h"
#include "massert.h"

class aallocator{
public:
	const int align;
	char * mem_beg;
	char * mem_end;
	char * mem_top;
public:
	aallocator(int _align=16):align(_align){
		mem_beg = 0;
		mem_end = 0;
		mem_top = 0;
	};
	void destroy(){
		if (mem_beg){
			free(mem_beg);
		};
		mem_beg = 0;
		mem_end = 0;
		mem_top = 0;
	};
	~aallocator(){
		destroy();
	};
	void init(size_t size_bytes){
		//std::cout << "aalloc: " << size_bytes << " bytes\n";
		size_bytes += align; // for initial adress missalignment
		std::cout << "aalloc: " << size_bytes << " bytes\n";
		mem_beg = (char*)malloc(size_bytes);
		mem_end = mem_beg + size_bytes;
		mem_top = mem_beg;
	};
	void init(){
		init(mem_top - mem_beg);
	};

	/*
	template<class T> T * post_allocate(T * t, int sz, int count = 1){// aligns
		//std::cout << "aal:  " << sz << "bytes\n";
		
	};
	*/

	void align_top(){
		//padd top to the alignment
		int sz = (size_t)mem_top % align;
		if (sz > 0){
			sz = align - sz;
		};
		//std::cout << "aal: +" << sz << "bytes\n";
		mem_top += sz;
		assert((size_t)mem_top % align == 0);
	};

	//! Allocate array of elements, with first element aligned
	template<class T> T * allocate_a(int count = 1){// aligns
		align_top();
		return allocate<T>(count);
	};

	//! Allocate array of elements, without aligning unless element size is alizgned
	template<class T> T * allocate(int count = 1){
		if (sizeof(T) % 16 == 0){
			// type T is of some aligned size, requst aligned allocation?
			align_top();
		};
		T * t = 0;
		int sz = sizeof(T)*count;
		if (mem_beg){ // for real, check fits what was booked
			if (mem_top > mem_end){
				throw debug_exception("insufficient space booked");
				exit(1);
			};
			t = (T*)mem_top;
			//placement new, initialize with default c-tr
			for (int i = 0; i < count; ++i){
				new ((void *)&t[i]) T();
			};
		};
		mem_top = mem_top + sz;
		return t;
	};

	// allocate and initialize from (S, this)
	template<class T, class S> T * allocate(const S & src){
		if (sizeof(T) % 16 == 0){
			align_top();
		};
		T * t = 0;
		int sz = sizeof(T);
		if (mem_beg){
			t = (T*)mem_top;
		};
		mem_top = mem_top + sz;
		// construct further
		if (!mem_beg){ // test allocation, to get the size
			//construct a temporary T on the stack, properly initialize
			T temp(src, this); // T might want to allocate some further data, count all that
		};
		if (mem_beg){ // for real, check fits what was booked
			if (mem_top > mem_end){
				throw debug_exception("insufficient space booked");
				exit(1);
			};
			//placement new, construct object(s) from (S,this), in case they want to allocate too 
			new ((void *)t) T(src, this);
		};
		return t;
	};
};


#endif