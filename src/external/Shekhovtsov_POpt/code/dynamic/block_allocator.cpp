#include "defs.h"
#include "block_allocator.h"
#include <limits>
#include <string.h>
//#include <omp.h>
#include <algorithm>
#include <iostream>

namespace dynamic{

	//___________________stack_allocator___________________________
	__forceinline int * stack_allocator::cap_beg()const{
		return _capbeg;
	};
	__forceinline int * stack_allocator::beg()const{
		return _beg;
	};
	__forceinline int * stack_allocator::end()const{
		return _end;
	};
	//! size in ints
	__forceinline int stack_allocator::size()const{
		return int(_end - _beg);
	};
	//! capacity in ints
	__forceinline int stack_allocator::capacity()const{
		if (!allocated())return 0;
		return int(_end - _capbeg);
	};
	__forceinline int stack_allocator::cap_free()const{
		return int(_beg - _capbeg);
	};
	__forceinline bool stack_allocator::empty()const{
		return size() == 0;
	};
	__forceinline bool stack_allocator::allocated()const{
		return (_capbeg != 0);
	};
	__forceinline stack_allocator::stack_allocator() :_beg(0), _end(0), _capbeg(0){
	};
	//stack_allocator()
	__forceinline void stack_allocator::attach(int * _Beg, size_t size){
		if (!empty())throw std::bad_alloc();
		_capbeg = _Beg;
		_end = _Beg + size;
		_beg = _end;
	};
	__forceinline void stack_allocator::detach(){
		_beg = 0;
		_end = 0;
		_capbeg = 0;
	};
	__forceinline stack_allocator::stack_allocator(int * _Beg, size_t size) :_capbeg(_Beg){
		_end = _Beg + size;
		_beg = _end;
	};
	//! stealing copy constructor
	__forceinline stack_allocator::stack_allocator(const stack_allocator & x) :_capbeg(0), _beg(0), _end(0){
		(*this) = x;
	};
	//! stealing operator =
	__forceinline void stack_allocator::operator = (const stack_allocator & x){
		if (allocated() || !empty())throw debug_exception("buffer is in use");
		_beg = x._beg;
		_end = x._end;
		_capbeg = x._capbeg;
		x._beg = 0;
		x._end = 0;
		x._capbeg = 0;
	};
	__forceinline stack_allocator::~stack_allocator(){
		//! all objects must have been destroyed
		assert(empty());
	};
	__forceinline bool stack_allocator::is_top_block(int * P)const{
		return (P - overhead == beg());
	};
	__forceinline bool stack_allocator::is_block_used(int * P){
		return block_sign(P) == sign_block_used;
	};
	__forceinline bool stack_allocator::is_block_unused(int * P){
		return block_sign(P) == sign_block_unused;
	};
	__forceinline void stack_allocator::mark_block_used(int * P){
		block_sign(P) = sign_block_used;
	};
	__forceinline void stack_allocator::mark_block_unused(int * P){
		block_sign(P) = sign_block_unused;
	};
	__forceinline int& stack_allocator::block_size(int * P){
		return *(P - 2);
	};
	__forceinline size_t stack_allocator::block_size_bytes(int * P){
		return size_t(*(P - 2))*sizeof(int);
	};
	__forceinline int& stack_allocator::block_sign(int * P){
		return *(P - 1);
	};
	__forceinline bool stack_allocator::can_allocate(size_t size_bytes, int align)const{
		// assume size_bytes is aligned to sizeof(size_t)
		int size = size_bytes/sizeof(int);
		int sz_add_bytes = size_t(beg() - size) % align;
		size = size + sz_add_bytes/sizeof(int);
		return (cap_free() >= size + overhead);
		//return big_size(cap_free())*sizeof(int) >= size_bytes + overhead*sizeof(int);
	};
	__forceinline char * stack_allocator::allocate(int size_bytes, int align){
		//size_bytes is rounded up to multiple of 4
		if (size_bytes == 0){
			perror("Allocating 0 bytes?\n"); fflush(stdout);
			abort();
		};
		// actually assume size_bytes is already aligned, so this just divides
		//int size = (size_bytes + 3) >> 2;
		//int size = size_bytes >> 2;
		// now alignment, must also be multiple of 4
		// how many extra space to add so that (beg()-size) % (align/4) == 0?
		//int sz_add = size_t(beg() - size) % (align / 4);
		//size = size + sz_add;
		int size = size_bytes / sizeof(int);
		int sz_add_bytes = size_t(beg() - size) % align;
		size = size + sz_add_bytes / sizeof(int);
		if (cap_free() < size + overhead){//check if it fits
			throw std::bad_alloc();
		};
		int * P = beg() - size;
		//chech address is aligned
		assert(size_t(P)%(align) == 0);
		block_size(P) = size;
		mark_block_used(P);
		_beg = P - overhead;
		return (char*)P;
	};
	__forceinline bool stack_allocator::deallocate(void * vP){
		int* P = (int*)vP;
		if (is_block_used(P) != true){
			perror("Deallocation failed: bad pointer\n"); fflush(stdout);
			abort();
		};
		//mark for detetion
		mark_block_unused(P);
		if (_beg + overhead == P){//this is the top block of this allocator
			//delete it
			_beg = _beg + overhead + block_size(_beg + overhead);
			if (empty() || is_block_unused(_beg + overhead)){//buffer is empty or more unused blocks
				//triger cascade deallocation
				return true;
			};
		};
		return false;
	};

	__forceinline void stack_allocator::clean_garbage(){
		while (!empty() && is_block_unused(_beg + overhead)){
			_beg = _beg + overhead + block_size(_beg + overhead);
			assert(_beg <= _end);
		};
		assert(empty() || is_block_used(_beg + overhead));
	};
	void stack_allocator::load(char * filename){
		//FILE * f = fopen(filename,"rb");
		//int sz_used;
		//fread(&sz_used,sizeof(sz_used),1,f);
		//if(sz_used<capacity())throw std::bad_alloc("cant load");
	};

	void stack_allocator::unload(char * filename){
	};

	void stack_allocator::check_integrity(){
		if (empty())return;
		int * P = _beg + overhead;
		int alive = 0;
		int dead = 0;
		/*
		if(!is_block_used(P)){
		if(!is_block_unused(P)){
		perror("integrity fails, top undefined\n"); fflush(stdout);
		abort();
		};
		//perror("integrity fails, top unused\n"); fflush(stdout);
		//abort();
		}
		*/
		while (P != _end + overhead){
			//check signature
			bool used = is_block_used(P);
			bool unused = is_block_unused(P);
			if (!(used || unused)){
				perror("integrity fails, signatur\n"); fflush(stdout);
				abort();
			};
			if (used){
				++alive;
			} else{
				++dead;
			};
			int sz = block_size(P);
			//check size
			if (P + sz>_end){
				perror("integrity fails, size\n"); fflush(stdout);
				abort();
			};
			// go to next block
			P = P + sz + overhead;
		};
		std::cout << "block: " << alive << " alive and: " << dead << " dead objects\n";
	};


	//__________________block_allocator________________________
	__forceinline void block_allocator::took_mem(size_t size_bytes){
		current_reserved += size_bytes;
		peak_reserved = std::max(peak_reserved, current_reserved);
	};
	__forceinline void block_allocator::released_mem(size_t size_bytes){
		current_reserved -= size_bytes;
	};

	void block_allocator::error_allocate(big_size size_bytes, const char * caller){
		//char s[200];
		printf("Error: memory allocation in %s\n", caller);
		printf("memory requested: %lli\n", big_size(size_bytes));
		printf("Total reserved: %lli\n", big_size(mem_reserved()));
		fflush(stdout);
		abort();
		throw std::bad_alloc();
	};

	__forceinline void block_allocator::add_buffer(size_t buffer_size_sp){
		if (spare.allocated() && spare.capacity()*size_t(sizeof(int)) >= buffer_size_sp){//have required ammount in the spare
			buffers.push_back(spare);//steal constructor will make spare empty
		} else{//spare is empty or too small
			//get a new buffer
			stack_allocator * buf = buffers.push_back();
			int * p = (int*)malloc(buffer_size_sp);
			if (!p)error_allocate(buffer_size_sp, "malloc");
			buf->attach(p, buffer_size_sp / sizeof(int));
			took_mem(buf->capacity()*sizeof(int));
		};
	};

	__forceinline void block_allocator::drop_buffer(){
		assert(!buffers.empty());
		if (spare.allocated()){
			int * p = spare.cap_beg();
			int cap = spare.capacity()*sizeof(int);
			assert(spare.empty());
			spare.detach();
			free(p);
			released_mem(cap);
		};
		spare = buffers.back();//spare steals the back buffer
		buffers.pop_back();//remove the empty buffer from the list
	};

	//!clean unused blocks in the buffers and drop empty buffers
	void block_allocator::clean_garbage(){
		while (!buffers.empty()){
			buffers.back().clean_garbage();
			if (buffers.back().empty()){//top buffer is empty
				drop_buffer();
			} else{//top buffer still has data
				return;
			};
		};
	};

	block_allocator::block_allocator(size_t default_buffer_size) :buffer_size(default_buffer_size){
		current_reserved = 0;
		peak_reserved = 0;
		current_used = 0;
		alloc_count = 0;
		buffers.reserve(1000);//Most probably never going to be reallocated
	};

	//__forceinline
	void block_allocator::reserve(size_t reserve_buffer_size){
#pragma omp critical(mem_allocation)
		{
			clean_garbage();
			add_buffer(reserve_buffer_size);
		};
	};

	block_allocator::block_allocator(const block_allocator & x) :buffer_size(x.buffer_size){
		current_reserved = 0;
		peak_reserved = 0;
		current_used = 0;
		alloc_count = 0;
	};

	void block_allocator::operator=(const block_allocator & x){
		current_reserved = 0;
		peak_reserved = 0;
		current_used = 0;
	};

	block_allocator::~block_allocator(){
#pragma omp critical (mem_allocation)
		{
			clean_garbage();
			printf("peak mem usage: %lli Mb ", big_size(mem_peak_reserved() / (1 << 20)));
			printf(" / %i allocations, ", alloc_count);
			printf("at exit: %lli B\n", big_size(mem_used()));
			//assert(buffers.empty() && spare.empty());
			if (spare.allocated()){
				int * p = spare.cap_beg();
				spare.detach();
				free(p);
			};
			if (!(buffers.empty() && spare.empty())){
				try{
					fprintf(stderr, "Memory leaks detected\n");
				} catch (...){
				};
			};
		};
	};

	size_t block_allocator::mem_used()const{
		return current_used;
	};

	size_t block_allocator::mem_reserved()const{
		size_t m = current_reserved - spare.cap_free()*sizeof(int);
		if (!buffers.empty())m -= buffers.back().cap_free()*sizeof(int);
		return m;
	};

	size_t block_allocator::mem_peak_reserved()const{
		return peak_reserved;
	};


	__forceinline size_t block_allocator::round_up(size_t size_bytes){
		return (((size_bytes + 3) >> 2) << 2);
	};

	__forceinline size_t block_allocator::align_up(size_t size_bytes){
		//return ((size_bytes + sizeof(size_t) - 1) / sizeof(size_t))*sizeof(size_t);
		return ((size_bytes + 15) >> 4)<< 4;
	};

	void * block_allocator::protect_allocate(size_t size_bytes, int align){
		int * P;
		size_bytes = round_up(size_bytes);
		++alloc_count;
		if (!buffers.empty() && buffers.back().can_allocate(size_bytes,align)){//fits in the top buffer
			P = (int*)buffers.back().allocate((int)size_bytes,align);
			//current_used += round_up(size_bytes);
			current_used += stack_allocator::block_size_bytes(P);
			return P;
		};
		if (spare.can_allocate(size_bytes,align)){//fits in the spare buffer
			buffers.push_back(spare);
			P = (int*)buffers.back().allocate((int)size_bytes,align);
			//current_used += round_up(size_bytes);
			current_used += stack_allocator::block_size_bytes(P);
			return P;
		};
		if (size_bytes < buffer_size / 16){//is small{
			add_buffer(buffer_size);
			P = (int*)buffers.back().allocate((int)size_bytes,align);
			//current_used += round_up(size_bytes);
			current_used += stack_allocator::block_size_bytes(P);
			return P;
		};
		// is large and does not fit in available buffers
		//use malloc for it (with 2 ints of overhead for signature and size)
		big_size cap = round_up(size_bytes);
		big_size size_allocate = cap + sizeof(int) + sizeof(size_t);
		if (size_allocate > (big_size)(std::numeric_limits<std::size_t>::max() / 2)){
			error_allocate(size_allocate , "size_check");
		};
		int * Q = (int*)malloc(size_t(size_allocate));
		if (Q == 0)error_allocate(size_allocate, "malloc (2)");
		took_mem(size_t(cap));
		P = (int*)((char*)Q + sizeof(int) + sizeof(size_t));
		*(P - 1) = sign_malloc;
		*((size_t*)(P - 1) - 1) = (size_t)cap;
		current_used += (size_t)cap;
		return P;
	};

	void * block_allocator::allocate(size_t size_bytes, int align){
		void * P;
#pragma omp critical (mem_allocation)
		P = protect_allocate(size_bytes, align);
		return P;
	};

	__forceinline size_t block_allocator::object_size(void * vP){
		assert(vP != 0);
		int * P = (int*)vP;
		int sign = stack_allocator::block_sign(P);
		if (sign == sign_block_used){
			return stack_allocator::block_size(P)*sizeof(int);
		} else if (sign == sign_malloc){
			return *((size_t*)(P - 1) - 1);
		} else{
			printf("Error:unrecognized signature\n");
			throw std::bad_alloc();
		};
	};

	//__forceinline
	void block_allocator::protect_deallocate(void * vP){
		//if(x==0)throw debug_exception("Invalid pointer.");
		if (vP == 0){
			perror("Deallocation failed: zero pointer\n"); fflush(stdout);
			abort();
		};
		int * P = (int*)vP;
		//check the signature
		int sign = stack_allocator::block_sign(P);
		if (sign == sign_block_used){//allocated by stack_allocator
			assert(!buffers.empty());
			size_t cap = stack_allocator::block_size(P)*sizeof(int);
			current_used -= cap;
			assert(current_used >= 0);
			//does not matter if cP is not from the top buffer (or even from other allocator) -- in that case it will only be marked for deallocation
			if (buffers.back().deallocate(vP)){//returns 1 when need to clean
				clean_garbage();
			};
			return;
		};
		if (sign == sign_malloc){//was a large separate block
			size_t cap = *((size_t*)(P - 1) - 1);
			stack_allocator::block_sign(P) = 321321321;
			current_used -= cap;
			assert(current_used >= 0);
			free((char*)P - sizeof(int) - sizeof(size_t));
			released_mem(cap);
			return;
		};
		if (sign == sign_block_unused){//this isn't good
			printf("Error: Block is already deallocated\n");
			throw std::bad_alloc();
		};
		printf("Unrecognized signature: memory corrupt\n");
		throw std::bad_alloc();
	};

	void block_allocator::deallocate(void * vP){
#pragma omp critical (mem_allocation)
		protect_deallocate(vP);
	};

	//__forceinline
	void* block_allocator::realloc(void * vP, size_t size_bytes){
		if (!vP)return allocate(size_bytes);
		size_t sz = object_size(vP);
		void * vQ = allocate(size_bytes);
		memcpy(vQ, vP, std::min(sz, size_bytes));
		deallocate(vP);
		//current_used = current_used -sz + round_up(size_bytes);
		return vQ;
	};

	void block_allocator::check_integrity(){
		for (int b = 0; b<buffers.size(); ++b){
			buffers[b].check_integrity();
		};
		std::cout << "block_allocator integrity checked\n";
	};

	//__________________page_allocator_________________________
	size_t page_allocator::load(const char * filename){
		clean_garbage();
		FILE * f = fopen(filename, "r+b");
		int sz_used;
		fread(&sz_used, sizeof(sz_used), 1, f);
		reserve(sz_used);
		assert(buffers.size() == 1);
		assert(buffers.back().empty());
		void * vP = buffers.back().end() - sz_used;
		int sz_read = fread(vP, sizeof(int), sz_used, f);
		if (sz_read < sz_used){
			perror("");
			throw debug_exception("fread failed");
		};
		buffers.back()._beg = (int*)vP;
		fclose(f);
		current_used = sz_used*sizeof(int);
		return sz_used*sizeof(int);
	};

	size_t page_allocator::unload(const char * filename){
		assert(buffers.size() == 1);
		FILE * f = fopen(filename, "w+b");
		if (!f)throw debug_exception("cant open file");
		int sz_used = buffers.back().size();
		int sz_written = 0;
		fwrite(&sz_used, sizeof(sz_used), 1, f);
		//wrtires only the first buffer
		int a = fwrite(buffers.back().beg(), sizeof(int), sz_used, f);
		if (a < sz_used){
			perror("");
			throw debug_exception("fwrite failed");
		};
		fclose(f);
		buffers.back()._beg = buffers.back()._end;
		current_used = 0;
		return sz_used*sizeof(int);
	};
	//_________________memserver____________________________
	void memserver::set_allocator(block_allocator & al){
		get_global() = &al;
		al.clean_garbage();//top object of al could have been marked unused in the meantime
	};

	void memserver::error_allocate(big_size size_bytes){
		get_global()->error_allocate(size_bytes, "external");
	};

	block_allocator & memserver::get_al_blocks(){
		static block_allocator al_blocks;
		return al_blocks;
	};
	block_allocator *& memserver::get_global(){
		static block_allocator * global = &get_al_blocks();
		return global;
	};
};

namespace custom_new{
#ifdef CUSTOM_NEW
	void* operator new(size_t _Count){// throw(std::bad_alloc){
		void * r;
//#pragma omp critical (mem_allocation)
		r = dynamic::memserver::get_global()->allocate(_Count);
		return r;
		//return malloc(_Count);
	};

	void operator delete(void* _Ptr){// throw( ){
//#pragma omp critical (mem_allocation)
		dynamic::memserver::get_global()->deallocate(_Ptr);
		//free(_Ptr);
	};

	void operator delete[](void* _Ptr){
//#pragma omp critical (mem_allocation)
		dynamic::memserver::get_global()->deallocate(_Ptr);
		//free(_Ptr);
	};
#endif
}

void * mmalloc(size_t size_bytes){
	void * r;
//#pragma omp critical (mem_allocation)
	r = dynamic::memserver::get_global()->allocate(size_bytes);
	return r;
};

void * mcalloc(size_t size_bytes){
	void * P;
//#pragma omp critical (mem_allocation)
	P = dynamic::memserver::get_global()->allocate(size_bytes);
	if (P){
		memset(P, 0, size_bytes);
	};
	return P;
};


void * mrealloc(void * ptr, size_t new_size){
	void * r;
//#pragma omp critical (mem_allocation)
	r = dynamic::memserver::get_global()->realloc(ptr, new_size);
	return r;
};

void mfree(void * ptr){
//#pragma omp critical (mem_allocation)
	dynamic::memserver::get_global()->deallocate(ptr);
};

//std::list<int *> extern_bla;
