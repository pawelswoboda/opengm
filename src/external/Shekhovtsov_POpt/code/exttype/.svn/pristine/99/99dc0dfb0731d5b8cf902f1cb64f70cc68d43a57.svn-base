#ifndef fixed_vect_h
#define fixed_vect_h

#include "dynamic/fixed_array1.h"
//#include "dynamic/array_allocator.h"
#include "ivector.h"
#include "debug/logs.h"

namespace exttype{
	template <typename type, class Allocator = DEFAULT_ALLOCATOR> class fixed_vect : public ivector<fixed_vect<type, Allocator>, dynamic::fixed_array1<type, Allocator > >{
	private:
		typedef ivector<fixed_vect<type, Allocator>, dynamic::fixed_array1<type, Allocator> > parent;
	public:
		int count()const{return this->size();};
		fixed_vect(){};
		explicit fixed_vect(int K){// constructor with initialization
			this->resize(K);
			for(int i=0;i<this->size();++i)(*this)[i] = 0;
		};
		explicit fixed_vect(int n, const type v0,...){
			this->resize(n);
			(*this)[0] = v0;
			va_list a;
			va_start(a,v0);
			for(int i=1;i<n;++i){
				(*this)[i] = va_arg(a,type);
			};
			va_end(a);
		};
		//default operator =
		//default copy constructor
	private:
		fixed_vect cpy()const{return fixed_vect(*this);};
	public:
	};

	typedef fixed_vect<char> charf;
	typedef fixed_vect<int> intf;
	typedef fixed_vect<double> doublef;
	typedef fixed_vect<float> floatf;

	template <class type> txt::TextStream & operator << (txt::TextStream & stream, const fixed_vect<type> & x){
		stream << "(";
		for (int i = 0; i<x.size(); ++i){
			stream << x[i];
			if (i<x.size() - 1)stream << ", ";
		};
		stream << ")";
		return stream;
	};

};


#endif
