#ifndef mex_io_h
#define mex_io_h

#ifdef WIN32
#include <yvals.h>
#endif

//#ifndef CHAR16_T
//#define CHAR16_T char16_t
//#endif
//#define _HAS_CHAR16_T_LANGUAGE_SUPPORT 1

#include <mex.h>
#include <typeinfo>
#include <engine.h>

#include "dynamic/num_array.h"
#include "streams/xstringstream.h"

#include "dynamic/options.h"

typedef struct engine Engine;
extern Engine *matlab;

class mx_exception{
public:
	mx_exception(const char * s){
		mexErrMsgTxt(s);
	};
	mx_exception(const std::string & s){
		mexErrMsgTxt(s.c_str());
	};
};


template<typename type> mxClassID mexClassId();

template<typename type> bool is_type(const mxArray *A){
	return (mexClassId<type>()==mxGetClassID(A));
};

template<typename type> void check_type(const mxArray *A){
	if(A==0)throw mx_exception("Bad mxArray");
	if(!is_type<type>(A))throw mx_exception((txt::String("Type ")+typeid(type).name()+" expected instead of type "+mxGetClassName(A)+" provided.").c_str());
	//if(!is_type<type>(A))throw mx_exception((String("Type ")+"no_typeinfo"+" expected instead of type "+mxGetClassName(A)+" provided.").c_str());
};

class mx_string : public std::string{
public:
	explicit mx_string(const mxArray * A){
		check_type<char*>(A);
		size_t data_length = mxGetNumberOfElements(A);
		static_cast<std::string&>(*this) = mxArrayToString(A);
	};
	mx_string(){};
	mxArray * get_mxArray()const{
		//allocate full copy for matlab-managed output
		//matlab mem manager takes care this is not lost
		mxArray * p = mxCreateString(this->c_str());
		return p;
	};
};

template<class type, int dims> class mx_array : public dynamic::num_array<type,dims>{
private:
	mxArray * _A;
	bool owned;
public:
	typedef dynamic::num_array<type,dims> parent;
	typedef typename parent::tindex tindex;
public:// mex input /output
	//! construct from mxArray * (used on input to mexFunction)
	explicit mx_array(const mxArray * A):_A(0),owned(false){
		//if type does not match throw exception
		check_type<type>(A);
		type * data = (type*)mxGetData(A);
		size_t data_length = mxGetNumberOfElements(A);

		//read off size
		//note: in matlab number of dimensions is alvays >=2 and can be less then expected if trailing dimensions are 1
		const int ndims = mxGetNumberOfDimensions(A);
		const int * sz = mxGetDimensions(A);
		// if not enough dimensions to represent A throw exception
		if((ndims>2 && dims<ndims) || (ndims==2 && dims==1 && sz[0]>1 && sz[1]>1)) throw mx_exception("Not enough dimensions to represent mxArray");
		int i0 = 0;
		//handle row vectors:
		tindex Size;
		if(ndims==2 && dims==1 && sz[i0]==1)++i0;
		//read remaining dimensions
		for(int i=0;i<std::min(dims,ndims-i0);++i){
			Size[i] = sz[i+i0];
		};
		//set missing dimensions to 1
		for(int i=ndims;i<dims;++i){
			Size[i] = 1;
		};
		// take the pointer
		assert(linsize(Size)==data_length);
		set_ref(data,Size);
		_A = (mxArray*)A;
	};

	//! get mxArray to assign to matlab output (to lhs of mexFunction)
	mxArray * get_mxArray_andDie(){
		if(!owned){
			//allocate full copy for matlab-managed output
			//matlab mem manager takes care this is not lost
			mxArray * p = mxCreateNumericArray(dims,this->size().begin(),mexClassId<type>(),mxREAL);
			type * p_data = (type*)mxGetData(p);
			std::copy(this->begin(),this->end(),p_data);
			return p;
		}else{// allready newly allocated mxArray
			mxArray *out = _A;
			set_ref((const type*)0,tindex()<<0);
			_A = 0;
			return out;
		};
	};
	//! get const mxArray to pass somewhere as argument
	const mxArray * get_mxArray()const{
		return _A;
	};
private:
	void destroy(){
		if(owned && _A){
			mxDestroyArray(_A);
		};
	};
public:
	~mx_array(){
		destroy();
	};
public: //refresh base constructors
	mx_array():_A(0),owned(true){};

	//! construct num_array of the given size
	explicit mx_array(const tindex & sz):parent(),_A(0),owned(true){
		//Allocate on matlab's
		_A = mxCreateNumericArray(dims,sz.begin(),mexClassId<type>(),mxREAL);
		type * data = (type*)mxGetData(_A);
		set_ref(data,sz);
	};
	//! construct num_array of the given size
	void resize(const tindex & sz){
		if (!owned){
			throw mx_exception("Cannot resize not owned array");
		};
		//Allocate on matlab's
		mxArray * A = mxCreateNumericArray(dims, sz.begin(), mexClassId<type>(), mxREAL);
		type * data = (type*)mxGetData(A);
		destroy();
		set_ref(data, sz);
		_A = A;
	};

static mxArray * out(const dynamic::num_array<type,dims> & x){
		//Allocate on matlab's
		mxArray * A = mxCreateNumericArray(dims,x.size().begin(),mexClassId<type>(),mxREAL);
		type * data = (type*)mxGetData(A);
		memcpy(data,x.begin(),x.length()*sizeof(type));
		return A;
	};

	explicit mx_array(const type * V, tindex & sz):parent(),_A(0),owned(true){
		//Allocate a copy on matlab's (mxArray does not allow to set pointer to arbitrary memory)
		_A = mxCreateNumericArray(dims,sz.begin(),mexClassId<type>(),mxREAL);
		type * data = (type*)mxGetData(_A);
		set_ref(data,sz);
		std::copy(V,V+length(),this->begin());
	};

	explicit mx_array(const dynamic::num_array<type,dims> & x):parent(),_A(0),owned(true){
		//Full copy
		//Allocate a copy on matlab's (mxArray does not allow to set pointer to arbitrary memory)
		tindex sz = x.size();
		_A = mxCreateNumericArray(dims,sz.begin(),mexClassId<type>(),mxREAL);
		type * data = (type*)mxGetData(_A);
		set_ref(data,sz);
		std::copy(x.begin(),x.end(),begin());
	};

public:
	mx_array(const mx_array<type,dims> & x):parent(),_A(0),owned(true){
		//Full copy
		//Allocate a copy on matlab's (mxArray does not allow to set pointer to arbitrary memory)
		tindex sz = x.size();
		_A = mxCreateNumericArray(dims,sz.begin(),mexClassId<type>(),mxREAL);
		type * data = (type*)mxGetData(_A);
		set_ref(data,sz);
		std::copy(x.begin(),x.end(),begin());
		/*
		_A = x._A;
		type * data = (type*)mxGetData(_A);
		set_ref(data,x.size());
		owned = false;
		*/
	};
	void operator = (const dynamic::num_array<type,dims> & x){
		destroy();
		//Full copy
		//Allocate a copy on matlab's (mxArray does not allow to set pointer to arbitrary memory)
		tindex sz = x.size();
		_A = mxCreateNumericArray(dims,sz.begin(),mexClassId<type>(),mxREAL);
		type * data = (type*)mxGetData(_A);
		set_ref(data,sz);
		std::copy(x.begin(),x.end(),begin());
	};

public:
	//base copy constructor
	//template<class type2> mx_array(const mx_array<type2,dims> & x):parent(x){};

	//! full copy with type conversion
	//template<class type2> mx_array(const dynamic::num_array<type2,dims> & x):parent(x){};
	//! auto cast from num_array
	//mx_array(const num_array<type,dims> & x):parent(x.ref()){};
public:// mx_array static
	static double mx_inf(){return mxGetInf();};
	static double mx_nan(){return mxGetNaN();};
};

template<typename type, int dims> mxArray * mx_out(mx_array<type,dims> & a){
	return mx_array<type,dims>::out(a);
};

template<typename type, int dims> mxArray * mx_out(dynamic::num_array<type,dims> & a){
	return mx_array<type,dims>::out(a);
};

template<typename type> type mx_get_field(const mxArray * A, const char * field, int ind=0){
	if(A==0 || !mxIsStruct(A))throw mx_exception("struct expected");
	mxArray *a = mxGetField(A,ind,field);
	if(a==0 || mxIsEmpty(a)){
		throw mx_exception(std::string(field) + ": structure field not found");
	};
	check_type<type>(a);
	size_t data_length = mxGetNumberOfElements(a);
	if(data_length!=1)throw mx_exception(std::string(field) + ": scalar field expected");
	type * data = (type*)mxGetData(a);
	return *data;
};

template<int n, int dims> mxArray * mx_out(dynamic::num_array<exttype::floatn<n>,dims> & a){
	typedef float type;
	exttype::intn<dims+1> sz = mint1(n)|a.size();
	mxArray * A = mxCreateNumericArray(dims+1,sz.begin(),mexClassId<type>(),mxREAL);
	type * data = (type*)mxGetData(A);
	memcpy(data,a.begin(),sz.prod()*sizeof(type));
	return A;
};
//__________________________________________________________________________________________

class mx_struct : public options {
public:
	typedef options parent;
public:
	explicit mx_struct(const mxArray * A){
		check_type<mx_struct>(A);
		//size_t data_length = mxGetNumberOfElements(A);
		int nfields = mxGetNumberOfFields(A);
		int NStructElems = mxGetNumberOfElements(A);
		if (NStructElems > 1)throw mx_exception("this struct cannot take arrays");
		int jstruct = 0;
		for (int i = 0; i < nfields; ++i){
			std::string name = mxGetFieldNameByNumber(A, i);
			mxArray * tmp = mxGetFieldByNumber(A, jstruct, i);
			mx_array<double, 1> a(tmp);
			if (a.length()>0){
				parent::operator[](name) = a[0];
			};
		};
	};
	mx_struct(){};
	mxArray * get_mxArray(){
		//allocate full copy for matlab-managed output
		//matlab mem manager takes care this is not lost
		int nfields = parent::size();
		const char **fnames;
		int ifield = 0;
		for (parent::iterator it = parent::begin(); it != end(); ++it, ++ifield){
			fnames[ifield] = it->first.c_str();
		};
		mxArray * A = mxCreateStructMatrix(1, 1, nfields, fnames);
		for (parent::iterator it = parent::begin(); it != end(); ++it, ++ifield){
			mx_array<double, 1> a(1);
			a[0] = it->second;
			mxSetFieldByNumber(A, 0, ifield, a.get_mxArray_andDie());
		};
		return A;
	};
};

//__________________________________________________________________________________________
//
void start_engine();

template<typename type, int dims> void eng_out(dynamic::num_array<type,dims> & a, char * name){
	mxArray * A = mxCreateNumericArray(dims,a.size().begin(),mexClassId<type>(),mxREAL);
	type * data = (type*)mxGetData(A);
	memcpy(data,a.begin(),a.length()*sizeof(type));
	engPutVariable(matlab,name,A);
	mxDestroyArray(A);
};

template<int n, int dims> void eng_out(dynamic::num_array<exttype::intn<n>,dims> & a, char * name){
	typedef int type;
	exttype::intn<dims+1> sz = mint1(n)|a.size();
	mxArray * A = mxCreateNumericArray(dims+1,sz.begin(),mexClassId<type>(),mxREAL);
	type * data = (type*)mxGetData(A);
	memcpy(data,a.begin(),sz.prod()*sizeof(type));
	engPutVariable(matlab,name,A);
	mxDestroyArray(A);
};

template<int n, int dims> void eng_out(dynamic::num_array<exttype::floatn<n>,dims> & a, char * name){
	typedef float type;
	exttype::intn<dims+1> sz = mint1(n)|a.size();
	mxArray * A = mxCreateNumericArray(dims+1,sz.begin(),mexClassId<type>(),mxREAL);
	type * data = (type*)mxGetData(A);
	memcpy(data,a.begin(),sz.prod()*sizeof(type));
	engPutVariable(matlab,name,A);
	mxDestroyArray(A);
};


#endif
