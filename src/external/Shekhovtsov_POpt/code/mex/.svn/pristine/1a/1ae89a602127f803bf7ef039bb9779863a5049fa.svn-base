#ifndef my_mexargs_h 
#define my_mexargs_h 

#include <mex.h>
//#include "matrix.h" //from matlab too

#include "data/dataset.h"
#include "exttype/pvect.h"
#include "exttype/intn.h"
#include "exttype/fixed_vect.h"
//#include <typeinfo>
#include "streams/xstringstream.h"

#include "mex_io.h"

namespace mexargs{
	using namespace exttype;
	using txt::String;
	class MexStream : public txt::TextStream{
	private:
		virtual TextStream & write(const char * x){
			mexPrintf("%s",x);
			return *this;
		};
	public:
	};

	class MexLogStream: public txt::pTextStream{
	public:
		MexLogStream(std::string filename, bool append=true){
			attach(new txt::TabbedTextStream(txt::EchoStream(txt::FileStream(filename.c_str(),append),MexStream())),true);
		};
		~MexLogStream(){
			detach();
		};
		//		static MexLogStream log;
	};
	//typecheck

	/*
	template<typename type> mxClassID mexClassId();

	template<typename type> bool is_type(const mxArray *A){
		return (mexClassId<type>()==mxGetClassID(A));
	};


	template<typename type> void check_type(const mxArray *A){
		//if(!is_type<type>(A))throw debug_exception((String("Type ")+typeid(type).name()+" expected instead of type "+mxGetClassName(A)+" provided.").c_str());
		if(!is_type<type>(A))throw debug_exception((String("Type ")+"no_typeinfo"+" expected instead of type "+mxGetClassName(A)+" provided.").c_str());
	};
*/

	//data conversion
	template<typename type, int rank> class mextype{
	public:
		typedef dynamic::DataSet<type,rank,false> DataSet;
		typedef dynamic::fixed_array1<exttype::fixed_vect<type> > varray2;
	};

	class mx_struct{
	protected:
		const mxArray * A;
	public:
		mx_struct(const mxArray * _A):A(_A){
			if(!mxIsStruct(A)){
				throw debug_exception("Struct expected");
			};
		};
		double get(const char * name){
			mxArray * a = mxGetField(A,0, name);
			if(a==0){
				throw debug_exception(std::string("No field ")+name+ " in the struct");
			};
			return *(double*)mxGetData(a);
		};
	};

	//! Read usual n-dimensional array into DataSet
	template<typename type, int rank> typename mextype<type,rank>::DataSet dataset(const mxArray *A){
		check_type<type>(A);
		const int ndims = mxGetNumberOfDimensions(A);
		//if size in some dimensions is one, the number of dimensions for matrix is 1 
		//const size_t * dims = mxGetDimensions(A);
		const int * dims = mxGetDimensions(A);
		intn<rank> size;
		int i0 = 0;
		//handle row vectors:
		if(ndims==2 && rank==1 && dims[i0]==1)++i0;
		int notempty = 1;
		for(int i=0;i<std::min(rank,ndims-i0);++i){
			size[i] = dims[i+i0];
			if(size[i]==0)notempty=0;
		};
		//set missing dimensions to 1 if not empty
		for(int i=ndims;i<rank;++i){
			size[i] = notempty;
		};
		return typename mextype<type,rank>::DataSet(size,(type*)mxGetData(A));
	};

	//! Read sparse bool matrix into varray2
	mextype<int,2>::varray2 varray2_bool(const mxArray *A);

	//! Read sparse double matrix into varray2
	mextype<double,2>::varray2 varray2_double(const mxArray *A);

	//! Write sparse double matrix from varray2
	mxArray * write_sp_varray2(mxArray * dest, mextype<double,2>::varray2 & src);


	template<typename type, int rank> typename mextype<type,rank>::DataSet create_dataset(const intn<rank> & size, mxArray *& A){
		//size_tn<rank> sz;
		//sz<<size;
		A = mxCreateNumericArray(rank,size.begin(), mexClassId<type>(), mxREAL);
		return typename mextype<type,rank>::DataSet(size,(type*)mxGetData(A));
	};

	template<typename type> const type& mexVal(const mxArray *A){
		check_type<type>(A);
		const int ndims = mxGetNumberOfDimensions(A);
		const int n = (int)mxGetNumberOfElements(A);
		if(n!=1)throw debug_exception(String("One value expected,") + String(n)+" provided.");
		return ((type*)mxGetData(A))[0];
	};
};

#endif