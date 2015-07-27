#include "mexargs.h"

namespace mexargs{
	//	MexLogStream MexLogStream::log;
/*

	template<typename type> mxClassID mexClassId(){
		throw debug_exception(String("Type is not recognized: "));//+typeid(type).name());
	};

	template<> mxClassID mexClassId<unsigned char>(){
		return mxUINT8_CLASS;
	};

	template<> mxClassID mexClassId<int>(){
		return mxmmint32_CLASS;
	};

	template<> mxClassID mexClassId<long long>(){
		return mxINT64_CLASS;
	};

	template<> mxClassID mexClassId<double>(){
		return mxDOUBLE_CLASS;
	};

	template<> mxClassID mexClassId<float>(){
		return mxSINGLE_CLASS;
	};

	template<> mxClassID mexClassId<bool>(){
		return mxLOGICAL_CLASS;
	};
*/

	//! Read sparse bool matrix into varray2
	mextype<int,2>::varray2 varray2_bool(const mxArray *A){
		check_type<bool>(A);
		if(!mxIsSparse(A)){
			throw debug_exception("Sparse input array is expected.");
		};
		int m  = (int)mxGetM(A);
		int n  = (int)mxGetN(A);
		double *pr = mxGetPr(A);
		int * ir = mxGetIr(A);
		int * jc = mxGetJc(A);

		intn<2> size(m,n);
		mextype<int,2>::varray2 R(size[1]);
		for(int j=0;j<size[1];++j){
			int nj = jc[j+1]-jc[j];
			R[j].resize(nj);
			for(int l = 0;l<nj;++l){
				R[j][l] = ir[jc[j]+l];
			}; 
		};
		return R;
	};

	//! Read sparse double matrix into varray2
	mextype<double,2>::varray2 varray2_double(const mxArray *A){
		check_type<double>(A);
		if(!mxIsSparse(A)){
			throw debug_exception("Sparse input array is expected.");
		};
		int m  = (int)mxGetM(A);
		int n  = (int)mxGetN(A);
		double *pr = mxGetPr(A);
		int * ir = mxGetIr(A);
		int * jc = mxGetJc(A);

		intn<2> size(m,n);
		mextype<double,2>::varray2 R(size[1]);
		for(int j=0;j<size[1];++j){
			int nj = jc[j+1]-jc[j];
			R[j].resize(nj);
			for(int l = 0;l<nj;++l){
				R[j][l] = pr[jc[j]+l];
			}; 
		};
		return R;
	};

	mxArray * write_sp_varray2(mxArray * dest, mextype<double,2>::varray2 & src){
		check_type<double>(dest);
		if(!mxIsSparse(dest)){
			throw debug_exception("Sparse input array is expected.");
		};
		int m  = (int)mxGetM(dest);
		int n  = (int)mxGetN(dest);
		double *pr = mxGetPr(dest);
		int * ir = mxGetIr(dest);
		int * jc = mxGetJc(dest);

		intn<2> size(m,n);
		for(int j=0;j<size[1];++j){
			int nj = jc[j+1]-jc[j];
			//R[j].resize(nj);
			for(int l = 0;l<nj;++l){
				if(src[j][l]==0){ 
					pr[jc[j]+l] = 1e-8;
				}else{
					pr[jc[j]+l] = src[j][l];
				};
			};
		}; 
		return dest;
	};
};