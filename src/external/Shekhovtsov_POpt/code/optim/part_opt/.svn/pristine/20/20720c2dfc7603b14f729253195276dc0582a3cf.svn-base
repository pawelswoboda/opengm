#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <fstream>
#include <string>

#include "mex/mex_io.h"
#include "dynamic/num_array.h"
#include "../src/interfaces/matlab/opengm/mex-src/model/matlabModelType.hxx"
#include "mex/mexargs.h"

using namespace std; // 'using' is used only in example code
using namespace opengm;
using namespace exttype;
using namespace dynamic;

void mexFunction_protect(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	//check if data is in correct format
	if(nrhs != 2) {
		mexErrMsgTxt("Usage: [E f1 f2] = opengm_read('file_name','gm')\n");
	}
	// get file name and corresponding dataset
	mx_string modelFilename(prhs[0]);
	mx_string dataset(prhs[1]);
	/*
	std::string modelFilename = mxArrayToString(prhs[0]);
	if(modelFilename.data()==NULL) {
		mexErrMsgTxt("load: could not convert input to string.");
	}
	std::string dataset = mxArrayToString(prhs[1]);
	if(dataset.data()==NULL) {
		mexErrMsgTxt("load: could not convert input to string.");
	}
	*/

	// load model
	typedef double value_type;
	typedef opengm::interface::MatlabModelType::GmType GmType;
	GmType gm;
	mexPrintf("Loading model...\n");
	opengm::hdf5::load(gm, modelFilename, dataset);
	mexPrintf("Loading model done\n");

	int nV = gm.numberOfVariables();
	int nF = gm.numberOfFactors();

	//number of labels for each variable: number of variables records in total
	mx_array<int,1> nL(nV);
	for(int i=0;i<nV;++i){
		nL[i] = (int)gm.numberOfLabels(i);
	};
	int maxL = nL.max().first;
	mexPrintf("model nV = %i, nF = %i, maxL = %i \n",nV,nF-nV,maxL);
	mx_array<value_type,2> f1(mint2(maxL,nV));
	dynamic::num_array<double, 1> _f1;
	_f1.reserve(maxL);
	mx_array<value_type, 3> f2;
	//mx_array<double, 2> _f2(mint2(maxL, maxL));
	dynamic::num_array<double, 2> _f2;
	_f2.reserve(mint2(maxL,maxL));
	mx_array<int,2> E;
	f1 << (1 << 29);
	f2 << 0;
	double mult = 1; // 1e5;
	int nE = 0;
	for(int loop=0;loop<2;++loop){
		int e = 0;
		for(int factorId=0;factorId<nF;++factorId){
			int o = gm[factorId].numberOfVariables();
			if(o==1){
				int v = gm[factorId].variableIndex(0);
				if(!loop){
					// add vertex to graph
				}else{
					_f1.resize(nL[v]);
					gm[factorId].copyValues(_f1.begin());
					for (int k = 0; k < nL[v]; ++k){
						if (abs(_f1(k)*mult) > double((1 << 29))){
							//mexPrintf("Cost too big\n");
							//return;
							_f1(k) = double((1 << 29)) / mult;
						};
					};
					f1.subdim<1>(v) << _f1 * mult;
					//for (int k = 0; k < nL[v]; ++k){
					//	f1(k,v) = _f1[k]*1e6;
					//};
				};
			}else if (o==2){
				int u = gm[factorId].variableIndex(0);
				int v = gm[factorId].variableIndex(1);
				if(!loop){
					// count # edges
					++nE;
				}else{
					E(0,e) = u;
					E(1,e) = v;
					_f2.resize(mint2(nL[u],nL[v]));
					gm[factorId].copyValues(_f2.begin());
					for (int k1 = 0; k1 < nL[u]; ++k1){
						for (int k2 = 0; k2 < nL[v]; ++k2){
							f2(k1, k2, e) = value_type(std::min(_f2(k1, k2)*mult, double((1 << 29))));
						};
					};
					
					//f2.subdim<2>(e) << _f2*1e6;
				};
				++e;
			}else{
				char s[1024];
				sprintf(s,"Only factor size 1 and 2 accepted, not %i -- ignored\n",o);
				//mexErrMsgTxt(s);
				mexPrintf(s);
			};
		};
		if(!loop){
			mexPrintf("Allocating %i MB for %i edges\n", (maxL*maxL*nE*sizeof(int)/1024/1024),nE);
			E.resize(mint2(2,nE));
			f2.resize(mint3(maxL, maxL, nE));
			mexPrintf("allocated\n");
		};
	};
	mexPrintf("Copying to matlab complete\n");
	if(nlhs>0)plhs[0] = E.get_mxArray_andDie();
	if(nlhs>1)plhs[1] = f1.get_mxArray_andDie();
	if(nlhs>2)plhs[2] = f2.get_mxArray_andDie();
	// create handle to model
	//plhs[0] = opengm::interface::handle<GmType>::createHandle(gm);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	mexargs::MexLogStream log("log/output.txt", false);
	debug::stream.attach(&log);
	mexargs::MexLogStream err("log/errors.log", true);
	debug::errstream.attach(&err);

	//try{
	mexFunction_protect(nlhs, plhs, nrhs, prhs);
	//} catch (std::exception & e){
	//	debug::stream << "!Exception:" << e.what() << "\n";
	//};

	memserver::get_global()->clean_garbage();
	debug::stream.detach();
	debug::errstream.detach();
};

