#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>

#include "opengm/inference/partialOptimality/popt_inference_base.hxx"
#include "opengm/inference/partialOptimality/popt_data.hxx"

#include <string>
#include "debug/logs.h"
#include "part_opt_opengm.h"
#include "files/xfs.h"

#include "optim/part_opt/energy.h"

using namespace opengm;

template<typename type>
energy_auto<type> * opengm_read(std::string modelFilename, std::string dataset){//, energy_auto<type> & E){
	typedef type ValueType;
	typedef opengm::meta::TypeListGenerator
		<
		opengm::PottsFunction<ValueType>,
		opengm::PottsNFunction<ValueType>,
		opengm::ExplicitFunction<ValueType>,
		opengm::TruncatedSquaredDifferenceFunction<ValueType>,
		opengm::TruncatedAbsoluteDifferenceFunction<ValueType>
		> ::type FunctionTypeList;


	typedef opengm::GraphicalModel<ValueType, opengm::Adder, FunctionTypeList>  GmType;
	typedef opengm::POpt_Data<GmType> POpt_DataType;

	GmType * gm = new GmType();
	debug::stream << "Loading model " << modelFilename << ": " << dataset << "\n";
	opengm::hdf5::load(*gm, modelFilename, dataset);
	debug::stream << "Loading model done\n";

	typedef POpt_Data<GmType> DATA;
	typedef opengm::Minimizer ACC;
	DATA data(*gm);
	// invoke my solver
	part_opt_opengm<DATA, opengm::Adder> alg(data);
	alg.instance_name = modelFilename;
	energy_auto<type> * r = alg.energy;
	alg.energy = 0;
	return r;
};

//
//#include "dynamic/num_array.h"
//
//#include <opengm/graphicalmodel/graphicalmodel.hxx>
//#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
//#include <opengm/functions/potts.hxx>
//#include <opengm/operations/adder.hxx>
//#include <opengm/inference/messagepassing/messagepassing.hxx>
//#include <opengm/inference/gibbs.hxx>
//#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
//#include <fstream>
//#include <string>
//
////#include "mex/mex_io.h"
//#include "../src/interfaces/matlab/opengm/mex-src/model/matlabModelType.hxx"
//
//#include "opengm_read.h"
//#include "debug/performance.h"
//
//using namespace std; // 'using' is used only in example code
//using namespace opengm;
//using namespace exttype;
//
//template<typename type>
//void opengm_read(std::string modelFilename, std::string dataset, energy_auto<type> & E){
//	typedef float type;
//	// load model
//	typedef opengm::interface::MatlabModelType::GmType GmType;
//	GmType * gm = new GmType();
//	debug::stream << "Loading model " << modelFilename<<"\n";
//	opengm::hdf5::load(*gm, modelFilename, dataset);
//	debug::stream << "Loading model done\n";
//	
//	int nV = gm->numberOfVariables();
//	int nF = gm->numberOfFactors();
//
//	//number of labels for each variable: number of variables records in total
//	//mx_array<int,1> nL(nV);
//	E.G._nV = nV;
//	E.K.resize(nV);
//	for(int i=0;i<nV;++i){
//		E.K[i] = (int)gm->numberOfLabels(i);
//	};
//	E.maxK = E.K.max().first;
//	debug::stream << "model nV = " << nV << ", nF =" << nF - nV << ", maxL =" << E.maxK << "\n";
//	//mx_array<int,2> f1(mint2(maxL,nV));
//	E.set_nV(nV);
//	//double mult = 1e6;
//	double mult;
//	//E.f1.resize(nV);
//	dynamic::num_array<double, 1> _f1;
//	_f1.reserve(E.maxK);
//	dynamic::num_array<double, 2> _f2;
//	_f2.reserve(mint2(E.maxK,E.maxK));
//	int nE = 0;
//	for(int loop=0;loop<2;++loop){
//		int e = 0;
//		double maxf = -INF(type);
//		double delta = INF(type);
//		for(int factorId=0;factorId<nF;++factorId){
//			int o = (*gm)[factorId].numberOfVariables();
//			if(o==1){
//				int v = (*gm)[factorId].variableIndex(0);
//				_f1.resize(E.K[v]);
//				(*gm)[factorId].copyValues(_f1.begin());
//				if(!loop){
//					// add vertex to graph
//					// calculate multiplier
//					maxf = std::max(maxf, _f1.maxabs().first);
//					delta = std::min(delta, _f1.second_min().first - _f1.min().first);
//				}else{
//					E.f1[v].resize(E.K[v]);		
//					/*
//					for (int k = 0; k < E.K[v]; ++k){
//						//assert(abs(_f1(k)) < 100);
//						//_f1[k] = floor(_f1[k]*mult);
//						_f1[k] = _f1[k] * mult;
//					};
//					*/
//					//_f1 *= mult;
//					E.f1[v] << _f1;
//				};
//			}else if (o==2){
//				int u = (*gm)[factorId].variableIndex(0);
//				int v = (*gm)[factorId].variableIndex(1);
//				mint2 sz(E.K[u], E.K[v]);
//				_f2.resize(sz);
//				(*gm)[factorId].copyValues(_f2.begin());
//				if(!loop){
//					// count # edges
//					++nE;
//					// calculate multiplier
//					maxf = std::max(maxf, _f2.maxabs().first);
//					delta = std::min(delta, _f2.second_min().first - _f2.min().first);
//				}else{
//					E.G.E[e][0] = u;
//					E.G.E[e][1] = v;
//					// create new pairwise term
//					// energy_full::t_f2 & f2 = E.f2(e);
//					// f2.resize(sz);
//					/*
//					for (int k1 = 0; k1 < sz[0]; ++k1){
//						for (int k2 = 0; k2 < sz[1]; ++k2){
//							//_f2(k1, k2) = floor(_f2(k1, k2)*mult);
//							_f2(k1, k2) = _f2(k1, k2)*mult;
//						};
//					};
//					*/
//					//_f2*=mult;
//					E.set_f2(e, _f2);
//				};
//				e++;
//			}else{
//				char s[1024];
//				sprintf(s,"Only factor size 1 and 2 accepted, not %i -- ignored\n",o);
//				debug::stream<<s;
//			};
//		};
//		if(!loop){ // initialize
//			E.set_nE(nE);
//			// rescale everything to have delta equal to 1, but only a factor of 10 scaling
//			mult = 1/pow(10, floor(log10(delta)))*10;
//			mult = std::max(1.0,std::min(mult,1/pow(10,floor(log10(maxf)))*1e6)); // take first 6 decimal digits
//			debug::stream << "Selecting multiplier: " << mult << "\n";
//			//E.G.E.resize(nE);
//			//E.G.edge_index();
//			//E.f2.resize(nE);
//			E.tolerance = 1.0 / mult;
//		};
//	};
//	E.G.edge_index();
//	//debug::stream << "Recognized models: " << E.npotts << "Potts terms, " << E.nfull << " full terms\n";
//	E.report();
//	debug::PerformanceCounter c1;
//	delete gm;
//	debug::stream << "GM destructor: " << c1.time() << "s.\n";
//}
//

template energy_auto<d_type>* opengm_read(std::string, std::string); // , energy_auto<type> &);
////template void opengm_read(std::string, std::string, energy_auto<double> &)
