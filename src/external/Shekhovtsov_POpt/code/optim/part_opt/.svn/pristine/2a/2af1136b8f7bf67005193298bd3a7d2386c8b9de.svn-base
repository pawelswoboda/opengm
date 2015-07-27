#pragma once
#ifndef PART_OPT_OPENGM
#define PART_OPT_OPENGM

#define _CRT_SECURE_NO_WARNINGS

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/visitors/visitors.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/view.hxx>

#include "opengm/inference/inference.hxx"
#include "opengm/inference/partialOptimality/popt_inference_base.hxx"
#include "opengm/inference/partialOptimality/popt_data.hxx"

#include "optim/part_opt/part_opt_interface.h"

namespace opengm {
	template<class DATA, class ACC>
	class part_opt_opengm : public POpt_Inference < DATA, ACC >, public part_opt_interface<typename DATA::GmType::ValueType>
	{
	public:
		typedef ACC AccumulationType;
		typedef typename DATA::GmType GmType;
		typedef typename DATA::GmType GraphicalModelType;
		OPENGM_GM_TYPE_TYPEDEFS;
		typedef ValueType GraphValueType;

		enum MethodType { IRI_TRWS1, DEE1 };

		struct Parameter {
			Parameter(MethodType m = IRI_TRWS1) {
				method_ = m;
			}
			MethodType method_;
		};


		part_opt_opengm(DATA&, const Parameter& param_ = Parameter());

		std::string name() const {
			return "IRI_TRWS1";
		};
		const GraphicalModelType& graphicalModel() const { return gm_; };

		InferenceTermination infer();

	private:
		DATA&   data_;
		const GmType& gm_;
	};

	template<class DATA, class ACC>
	part_opt_opengm<DATA, ACC>::part_opt_opengm
		(DATA& data, const Parameter& param) : data_(data), gm_(data_.graphicalModel())
	{
		const GmType * gm = &gm_;
		// construct solver and copy
		this->nV = gm->numberOfVariables();
		int nF = gm->numberOfFactors();
		this->maxK = 0;
		for (IndexType v = 0; v < this->nV; ++v){
			int K = gm_.numberOfLabels(v);
			this->maxK = std::max(this->maxK, K);
		};
		debug::stream << "model nV = " << this->nV << ", nF = " << nF << ", maxK = " << this->maxK << "\n";
		std::vector<double> _f1;
		_f1.reserve(this->maxK);
		std::vector<double> _f2;
		_f2.reserve(this->maxK*this->maxK);
		//
		this->energy_read_start(this->nV);
		for (int loop = 0; loop < 2; ++loop){
			for (int factorId = 0; factorId < nF; ++factorId){
				int o = (*gm)[factorId].numberOfVariables();
				if (o == 1){
					int v = (*gm)[factorId].variableIndex(0);
					int K = gm->numberOfLabels(v);
					if (loop == 0){
						this->energy_read_vertex(v, K);
					} else{
						_f1.resize(K);
						(*gm)[factorId].copyValues(_f1.begin());
						this->energy_read_f1(v, K, &_f1[0]);
					};
				} else if (o == 2){
					int u = (*gm)[factorId].variableIndex(0);
					int v = (*gm)[factorId].variableIndex(1);
					int Ku = gm->numberOfLabels(u);
					int Kv = gm->numberOfLabels(v);
					if (loop == 0){
						this->energy_read_edge(u, v);
					} else{
						_f2.resize(Ku*Kv);
						(*gm)[factorId].copyValues(_f2.begin());
						this->energy_read_f2(u, v, Ku, Kv, &_f2[0]);
					};
				} else{
					char s[1024];
					sprintf(s, "Only factor size 1 and 2 accepted, not %i -- ignored\n", o);
					std::cout << s;
				};
			};
			if (loop==0){ // initialize
				this->energy_init0();
			};
		};
		this->energy_init1();
	};
	
	template<class DATA, class ACC>
	InferenceTermination part_opt_opengm<DATA, ACC>::infer()
	{
		this->alg_run();
		for (IndexType v = 0; v < gm_.numberOfVariables(); ++v){

			int K = gm_.numberOfLabels(v);
			for (int k = 0; k < K; ++k){
				bool var_alive = this->is_alive(v, k);
				if (!var_alive){
					data_.setFalse(v,k);
				};
			};
		};
		return NORMAL;
	}
} // end namespace opengm

#endif // POPT_IRI_TRWS_HXX
