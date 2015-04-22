#pragma once
#ifndef OPENGM_COMBILP_DEFAULT_HXX
#define OPENGM_COMBILP_DEFAULT_HXX

#include "opengm/inference/combilp.hxx"
#include "opengm/inference/lpcplex.hxx"
#include "opengm/inference/trws/trws_adsal.hxx"
#include "opengm/inference/trws/trws_trws.hxx"

namespace opengm {

template<class GM, class ACC>
struct CombiLP_TRWSi_Gen {
	typedef TRWSi<GM, ACC> LPSolverType;
	typedef typename combilp_base::CombiLPReparametrizerTypeGenerator<typename LPSolverType::ReparametrizerType>::ReparametrizedGMType ReparametrizedGMType;
	typedef LPCplex<ReparametrizedGMType, ACC> ILPSolverType;
	typedef CombiLP<GM, ACC, LPSolverType, ILPSolverType> CombiLPType;
};

template<class GM, class ACC>
struct CombiLP_ADSal_Gen {
	typedef ADSal<GM, ACC> LPSolverType;
	typedef typename combilp_base::CombiLPReparametrizerTypeGenerator<typename LPSolverType::ReparametrizerType>::ReparametrizedGMType ReparametrizedGMType;
	typedef LPCplex<ReparametrizedGMType, ACC> ILPSolverType;
	typedef CombiLP<GM, ACC, LPSolverType, ILPSolverType> CombiLPType;
};

} // namespace opengm

#endif
