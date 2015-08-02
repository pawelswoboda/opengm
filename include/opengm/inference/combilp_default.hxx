#pragma once
#ifndef OPENGM_COMBILP_DEFAULT_HXX
#define OPENGM_COMBILP_DEFAULT_HXX

#include "opengm/inference/combilp.hxx"
#include "opengm/inference/labelcollapse.hxx"
#include "opengm/inference/lpcplex.hxx"
#include "opengm/inference/trws/trws_adsal.hxx"
#include "opengm/inference/trws/trws_trws.hxx"

namespace opengm {

template<class GM, class ACC>
struct CombiLP_ADSal_Gen {
	typedef ADSal<GM, ACC> LPSolverType;
	typedef LPCplex<typename CombiLP_ILP_TypeGen<LPSolverType>::GraphicalModelType, ACC> ILPSolverType;
	typedef CombiLP<GM, ACC, LPSolverType, ILPSolverType> CombiLPType;
};

template<class GM, class ACC>
struct CombiLP_TRWSi_Gen {
	typedef TRWSi<GM, ACC> LPSolverType;
	typedef LPCplex<typename CombiLP_ILP_TypeGen<LPSolverType>::GraphicalModelType, ACC> ILPSolverType;
	typedef CombiLP<GM, ACC, LPSolverType, ILPSolverType> CombiLPType;
};

template<class GM, class ACC, labelcollapse::ReparametrizationKind REPA = labelcollapse::ReparametrizationNone>
struct CombiLP_TRWSi_LC_Gen {
	typedef TRWSi<GM, ACC> LPSolverType;
	typedef typename CombiLP_ILP_TypeGen<LPSolverType>::GraphicalModelType ReparametrizedGMType;
	typedef typename LabelCollapseAuxTypeGen<ReparametrizedGMType, ACC>::GraphicalModelType AuxMType;
	typedef LabelCollapse<ReparametrizedGMType, LPCplex<AuxMType, ACC>, REPA> ILPSolverType;
	typedef CombiLP<GM, ACC, LPSolverType, ILPSolverType> CombiLPType;
};

} // namespace opengm

#endif
