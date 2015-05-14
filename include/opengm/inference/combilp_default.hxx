#pragma once
#ifndef OPENGM_COMBILP_DEFAULT_HXX
#define OPENGM_COMBILP_DEFAULT_HXX

#include "opengm/inference/combilp.hxx"
#include "opengm/inference/external/toulbar2.hxx"
#include "opengm/inference/labelcollapse/labelcollapse.hxx"
#include "opengm/inference/trws/trws_adsal.hxx"
#include "opengm/inference/trws/trws_trws.hxx"

namespace opengm {

template<class GM, class ACC>
struct CombiLP_TRWSi_Gen {
	typedef TRWSi<GM, ACC> LPSolverType;
	typedef typename combilp_base::CombiLPReparametrizerTypeGenerator<typename LPSolverType::ReparametrizerType>::ReparametrizedGMType ReparametrizedGMType;
	typedef ToulBar2<ReparametrizedGMType, ACC> ILPSolverType;
	typedef CombiLP<GM, ACC, LPSolverType, ILPSolverType> CombiLPType;
};

template<class GM, class ACC>
struct CombiLP_ADSal_Gen {
	typedef ADSal<GM, ACC> LPSolverType;
	typedef typename combilp_base::CombiLPReparametrizerTypeGenerator<typename LPSolverType::ReparametrizerType>::ReparametrizedGMType ReparametrizedGMType;
	typedef ToulBar2<ReparametrizedGMType, ACC> ILPSolverType;
	typedef CombiLP<GM, ACC, LPSolverType, ILPSolverType> CombiLPType;
};

template<class GM, class ACC>
struct CombiLP_TRWSi_LC_Gen {
	typedef TRWSi<GM, ACC> LPSolverType;
	typedef typename combilp_base::CombiLPReparametrizerTypeGenerator<typename LPSolverType::ReparametrizerType>::ReparametrizedGMType ReparametrizedGMType;
	typedef typename LabelCollapseAuxTypeGen<ReparametrizedGMType>::GraphicalModelType AuxMType;
	typedef LabelCollapse<ReparametrizedGMType, ToulBar2<AuxMType, ACC> > ILPSolverType;
	typedef CombiLP<GM, ACC, LPSolverType, ILPSolverType> CombiLPType;
};

} // namespace opengm

#endif
