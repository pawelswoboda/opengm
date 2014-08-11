#pragma once
#ifndef POPT_ITERATIVE_SOLVER_BASE_HXX
#define POPT_ITERATIVE_SOLVER_BASE_HXX

#include "opengm/opengm.hxx"

namespace opengm {

// solver for iteratice_relaxed_inference must implement the additional two functions specified below
// additional requirements:
//    SolverType must be defined
//    WarmStartParamType (=P in Set- and GetWarmStartParam) must be defined
template<class GM, class ACC> 
class POpt_IRI_SolverBase 
{  
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   typedef typename GraphicalModelType::LabelType LabelType;
   typedef typename GraphicalModelType::IndexType IndexType;
   typedef typename GraphicalModelType::ValueType ValueType;
   typedef typename GraphicalModelType::OperatorType OperatorType;
   typedef typename GraphicalModelType::FactorType FactorType;
   typedef typename GraphicalModelType::IndependentFactorType IndependentFactorType;
   typedef typename GraphicalModelType::FunctionIdentifier FunctionIdentifier;

   virtual bool IsGloballyOptimalSolution() = 0;
   virtual bool IncreaseImmovableLabels(
      std::vector<std::vector<bool> >& immovable, 
      const std::vector<IndexType>& l) = 0;
   template<class P> void SetWarmStartParam(P&) {throw;};
   template<class P> void GetWarmStartParam(P&) {throw;};
};

} // end namespace opengm

#endif // POPT_ITERATIVE_SOLVER_BASE_HXX
