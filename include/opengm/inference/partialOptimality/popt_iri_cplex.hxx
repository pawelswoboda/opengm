#pragma once
#ifndef POPT_IRI_CPLEX_HXX
#define POPT_IRI_CPLEX_HXX

#include "popt_iterative_solver_base.hxx"
#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/visitors/visitors.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/view.hxx>
#include "popt_data.hxx"
#include "popt_inference_base.hxx"

#include <opengm/inference/lpcplex.hxx>


namespace opengm {

template<class GM, class ACC> 
class POpt_IRI_CPLEX : public POpt_IRI_SolverBase<GM, ACC>, public LPCplex<GM,ACC>
{  
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;

   typedef LPCplex<GM,ACC> SolverType;
   
   typedef visitors::VerboseVisitor<SolverType> VerboseVisitorType;
   typedef visitors::EmptyVisitor<SolverType>   EmptyVisitorType;
   typedef visitors::TimingVisitor<SolverType>  TimingVisitorType;

   POpt_IRI_CPLEX(const GM& gm);
   virtual std::string name() const {return "POpt_CPLEX";}
   const GraphicalModelType& graphicalModel() const {return gm_;};
   //InferenceTermination infer() { EmptyVisitorType visitor; return infer(visitor);}
   //template<class VISITOR>
   //   InferenceTermination infer(VISITOR &);
   InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const; 
   //virtual ValueType value() const; // implement own value function based on custom arg

   bool IsGloballyOptimalSolution();
   bool IncreaseImmovableLabels(
      std::vector<std::vector<bool> >& immovable, 
      const std::vector<IndexType>& l);

private:
   const GM& gm_;
};

template<class GM, class ACC>
POpt_IRI_CPLEX<GM,ACC>::POpt_IRI_CPLEX(
      const GM& gm)
   : gm_(gm), 
   SolverType(gm) //,SolverType::Parameter().integerConstraint_(false))
{
}

template<class GM, class ACC>
bool
POpt_IRI_CPLEX<GM,ACC>::IsGloballyOptimalSolution()
{
   // get marginal solution and test for integrality
   for(size_t i=0; i<gm_.numberOfVariables(); i++) {
      IndependentFactorType indFac;
      SolverType::variable( i, indFac );
      OPENGM_ASSERT( indFac.numberOfVariables() == 1 );
      for(size_t x_i=0; x_i<indFac.numberOfLabels( 0 ); x_i++)
         if(indFac(x_i)>opengm::IRI::IRI<GM,ACC,POpt_IRI_TRWS>::eps_ && indFac(x_i)<1-opengm::IRI::IRI<GM,ACC,POpt_IRI_TRWS>::eps_)
            return false;
   }
   return true;
}

template<class GM, class ACC>
bool 
POpt_IRI_CPLEX<GM,ACC>::IncreaseImmovableLabels(
      std::vector<std::vector<bool> >& immovable, 
      const std::vector<IndexType>& l)
{
   // label becomes immovable if it has positive marginal
   for(size_t v=0; v<gm_.numberOfVariables(); v++) {
      IndependentFactorType indFac;
      SolverType::variable(v, indFac);
      OPENGM_ASSERT( indFac.numberOfVariables() == 1 );
      for(size_t i=0; i<indFac.numberOfLabels( 0 ); i++)
         if(indFac(i) > opengm::IRI::IRI<GM,ACC,POpt_IRI_TRWS>::eps_)
            immovable[v][i] = true;
   }
}

// attention: It seems to me that CPLEX rounds solutions. This may not occur in iterative relaxed inference for otherwise the result may not be partially optimal
template<class GM, class ACC>
InferenceTermination 
POpt_IRI_CPLEX<GM,ACC>::arg(std::vector<LabelType>& l, const size_t T) const
{
   OPENGM_ASSERT(T == 1);

   l.resize( gm_.numberOfVariables() );
	for(size_t i = 0; i < gm_.numberOfVariables(); ++i) {
		IndependentFactorType indFac;
      SolverType::variable( i, indFac );
		OPENGM_ASSERT( indFac.numberOfVariables() == 1 );
      // find maximum entry
      ValueType max_marg = 0.0;
		for(size_t x_i = 0; x_i < indFac.numberOfLabels( 0 ); ++x_i) {
         if(max_marg < indFac(i)) {
            max_marg = indFac(i);
            l[i] = x_i;
         }
		}
	}
   return NORMAL;
}


} // end namespace opengm

#endif // POPT_IRI_CPLEX_HXX

