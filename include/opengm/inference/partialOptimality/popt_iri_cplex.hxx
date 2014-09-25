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

// class for initializing parameters for CPLEX, (workaround for initializing parameters for CPLEX before LPCplex in POpt_IRI_CPLEX)
template<class GM, class ACC>
class POpt_IRI_CPLEX_Init
{
public:
   POpt_IRI_CPLEX_Init() : param_(typename LPCplex<GM,ACC>::Parameter()) { 
      param_.verbose_ = false; 
      param_.integerConstraint_ = false; 
   }
   typename LPCplex<GM,ACC>::Parameter param_;
};


template<class GM, class ACC> 
class POpt_IRI_CPLEX : protected POpt_IRI_CPLEX_Init<GM,ACC>, public POpt_IRI_SolverBase<GM, ACC>, public LPCplex<GM,ACC>
{  
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   
   typedef LPCplex<GM,ACC> SolverType;
   typedef void* WarmStartParamType; // no warm start implemented

   typedef typename LPCplex<GM,ACC>::Parameter ParamType;


   typedef visitors::VerboseVisitor<SolverType> VerboseVisitorType;
   typedef visitors::EmptyVisitor<SolverType>   EmptyVisitorType;
   typedef visitors::TimingVisitor<SolverType>  TimingVisitorType;

   POpt_IRI_CPLEX(const GM& gm);
   virtual std::string name() const {return "POpt_CPLEX";}
   const GraphicalModelType& graphicalModel() const {return gm_;};
   InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const; 
   ValueType value() const; 

   void consistent(std::vector<bool>& c);
   size_t IncreaseImmovableLabels(
      std::vector<std::vector<bool> >& immovable, 
      const std::vector<IndexType>& l);

   void SetWarmStartParam(WarmStartParamType& w) {};
   void GetWarmStartParam(WarmStartParamType& w) {};

   const static double eps_;
private:
   const GM& gm_;
   ParamType param_;
};

template<class GM,class ACC>
const double POpt_IRI_CPLEX<GM,ACC>::eps_ = 1.0e-6;

template<class GM, class ACC>
POpt_IRI_CPLEX<GM,ACC>::POpt_IRI_CPLEX(
      const GM& gm)
   : gm_(gm), 
   POpt_IRI_CPLEX_Init<GM,ACC>(),
   SolverType(gm,POpt_IRI_CPLEX_Init<GM,ACC>::param_) 
{
}

// indicate which variables have integral solutions
template<class GM, class ACC>
void
POpt_IRI_CPLEX<GM,ACC>::consistent(std::vector<bool>& consistent) 
{
   consistent.resize(gm_.numberOfVariables(),true);
   for(size_t i=0; i<gm_.numberOfVariables(); i++) {
      IndependentFactorType indFac;
      SolverType::variable( i, indFac );
      OPENGM_ASSERT( indFac.numberOfVariables() == 1 );
      for(size_t x_i=0; x_i<indFac.numberOfLabels( 0 ); x_i++) {
         if(indFac(x_i)>eps_ && indFac(x_i)<1.0-eps_) {
            consistent[i] = false;
         }
      }
   }
}

template<class GM, class ACC>
size_t 
POpt_IRI_CPLEX<GM,ACC>::IncreaseImmovableLabels(
      std::vector<std::vector<bool> >& immovable, 
      const std::vector<IndexType>& l) 
{
   size_t newImmovable = 0;
   // label becomes immovable if it has positive marginal
   for(size_t v=0; v<gm_.numberOfVariables(); v++) {
      IndependentFactorType indFac;
      SolverType::variable(v, indFac);
      OPENGM_ASSERT( indFac.numberOfVariables() == 1 );
      for(size_t i=0; i<indFac.numberOfLabels( 0 ); i++)
         if(indFac(i) > eps_)
            if(immovable[v][i] == false) {
               newImmovable++;
               immovable[v][i] = true;
            }
   }
   return newImmovable;
}

// attention: It seems to me that CPLEX rounds solutions. 
// This must not occur in iterative relaxed inference for otherwise the result may not be partially optimal. 
// We need labelings on the support of optimal fractional solutions of the local polytope relaxation
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
      // possibly: evaluate star for all ambiguous labelings and take best one
      ValueType max_marg = 0.0;
      for(size_t x_i=0; x_i<indFac.numberOfLabels( 0 ); ++x_i) {
         if(max_marg < indFac(x_i)) {
            max_marg = indFac(x_i);
            l[i] = x_i;
         }
      }
   }
   return NORMAL;
}

// get value based on own arg function.
template<class GM, class ACC>
typename GM::ValueType
POpt_IRI_CPLEX<GM,ACC>::value() const
{
   std::vector<LabelType> l;
   arg(l);
   gm_.evaluate(l.begin());
}

} // end namespace opengm

#endif // POPT_IRI_CPLEX_HXX

