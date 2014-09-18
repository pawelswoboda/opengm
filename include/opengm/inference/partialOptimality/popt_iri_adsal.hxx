#pragma once
#ifndef POPT_IRI_ADSal_HXX
#define POPT_IRI_ADSal_HXX

#include "popt_iterative_solver_base.hxx"
#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/visitors/visitors.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/view.hxx>
#include "popt_data.hxx"
#include "popt_inference_base.hxx"

#include <opengm/inference/trws/trws_base.hxx>
#include "opengm/inference/trws/trws_adsal.hxx"
#include <opengm/inference/trws/trws_reparametrization.hxx>
#include <opengm/inference/auxiliary/lp_reparametrization.hxx>

namespace opengm {

// class for initializing parameters for ADSal, (workaround for initializing ADSal::Parameter before ADSal in POpt_IRI_ADSal)
template<class GM, class ACC>
class POpt_IRI_ADSal_Init
{
public:
   POpt_IRI_ADSal_Init() : param_(typename ADSal<GM,ACC>::Parameter(1500)) { 
      param_.verbose_ = false; 
      param_.smoothingDecayMultiplier() = 0.05;
   }
   typename ADSal<GM,ACC>::Parameter param_;
};

template<class GM, class ACC> 
class POpt_IRI_ADSal : protected POpt_IRI_ADSal_Init<GM,ACC>, public POpt_IRI_SolverBase<GM, ACC>, public ADSal<GM,ACC>
{  
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;

   typedef ADSal<GM,ACC> SolverType;
   typedef void* WarmStartParamType; // no warm start implemented
   //typedef typename SolverType::DDVectorType WarmStartParamType; // reparametrization stored for subsequent speedup
  
   typedef typename ADSal<GM,ACC>::Parameter ParamType;
   typedef LPReparametrisationStorage<GM> RepaStorageType;
   typedef GraphicalModel<ValueType,opengm::Adder,opengm::ReparametrizationView<GM,RepaStorageType>,
           opengm::DiscreteSpace<IndexType,LabelType> > ReparametrizedGMType;

   typedef visitors::VerboseVisitor<ADSal<GM,ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<ADSal<GM,ACC> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<ADSal<GM,ACC> >  TimingVisitorType;

   POpt_IRI_ADSal(const GM& gm);
   virtual std::string name() const {return "POpt_ADSal";}
   const GraphicalModelType& graphicalModel() const {return gm_;};

   void consistent(std::vector<bool>& c);
   size_t IncreaseImmovableLabels(
      std::vector<std::vector<bool> >& immovable, 
      const std::vector<IndexType>& l);

   //void GetWarmStartParam(WarmStartParamType& w) { SolverType::getDDVector(&w); };
   //void SetWarmStartParam(WarmStartParamType& w) { SolverType::addDDVector(w) ; };
   void GetWarmStartParam(WarmStartParamType& w) {};
   void SetWarmStartParam(WarmStartParamType& w) {};

private:
   const GM& gm_;
   ParamType param_;
};

template<class GM, class ACC>
POpt_IRI_ADSal<GM,ACC>::POpt_IRI_ADSal(
      const GM& gm)
   : gm_(gm),
   POpt_IRI_ADSal_Init<GM,ACC>(),
   ADSal<GM,ACC>(gm,POpt_IRI_ADSal_Init<GM,ACC>::param_)
{
   OPENGM_ASSERT(gm_.factorOrder() <= 2); 
}

// indicate which variables have arc-consistent solutions
template<class GM, class ACC>
void
POpt_IRI_ADSal<GM,ACC>::consistent(std::vector<bool>& c)
{
   ADSal<GM,ACC>::getTreeAgreement(c);
}

template<class GM, class ACC>
size_t 
POpt_IRI_ADSal<GM,ACC>::IncreaseImmovableLabels(
      std::vector<std::vector<bool> >& immovable, 
      const std::vector<IndexType>& l) 
{
   size_t newImmovable = 0;
   // get reparametrized model and add minimal reparametrized labels for each potential
   ReparametrizedGMType repGmSolved;
   typename ADSal<GM,ACC>::ReparametrizerType* prepa = ADSal<GM,ACC>::getReparametrizer();
   //prepa->reparametrize(immovable); // reparametrize such that immovable labels have unaries == 0
   prepa->reparametrize(); 
   prepa->getReparametrizedModel(repGmSolved);
   OPENGM_ASSERT(graphicalModel().numberOfVariables() == repGmSolved.numberOfVariables());
   OPENGM_ASSERT(graphicalModel().numberOfFactors() == repGmSolved.numberOfFactors());

   // inspect reparametrized unary labels
   for(size_t v=0; v<gm_.numberOfVariables(); v++) {
      std::vector<ValueType> repPotential;
      opengm::IRI::GetUnaryFactor<ReparametrizedGMType>(repGmSolved,v,repPotential);
      ValueType minLabelCost = std::numeric_limits<ValueType>::max();
      for(size_t i=0; i<immovable[i].size(); i++) {
         minLabelCost = std::min(minLabelCost,repPotential[i]);
         //if(immovable[v][i]) { // only valid for reparametrizations with immovables
         //   OPENGM_ASSERT(repPotential[i] > (-1)*eps_);
         //}
      }
      for(size_t i=0; i<immovable[i].size(); i++) {
         if(repPotential[i] <= minLabelCost + opengm::IRI::IRI<GM,ACC,POpt_IRI_ADSal>::eps_) {
            if(immovable[v][i] == false) {
               immovable[v][i] = true;
               newImmovable++;
            }
         }
      }
   }

   delete prepa;
   return newImmovable;
}

} // end namespace opengm

#endif // POPT_IRI_ADSal_HXX

