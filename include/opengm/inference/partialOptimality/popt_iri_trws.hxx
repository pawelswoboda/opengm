#pragma once
#ifndef POPT_IRI_TRWS_HXX
#define POPT_IRI_TRWS_HXX

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
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/inference/trws/trws_reparametrization.hxx>
#include <opengm/inference/auxiliary/lp_reparametrization.hxx>


namespace opengm {

template<class GM, class ACC> 
class POpt_IRI_TRWS : public POpt_IRI_SolverBase<GM, ACC>, public TRWSi<GM,ACC>
{  
public:
   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;

   typedef TRWSi<GM,ACC> SolverType;
   typedef typename SolverType::DDVectorType WarmStartParamType; // reparametrization stored for subsequent speedup
  
   typedef TRWSi_Parameter<GM> ParamType;
   typedef LPReparametrisationStorage<GM> RepaStorageType;
   typedef GraphicalModel<ValueType,opengm::Adder,opengm::ReparametrizationView<GM,RepaStorageType>,
           opengm::DiscreteSpace<IndexType,LabelType> > ReparametrizedGMType;

   typedef visitors::VerboseVisitor<TRWSi<GM,ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<TRWSi<GM,ACC> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<TRWSi<GM,ACC> >  TimingVisitorType;

   POpt_IRI_TRWS(const GM& gm);
   virtual std::string name() const {return "POpt_TRWSi";}
   const GraphicalModelType& graphicalModel() const {return gm_;};
   //InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
   //virtual ValueType value() const; 

   bool IsGloballyOptimalSolution();
   bool IncreaseImmovableLabels(
      std::vector<std::vector<bool> >& immovable, 
      const std::vector<IndexType>& l);

   void GetWarmStartParam(WarmStartParamType& w) { SolverType::getDDVector(&w); };
   void SetWarmStartParam(WarmStartParamType& w) { SolverType::addDDVector(w);  };

private:
   const GM& gm_;
   TRWSi_Parameter<GM> param_; 
};

template<class GM, class ACC>
POpt_IRI_TRWS<GM,ACC>::POpt_IRI_TRWS(
      const GM& gm)
   : gm_(gm),
   //param_::verbose_(true),
   //param_.setTreeAgreeMaxStableIter(100),
   TRWSi<GM,ACC>(gm,TRWSi_Parameter<GM>(1500)) //.setTreeAgreeMaxStableIter(100)) // verbose(true)
{
   OPENGM_ASSERT(gm_.factorOrder() <= 2); 
   //TRWSi<GM,ACC>::_parameters.setTreeAgreeMaxStableIter(100);
   //TRWSi<GM,ACC>::verbose_=true;
}

template<class GM, class ACC>
bool
POpt_IRI_TRWS<GM,ACC>::IsGloballyOptimalSolution()
{
   std::vector<bool> treeAgreement;
   TRWSi<GM,ACC>::getTreeAgreement(treeAgreement);
   bool globalOptimal = true;
   for(size_t v=0; v<treeAgreement.size(); v++)
      if(!treeAgreement[v])
         globalOptimal = false;
   if(globalOptimal) { 
      std::cout << "Model LP-tight" << std::endl;
      return true;
   } else {
      return false;
   }
}

template<class GM, class ACC>
bool 
POpt_IRI_TRWS<GM,ACC>::IncreaseImmovableLabels(
      std::vector<std::vector<bool> >& immovable, 
      const std::vector<IndexType>& l)
{
   // get reparametrized model and add minimal reparametrized labels for each potential
   ReparametrizedGMType repGmSolved;
   typename TRWSi<GM,ACC>::ReparametrizerType* prepa = TRWSi<GM,ACC>::getReparametrizer();
   //prepa->reparametrize(immovable); // reparametrize such that immovable labels have unaries == 0
   prepa->reparametrize(); // reparametrize such that immovable labels have unaries == 0
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
         if(repPotential[i] <= minLabelCost + opengm::IRI::IRI<GM,ACC,POpt_IRI_TRWS>::eps_) {
            immovable[v][i] = true;
         }
      }
   }

   delete prepa;
}

} // end namespace opengm

#endif // POPT_IRI_TRWS_HXX
