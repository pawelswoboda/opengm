#ifndef OPENGM_IRI_HXX
#define OPENGM_IRI_HXX

//#define OPENGM_IRI_CPLEX
#define OPENGM_IRI_TRWS

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/visitors/visitors.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/view.hxx>
#include "popt_data.hxx"
#include "popt_inference_base.hxx"

#ifdef OPENGM_IRI_TRWS
#include <opengm/inference/trws/trws_base.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/inference/trws/trws_reparametrization.hxx>
#include <opengm/inference/auxiliary/lp_reparametrization.hxx>
#endif

#ifdef OPENGM_IRI_CPLEX
#include <opengm/inference/lpcplex.hxx>
#include <opengm/inference/auxiliary/lp_solver/lp_solver_cplex.hxx>
#endif

#include "persistency_potential_perm.hxx"

// For Pruning Cut
#include <opengm/inference/external/qpbo.hxx>

#include <vector>
#include <limits>
#include <queue>
#include <ctime>

namespace opengm {
namespace IRI {

template<class GM>
void
GetUnaryFactor(
      const GM& gm,
      const size_t i, // unary factor associated to variable i
      std::vector<typename GM::ValueType>& factor
)
{
   factor.resize(gm.numberOfLabels(i));
   marray::Vector<size_t> f;
   OPENGM_ASSERT(gm.numberOfNthOrderFactorsOfVariable(i,1) == 1);
   gm.numberOfNthOrderFactorsOfVariable(i,1,f);
   OPENGM_ASSERT(gm[f(0)].numberOfVariables() == 1);
   OPENGM_ASSERT(i == gm.variableOfFactor(f(0),0));
   gm[f(0)].copyValues(factor.begin()); // do zrobienia: check if values are copied in correct order into factor
}

template<class GM>
void
GetPairwiseFactor(
      const GM& gm,
      const size_t f, 
      marray::Matrix<typename GM::ValueType>& factor
)
{
   size_t v1 = gm.variableOfFactor(f,0);
   size_t v2 = gm.variableOfFactor(f,1);
   factor.resize(gm.numberOfLabels(v1),gm.numberOfLabels(v2));
   gm[f].copyValues(factor.begin()); // do zrobienia: check if values are copied in correct order into factor
}

// help function: evaluate labeling l on star graph around current node i in graphical model gm
template<class GM, class ITERATOR>
typename GM::ValueType
EvaluateStar(const GM& gm, typename GM::IndexType i, ITERATOR l)
{
   typedef typename GM::ValueType ValueType;
   typedef typename GM::IndexType IndexType;
   typedef typename GM::LabelType LabelType;
   typedef typename GM::ConstFactorIterator ConstFactorIterator;
   ValueType val = 0.0;
   for(ConstFactorIterator it = gm.factorsOfVariableBegin(i); it != gm.factorsOfVariableEnd(i); it++) {

      std::vector<LabelType> lFactor(gm[*it].numberOfVariables());
      for(size_t j=0; j<lFactor.size(); j++)
         lFactor[j] = l[gm[*it].variableIndex(j)];
      val += gm[*it](lFactor.begin());
   }
}

// visitor which terminates algorithm if primal bound is smaller than eps
template<class INFERENCE>
class PrimalBoundVisitor{
public:
   PrimalBoundVisitor(double eps, size_t minIter)
   :  eps_(eps),
   nIter_(0),
   minIter_(minIter)
      {}
   void begin(INFERENCE & inf){}
   size_t operator()(INFERENCE & inf){
      typename INFERENCE::ValueType value = inf.value();
      if(value < eps_)
         nIter_++;
      if(value < eps_ && nIter_ > minIter_) {
         std::cout<<"aborting optimization due to primal integer value " << value << " smaller than threshold " << eps_ << std::endl;
         return visitors::VisitorReturnFlag::StopInfBoundReached;
      } else {
         return visitors::VisitorReturnFlag::ContinueInf;
      }
   }
   void end(INFERENCE & inf){
      std::cout<<"value "<< inf.value() <<" bound "<< inf.bound() <<"\n";
   }
private:
   double eps_;
   size_t minIter_;
   size_t nIter_;
};



// Persistency by improving mappings: iterative algorithm
template<class DATA, class ACC> 
class IRI : public POpt_Inference<DATA, ACC>
{
public:
   typedef ACC AccumulationType;
   typedef typename DATA::GraphicalModelType GM;
   typedef typename DATA::GraphicalModelType GmType;
   typedef typename DATA::GraphicalModelType GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef typename GM::ConstFactorIterator ConstFactorIterator;
   typedef typename GM::ConstVariableIterator ConstVariableIterator;

   typedef opengm::IRI::Potential<GM> PersPotentialType;
   //typedef opengm::IRI::GenPotential<GM> PersPotentialType;
   typedef GraphicalModel<ValueType,opengm::Adder,PersPotentialType,
           opengm::DiscreteSpace<IndexType,LabelType> > PersistencyGMType;
   typedef typename PersistencyGMType::ConstFactorIterator PGMConstFactorIterator;
   typedef typename PersistencyGMType::ConstVariableIterator PGMConstVariableIterator;

  

#ifdef OPENGM_IRI_TRWS
   typedef LPReparametrisationStorage<PersistencyGMType> RepaStorageType;
   typedef GraphicalModel<ValueType,opengm::Adder,opengm::ReparametrizationView<PersistencyGMType,RepaStorageType>,
           opengm::DiscreteSpace<IndexType,LabelType> > ReparametrizedGMType;

   typedef TRWSi<GM,ACC> SolverType;
   typedef TRWSi_Parameter<GM> SolverParamType;

   typedef TRWSi<PersistencyGMType,ACC> PersSolverType;
   typedef TRWSi_Parameter<PersistencyGMType> PersSolverParamType;
#endif


#ifdef OPENGM_IRI_CPLEX
   typedef LPCplex<GM, ACC> SolverType;
   typedef LPCplex<PersistencyGMType, ACC> PersSolverType;

   typedef typename LPCplex<DATA,ACC >::Parameter SolverParamType;
   typedef typename LPCplex<PersistencyGMType,ACC >::Parameter PersSolverParamType;
#endif

   IRI(DATA& d);
   virtual std::string name() const {return "IRI";}
   const GraphicalModelType& graphicalModel() const {return gm_;};
   InferenceTermination infer();
   //template<class VISITOR>
   //   InferenceTermination infer(VISITOR &);
   //InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
   
private:
   void InitializeImprovingMap();
   void InitializeImmovable();
   void ImproveLabeling(std::vector<LabelType>& l);
   void ConstructMRF(PersistencyGMType& pgm, const std::vector<std::vector<LabelType> >& im);
   void UpdateMRF(PersistencyGMType& pgm, const std::vector<std::vector<LabelType> >& im);
   void UpdateMRF(PersistencyGMType& pgm, const std::vector<std::vector<LabelType> >& im, size_t v);
   void ConstructSubsetToOneMap(
         std::vector<LabelType> & im,
         const LabelType l,
         const std::vector<bool>& immovable);
   void ConstructSubsetToOneMap(
         std::vector<std::vector<LabelType> >& im,
         const std::vector<LabelType> l,
         const std::vector<std::vector<bool> >& immovable);
   bool AllLabelImmovable(const std::vector<std::vector<bool> >& immovable);
   bool IsProjection(const std::vector<LabelType>& p);
   std::vector<LabelType> PruningCut(
         const std::vector<LabelType>& curLabeling,
         const std::vector<LabelType>& projLabeling,
         const std::vector<std::vector<bool> >& immovable,
         const PersistencyGMType& gm); 
   bool SingleNodePruning(
         const size_t v,
         const size_t i,
         const PersistencyGMType& gm); 
   size_t SingleNodePruning(
         std::vector<std::vector<bool> >& immovable,
         PersistencyGMType& gm); 
   ValueType GetMinimalLabels(const std::vector<ValueType>& p, std::vector<bool>& m);
   ValueType AddPairwiseImmovable(
         const std::vector<std::vector<LabelType> >& p,
         const std::vector<ValueType>& p1,
         const std::vector<ValueType>& p2,
         std::vector<bool>& immovable1,
         std::vector<bool>& immovable2);
   void IncreaseImmovableLabels(
         std::vector<std::vector<bool> >& immovable,
         const std::vector<IndexType>& l,
         PersSolverType& solver,
         PersistencyGMType& pgm);
   bool IsGloballyOptimalSolution(SolverType& solver);
   size_t NoImmovableLabels(const std::vector<std::vector<bool> >& immovable);

   const static double eps_; 
   DATA& d_;
   const GM& gm_;
   const size_t n_; // number of variables
   SolverParamType SolverParam_;
   PersSolverParamType PersSolverParam_;

   // improving mapping
   std::vector<std::vector<LabelType> > im_;
   // initial labeling for constructing subset to one mapping
   std::vector<IndexType> l_;
   // immovable labels
   std::vector<std::vector<bool> > immovable_;

   // time measurements for individual operations
   std::clock_t totalTime, singleNodePruningTime, pruningCutTime, initialInferenceTime, subsequentInferenceTime, MRFModificationTime;
};

template<class DATA,class ACC>
const double IRI<DATA,ACC>::eps_ = 1.0e-6;

template<class DATA,class ACC>
IRI<DATA,ACC>::IRI(DATA& d) :
   d_(d),
   gm_(d.graphicalModel()),
   n_(d.graphicalModel().numberOfVariables())
{
#ifdef OPENGM_IRI_TRWS
   OPENGM_ASSERT(gm_.factorOrder() <= 2); 
#endif
   InitializeImprovingMap();
   InitializeImmovable();

#ifdef OPENGM_IRI_TRWS
   const static size_t TRWSIter = 1500;
   SolverParam_ = SolverParamType(TRWSIter);
   SolverParam_.verbose_=true;
   SolverParam_.precision() = 1e-6; 
   SolverParam_.setTreeAgreeMaxStableIter(100);
   PersSolverParam_ = PersSolverParamType(TRWSIter);
   PersSolverParam_.verbose_=true;
   PersSolverParam_.precision() = 1e-6; 
   PersSolverParam_.setTreeAgreeMaxStableIter(100);
#endif

#ifdef OPENGM_IRI_CPLEX
   SolverParam_.integerConstraint_ = false;
   PersSolverParam_.integerConstraint_ = false;
#endif

   totalTime = 0;
   singleNodePruningTime = 0;
   pruningCutTime= 0;
   initialInferenceTime = 0;
   subsequentInferenceTime = 0;
   MRFModificationTime = 0;
}

template<class DATA, class ACC>
void
IRI<DATA,ACC>::InitializeImprovingMap()
{
   im_.resize(n_);
   for(size_t v=0; v<n_; v++)
      im_[v].resize(gm_.numberOfLabels(v),0);
}

template<class DATA, class ACC>
void
IRI<DATA,ACC>::InitializeImmovable()
{
   immovable_.resize(n_);
   for(size_t i=0; i<n_; i++) 
      immovable_[i].assign(gm_.numberOfLabels(i),false);
}

template<class DATA, class ACC>
void 
IRI<DATA,ACC>::ImproveLabeling(std::vector<LabelType>& l)
{
   // to do: possibly improbe labeling with lazy flipper
   for(IndexType i=0; i<n_; i++) {
      if(d_.getPOpt(i,l[i]) == false) {
         // get best local labeling
         ValueType min_labeling_cost = std::numeric_limits<ValueType>::max();
         LabelType min_label = 0;
         for(LabelType x_i=0; x_i<gm_.numberOfLabels(i); x_i++) {
            if(d_.getPOpt(i,x_i) != false) {
               l[i] = x_i;
               ValueType cur_labeling_cost = EvaluateStar(gm_,i,l.begin());
               if(cur_labeling_cost < min_labeling_cost) {
                  min_labeling_cost = cur_labeling_cost;
                  min_label = x_i;
               }
            }
         }
      }
   }
}

// construct MRF with cost theta - P^\T * theta
template<class DATA, class ACC>
void
IRI<DATA,ACC>::ConstructMRF(PersistencyGMType& pgm, const std::vector<std::vector<LabelType> >& im)
{
   std::clock_t beginTime = clock();

   pgm = PersistencyGMType(gm_.space());
   for(size_t f=0; f<gm_.numberOfFactors(); f++) {
      // get associated variables
      std::vector<IndexType> varFactor(gm_[f].variableIndicesBegin(), gm_[f].variableIndicesEnd());
      std::vector<std::vector<LabelType> > imFactor(varFactor.size());
      // get associated pixelwise improving mappings
      for(size_t v=0; v<varFactor.size(); v++)
         imFactor[v] = im[varFactor[v]];
      // add modified potential: theta - P^\T * theta
      PersPotentialType p(gm_[f],imFactor);
      pgm.addFactor(pgm.addFunction(p), gm_[f].variableIndicesBegin(), gm_[f].variableIndicesEnd());
   }

   std::clock_t endTime = clock();
   MRFModificationTime += endTime - beginTime;
}

// update MRF with cost theta - P^\T * theta, when P has changed
template<class DATA, class ACC>
void
IRI<DATA,ACC>::UpdateMRF(PersistencyGMType& pgm, const std::vector<std::vector<LabelType> >& im)
{
   std::clock_t beginTime = clock();

   for(size_t f=0; f<pgm.numberOfFactors(); f++) {
      std::vector<IndexType> varFactor(pgm[f].variableIndicesBegin(), pgm[f].variableIndicesEnd());
      std::vector<std::vector<LabelType> > imFactor(varFactor.size());
      // get associated pixelwise improving mappings
      for(size_t v=0; v<varFactor.size(); v++)
         imFactor[v] = im[varFactor[v]];
      //size_t functionType = pgm[f].functionType();
      //OPENGM_ASSERT(functionType == 0);
      const_cast<PersPotentialType & >(pgm[f].template function<0>()).UpdateImprovingMapping(imFactor);
   }

   std::clock_t endTime = clock();
   MRFModificationTime += endTime - beginTime;
}

// Update MRF for only one variable
template<class DATA, class ACC>
void
IRI<DATA,ACC>::UpdateMRF(PersistencyGMType& pgm, const std::vector<std::vector<LabelType> >& im, size_t v)
{
   // Update affected factors
   for(size_t fc=0; fc<pgm.numberOfFactors(v); fc++) {
      size_t f = pgm.factorOfVariable(v,fc);
      std::vector<IndexType> varFactor(pgm[f].variableIndicesBegin(), pgm[f].variableIndicesEnd());
      std::vector<std::vector<LabelType> > imFactor(varFactor.size());
      // get associated pixelwise improving mappings
      for(size_t vf=0; vf<varFactor.size(); vf++) {
         imFactor[vf] = im_[varFactor[vf]];
      }
      const_cast<PersPotentialType &>(pgm[f].template function<0>()).UpdateImprovingMapping(imFactor);
   }
}


template<class DATA, class ACC>
void
IRI<DATA,ACC>::ConstructSubsetToOneMap(std::vector<typename GM::LabelType>& im,
                                     const typename GM::LabelType l,
                                     const std::vector<bool>& immovable)
{
   im.resize(immovable.size());
   for(size_t i=0; i<im.size(); i++)
      if(immovable[i] == false) 
         im[i] = l;
      else
         im[i] = i;
     
   OPENGM_ASSERT(IsProjection(im));
}

template<class DATA, class ACC>
void
IRI<DATA,ACC>::ConstructSubsetToOneMap(std::vector<std::vector<typename GM::LabelType> >& im,
                                     const std::vector<typename GM::LabelType> l,
                                     const std::vector<std::vector<bool> >& immovable)
{
   im.resize(n_);
   for(size_t v=0; v<n_; v++) 
      ConstructSubsetToOneMap(im[v],l[v],immovable[v]);
}

template<class DATA, class ACC>
bool 
IRI<DATA,ACC>::IsProjection(const std::vector<typename GM::LabelType>& p)
{
   // permutation property: p[i] < p.size();
   for(size_t i=0; i<p.size(); i++)
      if(p[i] >= p.size())
         return false;

   // involution property: im*im = im
   for(size_t i=0; i<p.size(); i++)
      if(p[p[i]] != p[i])
         return false;

   return true;
}

template<class DATA, class ACC>
bool 
IRI<DATA,ACC>::AllLabelImmovable(const std::vector<std::vector<bool> >& immovable)
{
   OPENGM_ASSERT(n_ == immovable.size());
   for(size_t v=0; v<n_; v++)
      for(size_t l=0; l<immovable[v].size(); l++)
         if(!immovable[v][l])
            return false;
   return true;
}

template<class DATA, class ACC>
size_t 
IRI<DATA,ACC>::NoImmovableLabels(const std::vector<std::vector<bool> >& immovable)
{
   OPENGM_ASSERT(n_ == immovable.size());
   size_t noImmovableLabels = 0;
   for(size_t v=0; v<n_; v++)
      for(size_t l=0; l<immovable[v].size(); l++)
         if(immovable[v][l])
            noImmovableLabels++;
   return noImmovableLabels;
}

template<class DATA, class ACC>
bool 
IRI<DATA,ACC>::IsGloballyOptimalSolution(SolverType& solver)
{
#ifdef OPENGM_IRI_TRWS
      std::vector<bool> treeAgreement;
      solver.getTreeAgreement(treeAgreement);
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
#endif

#ifdef OPENGM_IRI_CPLEX
      // get marginal solution
      for(size_t v=0; v<n_; v++) {
         IndependentFactorType indFac;
         solver.variable( v, indFac );
         OPENGM_ASSERT( indFac.numberOfVariables() == 1 );
         for(size_t i=0; i<indFac.numberOfLabels( 0 ); i++)
            if(indFac(i)>eps_ && indFac(i)<1-eps_)
               return false;
      }
      return true;
#endif
}

template<class DATA, class ACC>
typename DATA::GraphicalModelType::ValueType
IRI<DATA,ACC>::GetMinimalLabels(const std::vector<ValueType>& p, std::vector<bool>& m)
{
   m.assign(p.size(),false);
   ValueType minPotential = *std::min_element(p.begin(),p.end());
   for(size_t i=0; i<p.size(); i++)
      if(p[i] <= minPotential + eps_)
         m[i] = true;
   return minPotential;
}

template<class DATA, class ACC>
typename DATA::GraphicalModelType::ValueType
IRI<DATA,ACC>::AddPairwiseImmovable(
      const std::vector<std::vector<LabelType> >& p,
      const std::vector<ValueType>& p1,
      const std::vector<ValueType>& p2,
      std::vector<bool>& immovable1,
      std::vector<bool>& immovable2)
{
   std::vector<bool> unary1Minimal, unary2Minimal;
   GetMinimalLabels(p1,unary1Minimal);
   size_t noMinimal1 = 0;
   for(size_t i=0; i<unary1Minimal.size(); i++)
      if(unary1Minimal[i]) noMinimal1++;
   OPENGM_ASSERT(noMinimal1>0);
   GetMinimalLabels(p2,unary2Minimal);
   size_t noMinimal2 = 0;
   for(size_t i=0; i<unary2Minimal.size(); i++)
      if(unary2Minimal[i]) noMinimal2++;
   OPENGM_ASSERT(noMinimal2>0);

   // if there are only essential minimal label combinations, i.e. ones that cannot be trivially made non-minimal
   // do nothing, otherwise add all minimal label combinations to immovables

   ValueType minLabelCost = std::numeric_limits<ValueType>::max();
   for(size_t i=0; i<immovable1.size(); i++)
      for(size_t j=0; j<immovable2.size(); j++)
         minLabelCost = std::min(minLabelCost,p[i,j]);
   OPENGM_ASSERT(minLabelCost > (-1)*eps_);

   size_t noEssential = 0;
   for(size_t i=0; i<immovable1.size(); i++)
      for(size_t j=0; j<immovable2.size(); j++)
         if((p[i,j] <= minLabelCost+eps_) && unary1Minimal[i] && unary2Minimal[j]) 
            noEssential++;
   
   if(noEssential == 0) {
      for(size_t i=0; i<immovable1.size(); i++) {
         for(size_t j=0; j<immovable2.size(); j++) {
            if(p[i,j] <= minLabelCost+eps_) {
               immovable1[i] = true;
               immovable2[j] = true;
            }
         }
      }
   }
   return minLabelCost;
}

template<class DATA, class ACC>
std::vector<typename DATA::GraphicalModelType::LabelType> 
IRI<DATA,ACC>::PruningCut(
      const std::vector<LabelType>& curLabeling, // labeling with negative cost. Some labels will end up being immovable.
      const std::vector<LabelType>& projLabeling, // labeling to which subset to one map projects.
      const std::vector<std::vector<bool> >& immovable,
      const PersistencyGMType& gm)
{
   std::clock_t beginTime = clock();

   std::vector<LabelType> optRestrLabeling = curLabeling;

   // construct auxiliary boolean problem to determine which of the current labels should be made immovable
   OPENGM_ASSERT(gm.factorOrder() <= 2); 
   int noEdges = 0;
   for(size_t f=0; f<gm.numberOfFactors(); f++)
      if(gm[f].numberOfVariables() == 2)
         noEdges++;

   kolmogorov::qpbo::QPBO<ValueType> q(n_,noEdges);
   q.AddNode(gm.numberOfVariables());

   // first label is projLabeling, second current negative cost test labeling
   for(size_t f=0; f<gm.numberOfFactors(); f++) {
      if(gm[f].numberOfVariables() == 1) {
         size_t v = gm.variableOfFactor(f,0); 
         LabelType l[] = {curLabeling[v]};
         q.AddUnaryTerm(v, 0.0, gm[f].operator()(l));
      } else if(gm[f].numberOfVariables() == 2) {
         size_t v1 = gm.variableOfFactor(f,0);
         size_t v2 = gm.variableOfFactor(f,1);
         LabelType l[] = {projLabeling[v1],curLabeling[v2]};
         ValueType b = gm[f].operator()(l);
         l[0] = curLabeling[v1]; l[1] = projLabeling[v2];
         ValueType c = gm[f].operator()(l);
         l[0] = curLabeling[v1]; l[1] = curLabeling[v2];
         ValueType d = gm[f].operator()(l);
         q.AddPairwiseTerm(v1,v2, 0.0, b,c,d);
         // do zrobiena: bez redukcji
      } else {
         std::cout << "Only pairwise models supported right now" << std::endl;
         throw;
      }
   }

   q.Solve(); // sprawdz, czy tylko jeden max-flow jest uzywany (to znaczy bez pelnej transformacji)
   q.ComputeWeakPersistencies(); // non-unique variables will also be determined
   for(size_t v=0; v<n_; v++) {
      // check if all nodes have been labelled
      OPENGM_ASSERT(q.GetLabel(v) == 0 || q.GetLabel(v) == 1);
      if(q.GetLabel(v) == 0) 
         optRestrLabeling[v] = projLabeling[v];
      else 
         optRestrLabeling[v] = curLabeling[v];
   }

   std::clock_t endTime = clock();
   pruningCutTime += endTime - beginTime;

   return optRestrLabeling;
}

// do zrobienia: dodaj immovable do argumentow
template<class DATA, class ACC>
bool
IRI<DATA,ACC>::SingleNodePruning(const size_t v, const size_t i, const PersistencyGMType& pgm)
{
   marray::Vector<size_t> fList;
   pgm.numberOfNthOrderFactorsOfVariable(v,1,fList);
   if(fList.size() != 1) // exactly one unary
      throw;
   OPENGM_ASSERT(fList.size() == 1);
   size_t fUnary = fList[0];
   LabelType l[] = {i};
   ValueType reducedCost = pgm[fUnary].operator()(l);
   for(size_t fc=0; fc<pgm.numberOfFactors(v); fc++) {
      size_t fPairwise = pgm.factorOfVariable(v,fc);
      if(pgm[fPairwise].numberOfVariables() != 1) {
         OPENGM_ASSERT(pgm[fPairwise].numberOfVariables() == 2); // only pairwise
         if(pgm[fPairwise].numberOfVariables() != 2) throw;
         size_t v1 = pgm.variableOfFactor(fPairwise,0);
         size_t v2 = pgm.variableOfFactor(fPairwise,1);
         size_t vNb = pgm.secondVariableOfSecondOrderFactor(v,fPairwise);
         LabelType lPairwise[2];
         if(vNb == v2) {
            lPairwise[0] = i; lPairwise[1] = l_[vNb];
         } else if(vNb == v1) {
            lPairwise[0] = l_[vNb]; lPairwise[1] = i;
         } else
            throw;
         reducedCost += pgm[fPairwise].operator()(lPairwise);
      }
   }

   if(reducedCost < (-1)*eps_) 
      return true;
   else 
      return false;
}

// do zrobienia: add parameter im
template<class DATA, class ACC>
size_t
IRI<DATA,ACC>::SingleNodePruning(
      std::vector<std::vector<bool> >& immovable,
      PersistencyGMType& pgm)
{
   std::clock_t beginTime = clock();

   // queue for (variable,label) tupels for which to check single node pruning condition
   // do zrobienia: many variable,label pairs will be double-checked, introduce dirty array. Seems not to be worth the effort, though.
   std::queue<std::pair<size_t,size_t> > pruningQueue; 
   for(size_t v=0; v<pgm.numberOfVariables(); v++)
      for(size_t i=0; i<pgm.numberOfLabels(v); i++)
         if(!immovable[v][i])
            pruningQueue.push(std::pair<size_t,size_t>(v,i));

   size_t newImmovable = 0;
   while(!pruningQueue.empty()) {
      std::pair<size_t,size_t> vi = pruningQueue.front();
      pruningQueue.pop();
      size_t v = vi.first;
      size_t i = vi.second;
      if(!immovable[v][i]) {
         if(SingleNodePruning(v,i,pgm)) {
            newImmovable++;
            immovable[v][i] = true;
            ConstructSubsetToOneMap(im_[v],l_[v],immovable[v]);
            UpdateMRF(pgm,im_,v);

            for(PGMConstFactorIterator fIt = pgm.factorsOfVariableBegin(v); fIt != pgm.factorsOfVariableEnd(v); fIt++)
               if(!pgm[*fIt].numberOfVariables() == 1)
                  for(PGMConstVariableIterator vIt = pgm.variablesOfFactorBegin(*fIt); vIt != pgm.variablesOfFactorEnd(*fIt); vIt++)
                     if(*vIt != v)
                        for(size_t ivIt=0; ivIt<pgm.numberOfLabels(*vIt); ivIt++)
                           if(!immovable[*vIt][ivIt])
                              pruningQueue.push(std::pair<size_t,size_t>(*vIt,ivIt));
         }
      }
   }

   std::clock_t endTime = clock();
   singleNodePruningTime += endTime - beginTime;

   return newImmovable;
}

template<class DATA, class ACC>
void 
IRI<DATA,ACC>::IncreaseImmovableLabels(
      std::vector<std::vector<bool> >& immovable, 
      const std::vector<IndexType>& l,
      PersSolverType& solver,
      PersistencyGMType& pgm)
{
   // get integer labeling. If value < 0 add all labels to immovables
   std::vector<LabelType> curLabeling;
   solver.arg(curLabeling);
   if(solver.graphicalModel().evaluate(curLabeling.begin()) < (-1)*eps_) {
      std::cout << "Compute pruning cut of negative cost labeling" << std::endl;
      std::vector<LabelType> optRestrLabeling = PruningCut(curLabeling,l_,immovable,pgm);

      size_t newImmovable = 0;
      for(size_t v=0; v<n_; v++) {
         if(!immovable[v][optRestrLabeling[v]])
            newImmovable++;
         immovable[v][optRestrLabeling[v]] = true;
      }
      std::cout << "Added " << newImmovable << " new immovable labels" << std::endl;
      OPENGM_ASSERT(newImmovable>0); 
      if(newImmovable == 0)
         throw;
   } //else {

#ifdef OPENGM_IRI_TRWS
      // get reparametrized model and add minimal reparametrized labels for each potential
      ReparametrizedGMType repGmSolved;
      typename PersSolverType::ReparametrizerType* prepa=solver.getReparametrizer();
      //prepa->reparametrize(immovable); // reparametrize such that immovable labels have unaries == 0
      prepa->reparametrize(); // reparametrize such that immovable labels have unaries == 0
      prepa->getReparametrizedModel(repGmSolved);
      OPENGM_ASSERT(solver.graphicalModel().numberOfVariables() == repGmSolved.numberOfVariables());
      OPENGM_ASSERT(solver.graphicalModel().numberOfFactors() == repGmSolved.numberOfFactors());

      // inspect reparametrized unary labels
      for(size_t v=0; v<n_; v++) {
         std::vector<ValueType> repPotential;
         GetUnaryFactor<ReparametrizedGMType>(repGmSolved,v,repPotential);
         ValueType minLabelCost = std::numeric_limits<ValueType>::max();
         for(size_t i=0; i<immovable[i].size(); i++) {
            minLabelCost = std::min(minLabelCost,repPotential[i]);
            //if(immovable[v][i]) { // only valid for reparametrizations with immovables
            //   OPENGM_ASSERT(repPotential[i] > (-1)*eps_);
            //}
         }
         for(size_t i=0; i<immovable[i].size(); i++) {
            if(repPotential[i] <= minLabelCost+eps_) {
               immovable[v][i] = true;
            }
         }
      }

      delete prepa;
#endif

#ifdef OPENGM_IRI_CPLEX
      // label becomes immovable if it has positive marginal
      for(size_t v=0; v<n_; v++) {
         IndependentFactorType indFac;
         solver.variable(v, indFac);
         OPENGM_ASSERT( indFac.numberOfVariables() == 1 );
         for(size_t i=0; i<indFac.numberOfLabels( 0 ); i++)
            if(indFac(i) > eps_)
               immovable[v][i] = true;
      }
#endif

   //}

   ConstructSubsetToOneMap(im_,l,immovable);
   UpdateMRF(pgm, im_);

   //apply single node pruning
   std::cout << "Compute single node pruning" << std::endl;
   size_t newImmovable = SingleNodePruning(immovable,pgm);
   std::cout << "Added " << newImmovable << " new immovable labels" << std::endl;
}

template<class DATA, class ACC>
inline InferenceTermination
IRI<DATA,ACC>::infer()
{
#ifdef OPENGM_IRI_TRWS
   typename SolverType::DDVectorType dd; // reparametrization stored for subsequent speedup
#endif

   std::clock_t beginInferenceTime = clock();
   // first solve the original problem to get a labeling
   {
      std::clock_t beginTime = clock();
      SolverType solver(gm_,SolverParam_);
      solver.infer();
      solver.arg(l_); // do zrobienia: Wez wszystkie label z wszystkich drzew, moze tez jeszcze patrz lokalnie z lazy flipper za lepszymi itp.
      ImproveLabeling(l_);

      clock_t endTime = clock();
      initialInferenceTime = endTime - beginTime;
     
      std::cout << "Time for finding initial labeling: " << double(initialInferenceTime) / CLOCKS_PER_SEC << std::endl;

      for(size_t v=0; v<n_; v++)
         immovable_[v][l_[v]] = true;

      std::cout << "Energy of subset to one labeling = " << gm_.evaluate(l_.begin()) << std::endl;
      std::cout << "Lower bound of original problem = " << solver.bound() << std::endl;

      if(IsGloballyOptimalSolution(solver)) {
         std::cout << "Globally optimal solution found by relaxation" << std::endl;
         return NORMAL;
      }
#ifdef OPENGM_IRI_TRWS
      solver.getDDVector(&dd);
#endif
   }

   // construct the modified MRF based on the improving mapping
   PersistencyGMType pgm;
   // build subset to one mapping based on initial labeling l_
   std::cout << "Constructing modified model" << std::endl;
   ConstructSubsetToOneMap(im_,l_,immovable_);
   ConstructMRF(pgm, im_);

   // increment immovable elements until constructed subset to one map can be certified to be improving
   size_t iter=0;
   while(!AllLabelImmovable(immovable_)) {
      std::cout << "New iteration " << iter << " in improving mapping persistency algorithm" << std::endl;

      std::clock_t beginTime = clock();
#ifdef OPENGM_IRI_TRWS
      PersSolverParam_.initPoint_ = dd;
#endif
      PersSolverType solver(pgm,PersSolverParam_);
      std::cout << "Solving modified model" << std::endl;
#ifdef OPENGM_IRI_TRWS
      PrimalBoundVisitor<PersSolverType> visitor(-eps_,1); // create visitor stopping whenever some integer labeling with value < 0 is found
      solver.infer(visitor); 
#endif
#ifdef OPENGM_IRI_CPLEX
      solver.infer();
#endif
#ifdef OPENGM_IRI_TRWS
      solver.getDDVector(&dd);
#endif

      std::clock_t endTime = clock();
      subsequentInferenceTime += endTime - beginTime;
      std::cout << "Time for iteration " << iter << " = " << double(endTime - beginTime)/double(CLOCKS_PER_SEC) << std::endl;

      std::cout << "Lower bound of modified problem = " << solver.bound() << std::endl;
      if(solver.bound() > (-1)*eps_) {
         std::cout << "lower bound is zero, map is improving" << std::endl;
         size_t noLabels = 0;
         for(size_t v=0; v<n_; v++)
            noLabels += gm_.numberOfLabels(v);
         std::cout << "Number of immovable labels = " << NoImmovableLabels(immovable_) 
            << ", total number of labels = " << noLabels
            << ", percentage partial optimality = " << 1.0 - (double(NoImmovableLabels(immovable_))-double(n_))/(double(noLabels)-double(n_))
            << std::endl;
         break;
      }

      IncreaseImmovableLabels(immovable_,l_,solver,pgm);

      std::cout << "Number of immovable labels = " << NoImmovableLabels(immovable_) << std::endl;
      iter++;
      if(iter > 5000) {
         std::cout << "Could not obtain improving mapping after 5000 iterations, aborting" << std::endl;
         break;
      }
   }
   
   std::cout << "total time = " << double(clock() - beginInferenceTime)/double(CLOCKS_PER_SEC) << std::endl;
   std::cout << "Initial inference time = " << double(initialInferenceTime)/double(CLOCKS_PER_SEC) << std::endl;
   std::cout << "Subsequent inference time = " << double(subsequentInferenceTime)/double(CLOCKS_PER_SEC) << std::endl;
   std::cout << "MRF construction time = " << double(MRFModificationTime)/double(CLOCKS_PER_SEC) << std::endl;
   std::cout << "Pruning Cut time = " << double(pruningCutTime)/double(CLOCKS_PER_SEC) << std::endl;
   std::cout << "Single Node Pruning time = " << double(singleNodePruningTime)/double(CLOCKS_PER_SEC) << std::endl;

   // adjust popt_data object to account for excluded labels
   for(IndexType i=0; i<n_; i++)
      for(LabelType x_i=0; x_i<gm_.numberOfLabels(i); x_i++)
         if(immovable_[i][x_i] == false && x_i != l_[i]) // do zrobienia: sprawdz, czy nie == true
         //if(x_i != l_[i]) 
            d_.setFalse(i,x_i);

   return NORMAL;
}

/*template<class DATA,class ACC>
InferenceTermination 
IRI<DATA,ACC>::arg(
      std::vector<typename GM::LabelType>& l,
      const size_t T
      ) const
{
   l = l_; 
   return NORMAL;
}*/

} // namespace IRI
} // namespace opengm

#endif // OPENGM_IRI_HXX
