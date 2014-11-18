#ifndef OPENGM_IRI_HXX
#define OPENGM_IRI_HXX

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/visitors/visitors.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/view.hxx>
#include "popt_data.hxx"

#include "persistency_potential_perm.hxx"

// For Pruning Cut
#include <opengm/inference/external/qpbo.hxx>

#include <vector>
#include <limits>
#include <queue>
#include <ctime>

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>

#include <png++/png.hpp>
#include <stdint.h>
#include "lodepng.h"
#include "lodepng.cpp"

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
   OPENGM_ASSERT(gm[f].numberOfVariables() == 2);
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
      std::cout << "value " << inf.value() << " bound " << inf.bound() << std::endl;
   }

   void addLog(const std::string & logName){}
   void log(const std::string & logName,const double logValue){}

private:
   double eps_;
   size_t minIter_;
   size_t nIter_;
};

//! [class IRI]
/// IRI - iterative relaxed inference
/// Persistency with improving mappings: iterative algorithm.
/// Based on the improving mapping framework described in
/// "Maximum Persistency in Energy Minimization", A. Shekhovtsov, CVPR 2014.
///
/// Corresponding author: Paul Swoboda, email: swoboda@math.uni-heidelberg.de
///
///\ingroup inference

template<class DATA,class ACC,template <typename,typename> class SOLVER> 
// DATA is POpt_data<GM,ACC>
// SOLVER is derived from POpt_IRI_SolverBase<GM,ACC>
// do zrobienia: template data musi byc template template parameter
// do zrobienia: templatyzuj DATA tak jak SOLVER
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
   typedef GraphicalModel<ValueType,opengm::Adder,PersPotentialType,
           opengm::DiscreteSpace<IndexType,LabelType> > PersistencyGMType;

   typedef typename PersistencyGMType::ConstFactorIterator PGMConstFactorIterator;
   typedef typename PersistencyGMType::ConstVariableIterator PGMConstVariableIterator;

  
   typedef SOLVER<GM,ACC> InitSolverType;
   typedef SOLVER<PersistencyGMType,ACC> IterSolverType;

   // possibly make this available to the command line
   struct Parameter { 
      enum METHOD {SUBSET_TO_ONE, ALL_TO_ONE}; 
      METHOD method_;
      Parameter() : method_(SUBSET_TO_ONE) {};
      //Parameter() : method_(ALL_TO_ONE) {};
   } param_;

   IRI(DATA& d, const Parameter& param = Parameter());
   virtual std::string name() const {return "IRI";}
   const GraphicalModelType& graphicalModel() const {return gm_;};
   InferenceTermination infer();
   //template<class VISITOR>
   //   InferenceTermination infer(VISITOR &);
   const static double eps_; 
   
private:
   void InitializeImprovingMap();
   void InitializeImmovable();
   void ImproveLabeling(std::vector<LabelType>& l);
   void ConstructMRF(PersistencyGMType& pgm, const std::vector<LabelType>& l, const std::vector<std::vector<LabelType> >& im);
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
   void IncreaseImmovableLabels(
         std::vector<std::vector<bool> >& immovable,
         const std::vector<IndexType>& l,
         IterSolverType& solver,
         PersistencyGMType& pgm);
   //bool IsGloballyOptimalSolution(SOLVER& solver);
   size_t NoImmovableLabels(const std::vector<std::vector<bool> >& immovable);
   void savePartialOptimalityMask(const std::string& name, const std::vector<std::vector<LabelType> >& im) const;

   DATA& d_;
   const GM& gm_;
   const size_t n_; // number of variables

   // improving mapping
   std::vector<std::vector<LabelType> > im_;
   // initial labeling for constructing subset to one mapping
   std::vector<IndexType> l_;
   // immovable labels
   std::vector<std::vector<bool> > immovable_;

   // time measurements for individual operations
   std::clock_t totalTime, singleNodePruningTime, pruningCutTime, initialInferenceTime, subsequentInferenceTime, MRFModificationTime;
};

template<class DATA,class ACC,template <typename,typename> class SOLVER>
const double IRI<DATA,ACC,SOLVER>::eps_ = 1.0e-6;

template<class DATA,class ACC,template <typename,typename> class SOLVER>
IRI<DATA,ACC,SOLVER>::IRI(DATA& d, const Parameter& param) :
   param_(param),
   d_(d),
   gm_(d.graphicalModel()),
   n_(d.graphicalModel().numberOfVariables())
{
   if(gm_.factorOrder() > 2) {
      throw RuntimeError("Only second order models supported");
   }
   OPENGM_ASSERT(typeid(ACC) == typeid(opengm::Minimizer));
   //if(typeid(ACC) == typeid(opengm::Minimizer)) {
   //   throw RuntimeError("Only minimization supported");
   //}

   InitializeImprovingMap();
   InitializeImmovable();

   totalTime = 0;
   singleNodePruningTime = 0;
   pruningCutTime= 0;
   initialInferenceTime = 0;
   subsequentInferenceTime = 0;
   MRFModificationTime = 0;
}

template<class DATA,class ACC,template <typename,typename> class SOLVER>
void
IRI<DATA,ACC,SOLVER>::InitializeImprovingMap()
{
   im_.resize(n_);
   for(size_t v=0; v<n_; v++)
      im_[v].resize(gm_.numberOfLabels(v),0);
}

template<class DATA,class ACC,template <typename,typename> class SOLVER>
void
IRI<DATA,ACC,SOLVER>::InitializeImmovable()
{
   immovable_.resize(n_);
   for(size_t i=0; i<n_; i++) 
      immovable_[i].assign(gm_.numberOfLabels(i),false);
}

template<class DATA,class ACC,template <typename,typename> class SOLVER>
void 
IRI<DATA,ACC,SOLVER>::ImproveLabeling(std::vector<LabelType>& l)
{
   // given an initial labeling, we must find a labeling such that no label is forbidden. This need not hold for the inital labeling.
   // to do: possibly improve labeling with lazy flipper
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
template<class DATA,class ACC,template <typename,typename> class SOLVER>
void
IRI<DATA,ACC,SOLVER>::ConstructMRF(PersistencyGMType& pgm, const std::vector<LabelType>& l, const std::vector<std::vector<LabelType> >& im)
{
   std::clock_t beginTime = clock();

   pgm = PersistencyGMType(gm_.space());
   for(size_t f=0; f<gm_.numberOfFactors(); f++) {
      // get associated variables
      std::vector<IndexType> varFactor(gm_[f].variableIndicesBegin(), gm_[f].variableIndicesEnd());
      // get associated sublabeling
      std::vector<LabelType> subL(varFactor.size());
      for(size_t v=0; v<varFactor.size(); v++)
         subL[v] = l_[varFactor[v]];
      // get associated pixelwise improving mappings
      std::vector<std::vector<LabelType> > imFactor(varFactor.size());
      for(size_t v=0; v<varFactor.size(); v++)
         imFactor[v] = im[varFactor[v]];
      // add modified potential: theta - P^\T * theta
      PersPotentialType p(gm_[f],subL,imFactor);
      pgm.addFactor(pgm.addFunction(p), gm_[f].variableIndicesBegin(), gm_[f].variableIndicesEnd());
   }

   std::clock_t endTime = clock();
   MRFModificationTime += endTime - beginTime;
}

// update MRF with cost theta - P^\T * theta, when P has changed
template<class DATA,class ACC,template <typename,typename> class SOLVER>
void
IRI<DATA,ACC,SOLVER>::UpdateMRF(PersistencyGMType& pgm, const std::vector<std::vector<LabelType> >& im)
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
template<class DATA,class ACC,template <typename,typename> class SOLVER>
void
IRI<DATA,ACC,SOLVER>::UpdateMRF(PersistencyGMType& pgm, const std::vector<std::vector<LabelType> >& im, size_t v)
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


template<class DATA,class ACC,template <typename,typename> class SOLVER>
void
IRI<DATA,ACC,SOLVER>::ConstructSubsetToOneMap(
      std::vector<typename GM::LabelType>& im,
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

template<class DATA,class ACC,template <typename,typename> class SOLVER>
void
IRI<DATA,ACC,SOLVER>::ConstructSubsetToOneMap(
      std::vector<std::vector<typename GM::LabelType> >& im,
      const std::vector<typename GM::LabelType> l,
      const std::vector<std::vector<bool> >& immovable)
{
   im.resize(n_);
   for(size_t v=0; v<n_; v++) 
      ConstructSubsetToOneMap(im[v],l[v],immovable[v]);
}

template<class DATA,class ACC,template <typename,typename> class SOLVER>
bool 
IRI<DATA,ACC,SOLVER>::IsProjection(const std::vector<typename GM::LabelType>& p)
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

template<class DATA,class ACC,template <typename,typename> class SOLVER>
bool 
IRI<DATA,ACC,SOLVER>::AllLabelImmovable(const std::vector<std::vector<bool> >& immovable)
{
   OPENGM_ASSERT(n_ == immovable.size());
   for(size_t v=0; v<n_; v++)
      for(size_t l=0; l<immovable[v].size(); l++)
         if(!immovable[v][l])
            return false;
   return true;
}

template<class DATA,class ACC,template <typename,typename> class SOLVER>
size_t 
IRI<DATA,ACC,SOLVER>::NoImmovableLabels(const std::vector<std::vector<bool> >& immovable)
{
   OPENGM_ASSERT(n_ == immovable.size());
   size_t noImmovableLabels = 0;
   for(size_t v=0; v<n_; v++)
      for(size_t l=0; l<immovable[v].size(); l++)
         if(immovable[v][l])
            noImmovableLabels++;
   return noImmovableLabels;
}

template<class DATA,class ACC,template <typename,typename> class SOLVER>
std::vector<typename DATA::GraphicalModelType::LabelType> 
IRI<DATA,ACC,SOLVER>::PruningCut(
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
template<class DATA,class ACC,template <typename,typename> class SOLVER>
bool
IRI<DATA,ACC,SOLVER>::SingleNodePruning(const size_t v, const size_t i, const PersistencyGMType& pgm)
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
template<class DATA,class ACC,template <typename,typename> class SOLVER>
size_t
IRI<DATA,ACC,SOLVER>::SingleNodePruning(
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

template<class DATA,class ACC,template <typename,typename> class SOLVER>
void 
IRI<DATA,ACC,SOLVER>::IncreaseImmovableLabels(
      std::vector<std::vector<bool> >& immovable, 
      const std::vector<IndexType>& l,
      IterSolverType& solver,
      PersistencyGMType& pgm)
{
   // get integer labeling. If value < 0 add all labels to immovables
   std::vector<LabelType> curLabeling;
   solver.arg(curLabeling);
   size_t newImmovable =0;
   if(solver.graphicalModel().evaluate(curLabeling.begin()) < (-1)*eps_) {
      std::cout << "Compute pruning cut of negative cost labeling" << std::endl;
      std::vector<LabelType> optRestrLabeling = PruningCut(curLabeling,l_,immovable,pgm);

      for(size_t v=0; v<n_; v++) {
         if(!immovable[v][optRestrLabeling[v]])
            newImmovable++;
         immovable[v][optRestrLabeling[v]] = true;
      }
      std::cout << "Added " << newImmovable << " new immovable labels" << std::endl;
      OPENGM_ASSERT(newImmovable>0); 
      if(newImmovable == 0)
         throw;
   } else {
      newImmovable = solver.IncreaseImmovableLabels(immovable,l);
   }

   ConstructSubsetToOneMap(im_,l,immovable);
   UpdateMRF(pgm, im_);

   //apply single node pruning
   std::cout << "Compute single node pruning" << std::endl;
   newImmovable += SingleNodePruning(immovable,pgm);
   std::cout << "Added " << newImmovable << " new immovable labels" << std::endl;

   // if all to one set all labels of some node to immovable as soon as more than two labels are immovable
   if(param_.method_ == Parameter::ALL_TO_ONE) {
      for(size_t v=0; v<n_; v++) {
         size_t noImmovables = 0;
         for(size_t l=0; l<immovable[v].size(); l++) {
            if(immovable[v][l] == true) {
               noImmovables++;
            }
         }
         if(noImmovables > 1) {
            for(size_t l=0; l<immovable[v].size(); l++) {
               immovable[v][l] = true;
            }
         }
      }
      ConstructSubsetToOneMap(im_,l,immovable);
      UpdateMRF(pgm, im_);
   }
}

template<class DATA,class ACC,template <typename,typename> class SOLVER>
inline InferenceTermination
IRI<DATA,ACC,SOLVER>::infer()
{
   typename SOLVER<GM,ACC>::WarmStartParamType warmStartParam;

   std::clock_t beginInferenceTime = clock();
   size_t iter=0;
   PersistencyGMType pgm;
   // first solve the original problem to get a labeling
   {
      std::clock_t beginTime = clock();

      InitSolverType solver(gm_);
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

      if(solver.IsGloballyOptimalSolution()) {
         std::cout << "Globally optimal solution found by relaxation" << std::endl;
         goto FINISH;
      }
      solver.GetWarmStartParam(warmStartParam);
   }

   // construct the modified MRF based on the improving mapping
   // build subset to one mapping based on initial labeling l_
   std::cout << "Constructing modified model" << std::endl;
   ConstructSubsetToOneMap(im_,l_,immovable_);
   ConstructMRF(pgm, l_, im_);

   // increment immovable elements until constructed subset to one map can be certified to be improving
   while(!AllLabelImmovable(immovable_)) {
      std::cout << "New iteration " << iter << " in improving mapping persistency algorithm" << std::endl;

      std::clock_t beginTime = clock();
      IterSolverType solver(pgm);
      solver.SetWarmStartParam(warmStartParam);

      std::cout << "Solving modified model" << std::endl;
      PrimalBoundVisitor<typename IterSolverType::SolverType> visitor(-eps_,5); // visitor stops whenever some integer labeling with value < 0 is found
      solver.infer(visitor);  

      solver.GetWarmStartParam(warmStartParam);

      std::clock_t endTime = clock();
      subsequentInferenceTime += endTime - beginTime;
      std::cout << "Time for iteration " << iter << " = " << double(endTime - beginTime)/double(CLOCKS_PER_SEC) << std::endl;

      //std::cout << "Lower bound of modified problem = " << solver.bound() << std::endl;
      if(solver.bound() > (-1)*eps_) {
         //std::cout << "lower bound is zero, map is improving" << std::endl;
         size_t noLabels = 0;
         for(size_t v=0; v<n_; v++)
            noLabels += gm_.numberOfLabels(v);
         //std::cout << "Number of immovable labels = " << NoImmovableLabels(immovable_) 
         //   << ", total number of labels = " << noLabels
         //   << ", percentage partial optimality = " << 1.0 - (double(NoImmovableLabels(immovable_))-double(n_))/(double(noLabels)-double(n_))
         //   << std::endl;
         break;
      }

      IncreaseImmovableLabels(immovable_,l_,solver,pgm);

      //std::cout << "Number of immovable labels = " << NoImmovableLabels(immovable_) << std::endl;
      iter++;
      if(iter > 5000) {
         std::cout << "Could not obtain improving mapping after 5000 iterations, aborting" << std::endl;
         break;
      }

      std::stringstream iterations_str;
      iterations_str << iter;
      std::string intermed_file = ("iterative_relaxed_inference_mask_iter_");
      intermed_file.append(iterations_str.str());
      intermed_file.append(".pgm");
      savePartialOptimalityMask(intermed_file,im_);
   }

FINISH:
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



//write tikz image
template<class DATA,class ACC,template <typename,typename> class SOLVER>
void 
IRI<DATA,ACC,SOLVER>::savePartialOptimalityMask(
      const std::string& name, 
      const std::vector<std::vector<LabelType> >& im) const
{
   size_t n1, n2;
   if(im.size() == 30*30) {
      n1 = 30; n2 = 30;
   } else if(im.size() == 240*320) {
      n1 = 240; n2 = 320;
   } else if(im.size() == 240*360) {
      n1 = 240; n2 = 360;
   } else {
      std::cout << "Could not determine image dimensions" << std::endl;
      return;
   }

   /*
   std::vector<unsigned char> PngBuffer(im.size());
   for(IndexType i=0; i<im.size(); i++) {
      size_t x = i / n1;
      size_t y = i % n2;
      size_t noExcludedLabels = 0;
      for(size_t x_i=0; x_i<im[i].size(); x_i++)
         if(im_[i][x_i] != x_i)
            noExcludedLabels++;
      PngBuffer[i] = noExcludedLabels;
   }
   std::vector<unsignd char> ImageBuffer;
   lodepng::encode(ImageBuffer,PngBuffer, n1,n2);
   lodepng::save_file(ImageBuffer,name);
   */
   /*
   png::image<png::rgb_pixel> image(n1,n2);
   for(IndexType i=0; i<im.size(); i++) {
      size_t x = i / n1;
      size_t y = i % n2;
      size_t noExcludedLabels = 0;
      for(size_t x_i=0; x_i<im[i].size(); x_i++)
         if(im_[i][x_i] != x_i)
            noExcludedLabels++;
      image[x][y] = png::rgb_pixel(0,0,noExcludedLabels);
   }

   image.write(name);
   return;
   */

   std::ofstream fout;
	fout.open(name.c_str());
	if (!fout.is_open()) throw "Could not open file";

   fout << "P5\n" << n1 << " " << n2 << "\n255\n";
   for(IndexType i=0; i<im.size(); i++) {
      size_t x = i / n1;
      size_t y = i % n2;

      char noExcludedLabels = 0;
      for(size_t x_i=0; x_i<im[i].size(); x_i++)
         if(im_[i][x_i] != x_i)
            noExcludedLabels++;
      noExcludedLabels = char( double(noExcludedLabels)/im[i].size()*128.0 );
      fout.write(&noExcludedLabels,1*(sizeof(unsigned char)));
   }


   fout.close();
   return;


	//fout << "\\begin{tikzpicture}[scale=0.25] \n\\tikzstyle{outside}=[circle,thick,draw=red!75,scale=0.3] \n\\tikzstyle{inside}=[circle,thick,draw=yellow!75,fill=yellow!20,scale=0.3] \n\\tikzstyle{boundary}=[circle,thick,draw=green!75,fill=green!20,scale=0.3]" << std::endl;

	//fout  << "\\node[circle,thick,draw=red!75,scale=0.3,label=0:Outside node] at (33,29) {};" << std::endl;
	//fout  << "\\node[circle,thick,draw=yellow!75,fill=yellow!20,scale=0.3,label=0:Inside node] at (33,25) {};" << std::endl;
	//fout  << "\\node[circle,thick,draw=green!75,fill=green!20,scale=0.3,label=0:Boundary node] at (33,21) {};" << std::endl;

   // assume that we have a grid with n1 x n2 pixels. Automatically infer n1 and n2 from a precomputed table.
   // write number of labels that cannot be excluded down at each pixel position
   for(IndexType i=0; i<im.size(); i++) {
      size_t x = i / n1;
      size_t y = i % n2;

      ValueType noExcludedLabels = 0;
      for(size_t x_i=0; x_i<im[i].size(); x_i++)
         if(im_[i][x_i] != x_i)
            noExcludedLabels++;

      fout << "\\node[" << noExcludedLabels << "of" << im[i].size() << "]";
      fout << " (" <<x<<"a"<<y << ") at (" <<x<<","<<y<< ")  {};" << std::endl;
	}
	fout << std::endl << std::endl;

   // write edges
	fout  << "\\foreach \\from/\\to in{" << std::endl;
	for(IndexType i=0; i<im.size(); i++) {
		for(IndexType j=0; j<im.size(); j++) {
			int x1 = i / n1;
			int y1 = i % n2;
			int x2 = j / n1;
			int y2 = j % n2;
         if(x1!=x2||y1!=y2)
            if(abs(x1-x2) <= 1 && abs(y1-y2) <= 1)
               if(!(abs(x1-x2) == 1 && abs(y1-y2) == 1)) 
                  if(!(abs(x1-x2) == 0 && abs(y1-y2) == 0)) {
                     fout << x1 <<"a"<< y1 << "/" << x2<<"a"<< y2;
                     if(!( i==im.size()-1 && j==im.size()-2)) {
                        fout << "," << std::endl;
                     }
                  }
      }
	}
	fout << "}\\draw (\\from) -- (\\to);" << std::endl;
	fout << std::endl;
	//fout << "\\end{tikzpicture}";
	fout.close();
}


} // namespace IRI
} // namespace opengm

#endif // OPENGM_IRI_HXX
