#pragma once
#ifndef OPENGM_PBP_HXX
#define OPENGM_PBP_HXX

#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/visitors/visitors.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/view.hxx>
#include "opengm/functions/modelviewfunction.hxx"
#include "popt_data.hxx"
#include "connected_components.hxx"
#include "boundaryviewfunction.hxx"
#include "sum_view_function.hxx"

#include <vector>
#include <limits>
#include <queue>
#include <ctime>

// do zrobienia: skasuj
//#include "store_into_explicit.hxx"

namespace opengm {

namespace PBP {

//! [class PBP]
/// PBP - partial optimality by pruning]
/// Persistency by pruning for computing partially optimal labelings on general graphical models, 
/// published in CVPR 2014 as 
/// "Partial Optimality by Pruning for MAP-inference with General Graphical Models"
/// by P. Swoboda, B. Savchynskyy, J.H. Kappes and C. Schn\"orr.
///
/// Corresponding author: Paul Swoboda, email: swoboda@math.uni-heidelberg.de
///
///\ingroup inference

template<class DATA,class ACC,template <typename,typename> class SOLVER> 
// DATA is POpt_data<GM,ACC>
// SOLVER is derived from POpt_PBP_SolverBase<GM,ACC>
// do zrobienia: template data musi byc template template parameter
class PBP : public POpt_Inference<DATA, ACC>
{
public:
   typedef ACC AccumulationType;
   typedef typename DATA::GraphicalModelType   GM;
   typedef typename DATA::GraphicalModelType   GmType;
   typedef typename DATA::GraphicalModelType   GraphicalModelType;
   typedef typename GM::FunctionIdentifier     FunctionIdentifierType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef ViewFunction<GraphicalModelType>    ViewFunctionType;
   typedef ExplicitFunction<ValueType>         ExplicitFunctionType;
   typedef typename GM::ConstFactorIterator ConstFactorIterator;
   typedef typename GM::ConstVariableIterator ConstVariableIterator;

   typedef opengm::BoundaryViewFunction<GM> PersPotentialType;
   typedef opengm::SumViewFunction<GM> SumFunctionType;

   typedef GraphicalModel<ValueType,opengm::Adder,
           OPENGM_TYPELIST_4(ExplicitFunctionType,ViewFunctionType,PersPotentialType,SumFunctionType),
           opengm::DiscreteSpace<IndexType,LabelType> > PersistencyGMType;

   typedef typename PersistencyGMType::ConstFactorIterator PGMConstFactorIterator;
   typedef typename PersistencyGMType::ConstVariableIterator PGMConstVariableIterator;

   typedef SOLVER<GM,ACC> InitSolverType;
   typedef SOLVER<PersistencyGMType,ACC> IterSolverType;

   struct Parameter{};

   PBP(DATA& d, const Parameter& param = Parameter());
   virtual std::string name() const {return "PBP";}
   const GraphicalModelType& graphicalModel() const {return gm_;};
   InferenceTermination infer();
   //template<class VISITOR>
   //   InferenceTermination infer(VISITOR &);
   const static double eps_; 
   
private:
   DATA& d_;
   const GM& gm_;
   const size_t n_; // number of variables

   void buildRedGm(PersistencyGMType& redGm, const std::vector<LabelType>& initLabeling, const std::vector<nodePos::Type>& persistencyLoc); 
   bool isInsideFactor(const size_t factorId, const std::vector<nodePos::Type>& persistencyLoc) const;
   bool isBoundaryFactor(const size_t factorId, const std::vector<nodePos::Type>& persistencyLoc) const;
   IndexType BoundaryFuncIndex(
         const std::vector<std::vector<IndexType> >& boundaryFactorIndices,
         const std::vector<IndexType>& redFactorIndices,
         const std::vector<IndexType>& origFactorIndices,
         const std::vector<nodePos::Type>& persistencyLoc) const;

   IndexType updateInterior(
         const std::vector<LabelType>& initLabeling, 
         const std::vector<LabelType>& curLabeling,
         const std::vector<bool>& consistent,
         std::vector<nodePos::Type>& persistencyLoc);
   std::pair<ValueType,ValueType> solveGM(
         const PersistencyGMType& gm, 
         std::vector<LabelType>& l,
         std::vector<bool>& c);

   bool PersistencyCandidatesExist(const std::vector<nodePos::Type>& persistencyLoc) const;

   std::vector<LabelType> SubToOrigLabeling(const std::vector<LabelType>& initLabeling, const std::vector<LabelType>& curLabeling, const std::vector<nodePos::Type>& persistencyLoc) const;
   std::vector<bool> SubToOrigConsistent(const std::vector<bool>& consistent, const std::vector<nodePos::Type>& persistencyLoc) const;

};

template<class DATA,class ACC,template <typename,typename> class SOLVER>
const double PBP<DATA,ACC,SOLVER>::eps_ = 1.0e-6;

template<class DATA,class ACC,template <typename,typename> class SOLVER>
PBP<DATA,ACC,SOLVER>::PBP(DATA& d, const Parameter& param) :
   d_(d),
   gm_(d.graphicalModel()),
   n_(d.graphicalModel().numberOfVariables())
{
   OPENGM_ASSERT(typeid(ACC) == typeid(opengm::Minimizer));
}

template<class DATA, class ACC,template <typename,typename> class SOLVER>
InferenceTermination 
PBP<DATA, ACC, SOLVER>::infer() 
{
   std::clock_t beginTime = clock();
   std::vector<std::clock_t> iterationTime;

   // first solve the original problem to get a labeling
   std::vector<LabelType> initLabeling, curLabeling;
   std::vector<bool> consistent;
   std::vector<nodePos::Type> persistencyLoc(n_,nodePos::inside);
   {
      InitSolverType solver(gm_);
      solver.infer();
      InferenceTermination status = solver.arg(initLabeling); 
      if( status != NORMAL ) {
         std::cout << "Solver did not converge" << std::endl;
         return status;
      }

      if(solver.IsGloballyOptimalSolution()) {
         if(std::abs(solver.bound()-solver.value()) > eps_)
            throw RuntimeError("Globally optimal labeling found, but integrality gap persists!");

         std::cout << "Globally optimal solution found by relaxation" << std::endl;
         for(size_t i=0; i<n_; i++)
            d_.setTrue(i,initLabeling[i]);

         return NORMAL;
      } else {
         solver.consistent(consistent);
      }
   }

   curLabeling = initLabeling;
   iterationTime.push_back(clock() - beginTime);

   size_t iterations = 1;
   while( (updateInterior(initLabeling,curLabeling,consistent,persistencyLoc) > 0) && PersistencyCandidatesExist(persistencyLoc) ) { // shrink interior by pruning variables


      PersistencyGMType redGm;
      buildRedGm(redGm,initLabeling,persistencyLoc);

      // do zrobienia: skasuj
      //if(iterations == 1)
      //   store_into_explicit(redGm,std::string("test.h5"));

      /*
      IterSolverType solver(redGm);
      InferenceTermination status = solver.infer();
      if( status != NORMAL ) {
         std::cout << "Solver did not converge" << std::endl;
         return status;
      }
      */

      std::pair<ValueType,ValueType> bounds;
      bounds = solveGM(redGm,curLabeling,consistent);
      
      std::cout << "current lower bound = " << bounds.first << ", labeling value = " << bounds.second << std::endl;

      iterations++;
      iterationTime.push_back(clock() - beginTime);
   }

   std::cout << "Found persistent set after " << iterations << " iterations." << endl;
   IndexType p=0;
   for(size_t i=0; i<n_; i++) {
      if(persistencyLoc[i] != nodePos::outside) {
         d_.setTrue(i,initLabeling[i]);
         p++;
      }
   }
   std::cout << "Algorithm: Percentage partial optimality = " << double(p)/double(n_) << std::endl;
   std::cout << "Data object: Percentage partial optimality = " << d_.getPOpt() << std::endl;

   for(size_t i=0; i<iterationTime.size(); i++)
      std::cout << "Time for iteration " << i << " = " <<  double(iterationTime[i]) / CLOCKS_PER_SEC << std::endl;


   return NORMAL;
}


template<class DATA, class ACC, template <typename,typename> class SOLVER>
typename DATA::GraphicalModelType::IndexType
PBP<DATA,ACC,SOLVER>::updateInterior(
      const std::vector<LabelType>& initLabeling, 
      const std::vector<LabelType>& curLabeling,
      const std::vector<bool>& consistent,
      std::vector<nodePos::Type>& persistencyLoc)
{
   OPENGM_ASSERT(initLabeling.size() == persistencyLoc.size());
   OPENGM_ASSERT(curLabeling.size() == consistent.size());
   OPENGM_ASSERT(curLabeling.size() <= initLabeling.size());

   IndexType newOutside = 0;
   std::vector<LabelType> curLabelingOrig = SubToOrigLabeling(initLabeling,curLabeling,persistencyLoc);
   std::vector<bool> consistentOrig = SubToOrigConsistent(consistent,persistencyLoc);
   
   // determine new outside nodes
   for(size_t i=0; i<n_; i++) {
      if( (curLabelingOrig[i] != initLabeling[i]) || !consistentOrig[i]) {
         if(persistencyLoc[i] != nodePos::outside) {
            newOutside ++;
            persistencyLoc[i] = nodePos::outside;
         }
      }
   }

   // new outside nodes determined, now determine boundary
	for( IndexType i = 0; i<n_; i++) {
		if( persistencyLoc[i] == nodePos::inside ) {
			// check if  there is an outside node in a common factor
			bool has_outside_neighbor = false;
			for( IndexType factor=0; factor<gm_.numberOfFactors( i ); factor++ ) {
				IndexType factorId = gm_.factorOfVariable(i,factor);
				for( IndexType factorNode=0; factorNode<gm_.numberOfVariables( factorId ); factorNode++ ) 
					if( persistencyLoc[ gm_.variableOfFactor(factorId,factorNode) ] == nodePos::outside )
						has_outside_neighbor = true;
			}
			if( has_outside_neighbor )
				persistencyLoc[i] = nodePos::boundary;
		}
   }
   return newOutside;
}

// build redGm, which is part of gm_ with adjusted boundary factors and no outside factors
template<class DATA, class ACC, template <typename,typename> class SOLVER>
void 
PBP<DATA,ACC,SOLVER>::buildRedGm(
      PersistencyGMType& redGm,
      const std::vector<LabelType>& initLabeling,
      const std::vector<nodePos::Type>& persistencyLoc)
{
   // we do not want to include outside nodes into the reduced graphical model. Hence build a table which translates nodes from reduced gm to original gm and vice versa
   std::vector<IndexType> nodeTranslationRedToOrig;
   std::vector<IndexType> nodeTranslationOrigToRed;
   nodeTranslationOrigToRed.resize( gm_.numberOfVariables(), -1 );
   for(size_t i=0; i<n_; i++) {
      if( persistencyLoc[i] != nodePos::outside ) {
         nodeTranslationRedToOrig.push_back(i);
         nodeTranslationOrigToRed[i] = nodeTranslationRedToOrig.size()-1;
      }
   }

   for(size_t i=0; i<nodeTranslationRedToOrig.size(); i++) {
      OPENGM_ASSERT(i == nodeTranslationOrigToRed[ nodeTranslationRedToOrig[i] ]);
   }

   // add nodes to redGm_
   std::vector<size_t> numLabelsRed(nodeTranslationRedToOrig.size());
   for(size_t iRed=0; iRed<nodeTranslationRedToOrig.size(); iRed++) {
      numLabelsRed[iRed] = gm_.numberOfLabels(nodeTranslationRedToOrig[iRed]);        
   }
   redGm = PersistencyGMType( opengm::DiscreteSpace<>(numLabelsRed.begin(), numLabelsRed.end() ));

   // build vector of factor indices of gm_
   std::vector< std::vector<LabelType> > factorIndices(gm_.numberOfFactors()); 
   for(size_t factorId=0; factorId<gm_.numberOfFactors(); ++factorId)
      for(size_t varId=0; varId < gm_.numberOfVariables(factorId); ++varId)
         factorIndices[factorId].push_back( gm_.variableOfFactor(factorId,varId) );

   // factor indices for nodes in redGm
   std::vector< std::vector<LabelType> > redFactorIndices(gm_.numberOfFactors()); 
   for(size_t factorId=0; factorId<gm_.numberOfFactors(); ++factorId)
      for(size_t varId=0; varId < gm_.numberOfVariables(factorId); ++varId)
         if(persistencyLoc[factorIndices[factorId][varId]] != nodePos::outside)
            redFactorIndices[factorId].push_back( nodeTranslationOrigToRed[ factorIndices[factorId][varId] ] ); 

   // factor indices of boundary factors
   std::vector< std::vector<IndexType> > boundaryFactorIndices;

   // functions for accumulating inside and boundary factors sharing the same support, needed for TRWS, which cannot handle duplicate unary factors
   std::vector<SumFunctionType> boundaryFunc;
   for(size_t factorId=0; factorId<gm_.numberOfFactors(); ++factorId) {
      if(isBoundaryFactor(factorId,persistencyLoc)) {
         const size_t boundaryFuncIndex = BoundaryFuncIndex(boundaryFactorIndices,redFactorIndices[factorId],factorIndices[factorId],persistencyLoc);
         if(boundaryFuncIndex == -1) {
            std::vector<LabelType> boundaryShape;
            for(size_t i=0; i<redFactorIndices[factorId].size(); i++) {
               boundaryShape.push_back(redGm.numberOfLabels( redFactorIndices[factorId][i] ));
            }
            boundaryFunc.push_back(SumFunctionType(boundaryShape));
            boundaryFactorIndices.push_back(redFactorIndices[factorId]);
            OPENGM_ASSERT(BoundaryFuncIndex(boundaryFactorIndices,redFactorIndices[factorId],factorIndices[factorId],persistencyLoc) == boundaryFactorIndices.size() - 1);
         }
      }
   }

   // add factors to redGm
   for(size_t factorId=0; factorId<gm_.numberOfFactors(); ++factorId) {
      const size_t boundaryFuncIdx = BoundaryFuncIndex(boundaryFactorIndices,redFactorIndices[factorId],factorIndices[factorId],persistencyLoc);
      if( isInsideFactor(factorId,persistencyLoc) ) { // Attention: this means that factor support is in inside *or* in boundary nodes!
         if( boundaryFuncIdx == -1 ) {
            OPENGM_ASSERT(redFactorIndices[factorId].size() == factorIndices[factorId].size());
            // add a view function to the original function
            const ViewFunctionType function( gm_[factorId] );
            const FunctionIdentifierType funcId = redGm.addFunction( function );
            // unsure if this is the right way (memory leak?). However program crashes otherwise
            std::vector<LabelType> * factorIndicesRed = new std::vector<LabelType>(redFactorIndices[factorId]);
            redGm.addFactor( funcId, factorIndicesRed->begin(), factorIndicesRed->end());
         } else { // accumulate function into boundaryFunc
            marray::Marray<ValueType> f(gm_[factorId].shapeBegin(), gm_[factorId].shapeEnd());
            gm_[factorId].copyValues(f.begin());
            boundaryFunc[boundaryFuncIdx].AddFunction( f );
         }
      } else if( isBoundaryFactor(factorId,persistencyLoc) ) { // means factor support contains boundary and outside nodes
         OPENGM_ASSERT(redFactorIndices[factorId].size() < factorIndices[factorId].size());
         OPENGM_ASSERT(boundaryFuncIdx != -1);
         const PersPotentialType function( gm_, factorId, persistencyLoc, initLabeling);
         std::vector<LabelType> shape;
         for(size_t i=0; i<function.dimension(); i++)
            shape.push_back(function.shape(i));
         marray::Marray<ValueType> f(shape.begin(), shape.end());

         // copy values from function into f, can possibly be better implemented with ShapeWalker.
         std::vector<LabelType> shapeIterator(function.dimension());
         for(size_t i=0; i<f.size(); i++) {
            size_t i_tmp = i;
            for(size_t j=0; j<function.dimension(); j++) {
               shapeIterator[j] = i_tmp % function.shape(j);
               i_tmp = i_tmp/function.shape(j);
            }
            f(i) = function(shapeIterator.begin());
            OPENGM_ASSERT(function(shapeIterator.begin()) == f(shapeIterator.begin()));
         }

         boundaryFunc[boundaryFuncIdx].AddFunction(f);

         // do zrobienia: skasuj
         //const size_t boundaryNode = persistencyLoc[gm_.variableOfFactor(factorId,0)] == nodePos::boundary ?
         //   gm_.variableOfFactor(factorId,0) : gm_.variableOfFactor(factorId,1);
         //std::cout << initLabeling[boundaryNode] << ": ";
         //for(size_t i=0; i<f.size(); i++)
         //   std::cout << f(i) << ", ";
         //std::cout << std::endl;


         // do zrobienia: skasuj
         //const FunctionIdentifierType funcId = redGm.addFunction( function );
         //std::vector<LabelType> * factorIndicesRed = new std::vector<LabelType>(redFactorIndices[factorId]);
         //redGm.addFactor( funcId, factorIndicesRed->begin(), factorIndicesRed->end());
      }
   }

   cout << "Add boundary factors ...";
   // add boundary factors 
   for(size_t i=0; i<boundaryFunc.size(); i++) {
      const FunctionIdentifier fid = redGm.addFunction(boundaryFunc[i]);
      const size_t factorId = redGm.addFactor(fid, boundaryFactorIndices[i].begin(), boundaryFactorIndices[i].end());

      // do zrobienia: skasuj
      //std::cout << initLabeling[ nodeTranslationRedToOrig[boundaryFactorIndices[i][0]] ] << ": ";
      //for(size_t x_i=0; x_i<boundaryFunc[i].size(); x_i++)
      //   std::cout << boundaryFunc[i].operator()(&x_i) << ", ";
      //std::cout << std::endl;
   }
   std::cout << "done." << std::endl;

   {
      // important for TRWS
      size_t noUnaryFactors = 0;
      for(size_t factorId=0; factorId<redGm.numberOfFactors(); factorId++)
         if(redGm.numberOfVariables(factorId) == 1)
            noUnaryFactors++;
      std::cout << "Number unary factors = " << noUnaryFactors << ", number of variables = " << redGm.numberOfVariables() << std::endl;
      //OPENGM_ASSERT(redGm.numberOfVariables() == noUnaryFactors);
   }
}

template<class DATA, class ACC, template <typename,typename> class SOLVER>
bool 
PBP<DATA,ACC,SOLVER>::isInsideFactor( 
      const size_t factorId,
      const std::vector<nodePos::Type>& persistencyLoc) const
{
	for(size_t node=0; node<gm_.numberOfVariables(factorId); node++)
		if( persistencyLoc[ gm_.variableOfFactor(factorId,node) ] == nodePos::outside )
			return false;
	return true;
}

template<class DATA, class ACC, template <typename,typename> class SOLVER>
bool
PBP<DATA,ACC,SOLVER>::isBoundaryFactor( 
      const size_t factorId,
      const std::vector<nodePos::Type>& persistencyLoc ) const
{
	bool outside = false;
	bool not_outside= false;
	for(size_t node=0; node<gm_.numberOfVariables(factorId); node++)
		if( persistencyLoc[ gm_.variableOfFactor(factorId,node) ] == nodePos::outside )
			outside = true;
		else
			not_outside = true;

	return outside && not_outside;
}

template<class DATA, class ACC, template <typename,typename> class SOLVER>
typename DATA::GraphicalModelType::IndexType
PBP<DATA,ACC,SOLVER>::BoundaryFuncIndex(
      const std::vector<std::vector<IndexType> >& boundaryFactorIndices,
      const std::vector<IndexType>& redFactorIndices,
      const std::vector<IndexType>& origFactorIndices,
      const std::vector<nodePos::Type>& persistencyLoc) const
{
   bool only_inside_nodes = true;
   bool only_outside_nodes = true;
   for(size_t i=0; i<origFactorIndices.size(); i++) {
      if(persistencyLoc[origFactorIndices[i]] != nodePos::inside)
         only_inside_nodes = false;
      if(persistencyLoc[origFactorIndices[i]] != nodePos::outside)
         only_outside_nodes = false;
   }
   if(only_inside_nodes)
      return -1;
   if(only_outside_nodes)
      return -1;

   for(size_t i=0; i<boundaryFactorIndices.size(); i++)
      if(redFactorIndices.size() == boundaryFactorIndices[i].size())
         if(std::equal(redFactorIndices.begin(), redFactorIndices.end(), boundaryFactorIndices[i].begin()))
            return i;
   return -1;
}

// solve connected components separately and reassemble solution
template<class DATA, class ACC, template <typename,typename> class SOLVER>
std::pair<typename DATA::GraphicalModelType::ValueType,typename DATA::GraphicalModelType::ValueType>
PBP<DATA,ACC,SOLVER>::solveGM(
      const PersistencyGMType& gm, 
      std::vector<LabelType>& l,
      std::vector<bool>& consistent)
{
   std::pair<ValueType,ValueType> bounds(0.0,0.0);
   typedef GraphicalModel<ValueType, OperatorType, OPENGM_TYPELIST_2(ExplicitFunctionType, ViewFunction<PersistencyGMType>) > RedGmTypeV;

   l.resize(gm.numberOfVariables());
   consistent.resize(gm.numberOfVariables());

   if(gm_.factorOrder() <= 2) {
      std::vector<std::vector<typename RedGmTypeV::IndexType> > cc2gm;
      std::vector<RedGmTypeV> cc;
      std::vector<std::vector<bool> > ccc; // consistency in connected components
      std::vector<std::vector<LabelType> > lcc; // labeling in connected components

      std::cout << "Computed connected components" << std::endl;
      getConnectComp<PersistencyGMType, RedGmTypeV>(gm, cc2gm, cc);
      std::cout << "done" << std::endl;
      lcc.resize(cc2gm.size());
      ccc.resize(cc2gm.size());


      if(cc2gm.size() > 1) {
         for( IndexType c=0; c<cc2gm.size(); c++ ) {
            if( cc2gm[c].size() == 1 ) { // for TRWSi, which cannot handle isolated nodes!
               std::cout << "Solving component " << c << " with 1 node: " << cc2gm[c][0] << std::endl;
               ccc[c].resize(1);
               ccc[c][0] = true;
               OPENGM_ASSERT(cc[c].numberOfFactors() == 1);
               std::vector<IndexType> idx(1,0);
               ValueType v = cc[c][0].operator()(idx.begin());
               IndexType min_idx = 0;
               for( int i=0; i<cc[c].numberOfLabels(0); i++ ) {
                  idx[0] = i;
                  if( v > min(v,cc[c][0].operator()(idx.begin())) ) {
                     min_idx = i;
                     v = cc[c][0].operator()(idx.begin());
                  }
               }
               lcc[c].resize(1);
               lcc[c][0] = min_idx;

               bounds.first += cc[c][0].operator()(&min_idx);
               bounds.second += cc[c][0].operator()(&min_idx);
            } else {
               std::cout << "Solving component " << c << " with " << cc[c].numberOfVariables() << " nodes." << std::endl;
               SOLVER<RedGmTypeV,ACC> solver(cc[c]);
               solver.infer();
               solver.arg(lcc[c]);
               solver.consistent(ccc[c]);

               bounds.first += solver.bound();
               bounds.second += solver.graphicalModel().evaluate(lcc[c].begin());
            }
         }

         std::cout << "Solved all components" << std::endl;

         reassembleFromCC<PersistencyGMType,IndexType>(l, lcc, cc2gm);
         reassembleFromCC<PersistencyGMType,bool>(consistent, ccc, cc2gm);

         std::cout << "reassembled all components" << std::endl;
      } else { // single component
         std::cout << "Solving single component" << std::endl;
         //store_into_explicit(gm,std::string("test.h5"));
         IterSolverType solver(gm);
         solver.infer();
         solver.arg(l);
         solver.consistent(consistent);
         bounds.first = solver.bound();
         bounds.second = gm.evaluate(l.begin());
         std::cout << "Solving single component ... done" << std::endl;
      }
   } else { // higher order
      IterSolverType solver(gm);
      solver.infer();
      solver.arg(l);
      solver.consistent(consistent);
      bounds.first = solver.bound();
      bounds.second = gm.evaluate(l.begin());
   }

   /*
   OPENGM_ASSERT(std::abs(bounds.second - gm.evaluate(l)) < eps_);
   // do zrobienia: skasuj
   {
      IterSolverType solver(gm);
      solver.infer();
      std::vector<LabelType> l_test;
      solver.arg(l_test);
      //solver.consistent(consistent);
      OPENGM_ASSERT(l.size() == l_test.size());
      OPENGM_ASSERT(std::abs(bounds.first - solver.bound()) < eps_);
      OPENGM_ASSERT(std::equal(l.begin(),l.end(), l_test.begin()));
   }
   */

   return bounds;
}

template<class DATA, class ACC, template <typename,typename> class SOLVER>
bool 
PBP<DATA,ACC,SOLVER>::PersistencyCandidatesExist(const std::vector<nodePos::Type>& persistencyLoc) const
{
   for(size_t i=0; i<persistencyLoc.size(); i++)
      if(persistencyLoc[i] != nodePos::outside)
         return true;
   return false;
}

template<class DATA, class ACC, template <typename,typename> class SOLVER>
std::vector<typename DATA::GraphicalModelType::LabelType> 
PBP<DATA,ACC,SOLVER>::SubToOrigLabeling(
      const std::vector<LabelType>& initLabeling,
      const std::vector<LabelType>& curLabeling,
      const std::vector<nodePos::Type>& persistencyLoc) const
{
   std::vector<LabelType> curLabelingOrig(initLabeling);
   size_t iRed=0;
   for(size_t i=0; i<n_; i++) {
      if(persistencyLoc[i] != nodePos::outside) {
         OPENGM_ASSERT(iRed < curLabeling.size());
         curLabelingOrig[i] = curLabeling[iRed];
         iRed++;
      }
   }
   return curLabelingOrig;
}

template<class DATA, class ACC, template <typename,typename> class SOLVER>
std::vector<bool> 
PBP<DATA,ACC,SOLVER>::SubToOrigConsistent(
      const std::vector<bool>& consistent,
      const std::vector<nodePos::Type>& persistencyLoc) const
{
   std::vector<bool> consistentOrig(n_,false);
   size_t iRed=0;
   for(size_t i=0; i<n_; i++) {
      if(persistencyLoc[i] != nodePos::outside) {
         OPENGM_ASSERT(iRed < consistent.size());
         consistentOrig[i] = consistent[iRed];
         iRed++;
      }
   }
   return consistentOrig;
}

} // namespace PBP
} // namespace opengm

#endif // OPENGM_PBP_HXX
