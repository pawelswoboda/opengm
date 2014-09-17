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

#include <vector>
#include <limits>
#include <queue>
#include <ctime>

namespace opengm {

namespace PBP {

//! [class PBP partial optimality by pruning]
/// Persistency by pruning, published in CVPR 2014 as 
/// "Partial Optimality by Pruning for MAP-inference with General Graphical Models"
///
/// Corresponding author: Paul Swoboda
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
   typedef GraphicalModel<ValueType,opengm::Adder,
           OPENGM_TYPELIST_3(ExplicitFunctionType,ViewFunctionType,PersPotentialType),
           opengm::DiscreteSpace<IndexType,LabelType> > PersistencyGMType;

   typedef typename PersistencyGMType::ConstFactorIterator PGMConstFactorIterator;
   typedef typename PersistencyGMType::ConstVariableIterator PGMConstVariableIterator;

   typedef SOLVER<GM,ACC> InitSolverType;
   typedef SOLVER<PersistencyGMType,ACC> IterSolverType;


   PBP(DATA& d);
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
   IndexType updateInterior(
         const std::vector<LabelType>& initLabeling, 
         const std::vector<LabelType>& curLabeling,
         const std::vector<bool>& consistent,
         std::vector<nodePos::Type>& persistencyLoc);
   void solveGM(
         const PersistencyGMType& gm, 
         std::vector<LabelType>& l,
         std::vector<bool>& c);


   std::vector<LabelType> SubToOrigLabeling(const std::vector<LabelType>& initLabeling, const std::vector<LabelType>& curLabeling, const std::vector<nodePos::Type>& persistencyLoc) const;
   std::vector<bool> SubToOrigConsistent(const std::vector<bool>& consistent, const std::vector<nodePos::Type>& persistencyLoc) const;

};

template<class DATA,class ACC,template <typename,typename> class SOLVER>
const double PBP<DATA,ACC,SOLVER>::eps_ = 1.0e-6;

template<class DATA,class ACC,template <typename,typename> class SOLVER>
PBP<DATA,ACC,SOLVER>::PBP(DATA& d) :
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
         std::cout << "Globally optimal solution found by relaxation" << std::endl;
         for(size_t i=0; i<n_; i++)
            d_.setTrue(i,initLabeling[i]);

         return NORMAL;
      } else {
         solver.consistent(consistent);
      }
   }

   curLabeling = initLabeling;

   size_t iterations = 1;
   while( updateInterior(initLabeling,curLabeling,consistent,persistencyLoc) > 0 ) { // shrink interior by pruning variables

      PersistencyGMType redGm;
      buildRedGm(redGm,initLabeling,persistencyLoc);

      IterSolverType solver(redGm);
	   InferenceTermination status = solver.infer();
	   if( status != NORMAL ) {
         std::cout << "Solver did not converge" << std::endl;
         return status;
      }
      solveGM(redGm,curLabeling,consistent);
      //solver.arg(curLabeling);
      //solver.consistent(consistent);

      iterations++;
   }

   std::cout << "Found persistent set after " << iterations << " iterations." << endl;
   IndexType p=0;
   for(size_t i=0; i<n_; i++) {
      if(persistencyLoc[i] != nodePos::outside) {
         d_.setTrue(i,initLabeling[i]);
         p++;
      }
   }
   std::cout << "Percentage partial optimality = " << double(p)/double(n_) << std::endl;
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
   IndexType newOutside = 0;
   std::vector<LabelType> curLabelingOrig = SubToOrigLabeling(initLabeling,curLabeling,persistencyLoc);
   std::vector<bool> consistentOrig = SubToOrigConsistent(consistent,persistencyLoc);
   
   // determine new outside nodes
   for(size_t i=0; i<n_; i++) {
      if(curLabelingOrig[i] != initLabeling[i] || !consistentOrig[i]) {
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
      const std::vector<nodePos::Type>& persistencyLoc )
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

   // vector of unary potentials for redGm_: original unaries and boundary factors
   std::vector<std::vector<ValueType> > unariesRed(redGm.numberOfVariables());
   for(size_t varIdRed=0; varIdRed<redGm.numberOfVariables(); ++varIdRed){
      unariesRed[varIdRed].resize( redGm.numberOfLabels(varIdRed), 0.0 );        
   }

   // add pairwise Factors to redGm_, accumulate unary factors
	for(size_t factorId=0; factorId<gm_.numberOfFactors(); ++factorId) {
		if( isInsideFactor(factorId,persistencyLoc) ) {
			if( gm_[factorId].numberOfVariables() !=1 ) {
				const ViewFunctionType function( gm_[factorId] );
				const FunctionIdentifierType funcId = redGm.addFunction( function );
				// unsure if this is the right way (memory leak?). However program crashes otherwise
				std::vector<LabelType> * factorIndicesRed = new std::vector<LabelType>(factorIndices[factorId].size());
				for( IndexType i=0; i<factorIndices[factorId].size(); i++ )
					factorIndicesRed->operator[](i) = nodeTranslationOrigToRed[ factorIndices[factorId][i] ];
				redGm.addFactor( funcId, factorIndicesRed->begin(), factorIndicesRed->end());
			} else { // unary factor
				size_t varIdRed =  nodeTranslationOrigToRed[factorIndices[factorId][0]];
				for( int l=0; l<redGm.numberOfLabels(varIdRed); l++ ) {
					std::vector<LabelType> lVec(1);
					lVec[0] = l;
					unariesRed[varIdRed][l] += gm_[factorId](lVec.begin());
				}
			}
		} else if( isBoundaryFactor(factorId,persistencyLoc) ) { // currently only unary boundary factors supported
			const PersPotentialType function( gm_, factorId, persistencyLoc, initLabeling);
			std::vector<size_t> redFactor;
			for( std::vector<size_t>::iterator it = factorIndices[factorId].begin(); it != factorIndices[factorId].end(); it++ ) 
				if( persistencyLoc[*it] != nodePos::outside )
					redFactor.push_back(nodeTranslationOrigToRed[*it]);
			OPENGM_ASSERT( redFactor.size() == 1 ); // only for boundary factor size == 1
			int varIdRed = redFactor[0];

			for( int l=0; l<redGm.numberOfLabels(varIdRed); l++ ) {
				std::vector<LabelType> lVec(1);
				lVec[0] = l;
				unariesRed[varIdRed][l] += function(lVec.begin());
			}
		}
	}
	
	cout << "Add unary factors:" << endl;
	// add unary factors 
	for(size_t varIdRed=0; varIdRed<redGm.numberOfVariables(); ++varIdRed){
		const size_t shape[] = {redGm.numberOfLabels(varIdRed)};
		ExplicitFunctionType f(shape, shape + 1);
		for(size_t state = 0; state < redGm.numberOfLabels(varIdRed); ++state) {
			f(state) = unariesRed[varIdRed][state];
		}
		FunctionIdentifier fid = redGm.addFunction(f);
		size_t variableIndex[] = {varIdRed};
		redGm.addFactor(fid, variableIndex, variableIndex + 1);
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

// solve connected components separately and reassemble solution
template<class DATA, class ACC, template <typename,typename> class SOLVER>
void 
PBP<DATA,ACC,SOLVER>::solveGM(
      const PersistencyGMType& gm, 
      std::vector<LabelType>& l,
      std::vector<bool>& consistent)
{
   typedef GraphicalModel<ValueType, OperatorType, OPENGM_TYPELIST_2(ExplicitFunctionType, ViewFunction<PersistencyGMType>) > RedGmTypeV;

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
         if( cc2gm[c].size() == 1 ) { // TRWSi cannot handle isolated nodes!
            std::cout << "Solving component " << c << " with one node: " << cc2gm[c][0] << std::endl;
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
         } else {
            std::cout << "Solving component " << c << " with " << cc[c].numberOfVariables() << " nodes." << std::endl;
            SOLVER<RedGmTypeV,ACC> solver(cc[c]);
            solver.infer();
            solver.arg(lcc[c]);
            solver.consistent(ccc[c]);
         }
      }
      
      std::cout << "Solved all components" << std::endl;

      reassembleFromCC<PersistencyGMType,IndexType>(l, lcc, cc2gm);
      reassembleFromCC<PersistencyGMType,bool>(consistent, ccc, cc2gm);

      std::cout << "reassembled all components" << std::endl;
   } else {
      IterSolverType solver(gm);
      solver.infer();
      solver.arg(l);
      solver.consistent(consistent);
   }
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
         curLabelingOrig[i] = curLabeling[i];
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
         consistentOrig[i] = consistent[i];
         iRed++;
      }
   }
   return consistentOrig;
}












} // namespace PBP
} // namespace opengm

#endif // OPENGM_PBP_HXX
