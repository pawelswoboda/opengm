#pragma once
#ifndef OPENGM_BOUNDARYVIEWFUNCTION_HXX
#define OPENGM_BOUNDARYVIEWFUNCTION_HXX

#define PBP_INVARIANT_TO_REPARAMETRIZATION

#include <limits>
#include <iostream>

#include "opengm/functions/function_properties_base.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/graphicalmodel_factor.hxx"

using namespace std;

namespace opengm {

// denotes if a node of a graphical model is inside, on the boundary or outside the active set
namespace nodePos {
	enum Type {
		inside,
		boundary,
		outside,
	};
}

namespace minMax {
	enum Type { min, max };
}

/// Function that uses a factor of another GraphicalModel and a boundary labelling
/// as base for a boundary potential in persistency by pruning
///
/// \tparam GM type of the graphical model which we want to view
///
/// \ingroup functions
/// \ingroup view_functions
template<class GM>
class BoundaryViewFunction
: public FunctionBase<BoundaryViewFunction<GM>,
            typename GM::ValueType,
            typename GM::IndexType,
            typename GM::LabelType>
{
public:
   typedef GM GraphicalModelType;
   typedef typename GM::ValueType ValueType;
   typedef typename GM::IndexType IndexType;
   typedef typename GM::LabelType LabelType;

   BoundaryViewFunction(const GraphicalModelType& gm, const IndexType factorIndex, const std::vector<nodePos::Type>& persistencyLoc, const std::vector<LabelType>& boundaryLabeling);

   template<class Iterator> ValueType operator()(Iterator begin) const;
   size_t size() const;
   LabelType shape(const size_t i) const;
   size_t dimension() const;
   bool operator==(const BoundaryViewFunction& ) const;

private:
   bool conformsToBoundaryConditions(std::vector<IndexType>&) const;
   ValueType boundaryCost(std::vector<IndexType>&) const;
   ValueType boundaryCost(std::vector<IndexType>&,minMax::Type) const;
   bool incrementIndex(std::vector<IndexType>&,const std::vector<IndexType>&) const;
   IndexType indexFromVector(std::vector<IndexType>&,const std::vector<IndexType>&) const;

   const GraphicalModelType * gm_;
   const std::vector<nodePos::Type> * persistencyLoc_;
   const std::vector<LabelType> * boundaryLabeling_;
   IndexType factorIndex_;

   std::vector<IndexType> outsideNodes_;
   std::vector<IndexType> insideNodes_;
   std::vector<LabelType> insideLabeling_;
   std::vector<ValueType> boundaryCost_;

};

template <class GM>
struct FunctionRegistration< BoundaryViewFunction<GM> >{
	/// Id  of BoundaryViewFunction
	enum ID {
		Id = opengm::FUNCTION_TYPE_ID_OFFSET - 1791
	};
};

/// Constructor
/// \param gm graphical model we want to view
/// \param factorIndex index of the factor of gm we want to view
/// \param persistencyLoc inside/outside information from active set of variables
/// \param boundaryLabeling Labeling of variables at border of active set
/// \param 
template<class GM>
inline BoundaryViewFunction<GM>::BoundaryViewFunction
(
   const GraphicalModelType & gm,
   const IndexType factorIndex,
   const std::vector<nodePos::Type> & persistencyLoc, 
   const std::vector<LabelType> & boundaryLabeling
)
:  gm_(&gm),
   persistencyLoc_(&persistencyLoc),
   boundaryLabeling_(&boundaryLabeling),
   factorIndex_(factorIndex)
{
	for(IndexType node=0; node<gm.numberOfVariables(factorIndex); node++) {
		if( persistencyLoc[ gm.variableOfFactor(factorIndex,node) ] == nodePos::outside ) {
			outsideNodes_.push_back(node);
		} else {
			insideNodes_.push_back(node);
			insideLabeling_.push_back(boundaryLabeling[ gm.variableOfFactor(factorIndex,node) ]);
		}
	}

	boundaryCost_.resize(size());
	std::vector<IndexType> indexVec(insideNodes_.size(),0);
	//cout << "Boundary label(s): " << flush;
	//for( int i=0; i< insideLabeling_.size(); i++ ) {
	//	cout << insideLabeling_[i] << ", ";
	//}
	//cout << endl;
	//cout << "Boundary Cost: " << flush;
	do {
		boundaryCost_[indexFromVector(indexVec,insideNodes_)] = boundaryCost(indexVec);
		//cout << boundaryCost_[indexFromVector(indexVec,insideNodes_)] << ", " << flush;
	} while( incrementIndex(indexVec,insideNodes_) );
	//cout << "Boundary Cost computed" << endl;
}

template<class GM>
inline bool BoundaryViewFunction<GM>::conformsToBoundaryConditions
(
   std::vector<IndexType>& insideLabeling
) const
{
	for(IndexType i=0; i<dimension(); i++)
		//if(insidelabeling[i] != boundaryLabeling_->operator[](i))
		if(insideLabeling[i] != insideLabeling_[i])
			return false;
	return true;
}

template<class GM>
inline bool BoundaryViewFunction<GM>::incrementIndex
(
   std::vector<IndexType>& indexVector,
   const std::vector<IndexType>& nodes
) const
{
   OPENGM_ASSERT( indexVector.size() == nodes.size() );
   //std::cout << gm_->numberOfLabels(gm_->variableOfFactor(factorIndex_,nodes[0])) << ", " << indexVector[0] << std::endl;
   for(IndexType node=0; node<nodes.size(); node++) {
      if( indexVector[node] < gm_->numberOfLabels(gm_->variableOfFactor(factorIndex_,nodes[node])) -1 ) { //daj tutaj shape?
         indexVector[node]++;
         for(IndexType nodePrev = 0; nodePrev<node; nodePrev++)
            indexVector[nodePrev] = 0;
         return true;
      }
   }
   return false;
}

template<class GM>
inline typename opengm::BoundaryViewFunction<GM>::IndexType
BoundaryViewFunction<GM>::indexFromVector
(
   std::vector<IndexType>& begin,
   const std::vector<IndexType>& nodes
) const
{
	IndexType idx=0;
	IndexType offset = 1;
	for(IndexType i=0; i<nodes.size(); i++) {
		idx += begin[i]*offset;
		offset *= gm_->operator[](factorIndex_).shape(nodes[i]);
	}
	OPENGM_ASSERT(idx < size());
	return idx;
}

template<class GM>
inline typename opengm::BoundaryViewFunction<GM>::ValueType 
BoundaryViewFunction<GM>::boundaryCost 
(
   std::vector<IndexType>& insideIndexVec,
   minMax::Type m
) const
{
#ifndef PBP_INVARIANT_TO_REPARAMETRIZATION
	ValueType val = (m == minMax::max ? -std::numeric_limits<ValueType>::infinity() : std::numeric_limits<ValueType>::infinity());
	std::vector<IndexType> fullLabelIndex(gm_->operator[](factorIndex_).numberOfVariables(),0);
	std::vector<IndexType> outsideIndexVec(outsideNodes_.size(),0);
   do {
		for(IndexType node=0; node<insideNodes_.size(); node++)
			fullLabelIndex[insideNodes_[node]] = insideIndexVec[node];
		for(IndexType node=0; node<outsideNodes_.size(); node++)
			fullLabelIndex[outsideNodes_[node]] = outsideIndexVec[node];

		if(m==minMax::max)
			val = std::max(val, gm_->operator[](factorIndex_)(fullLabelIndex.begin()));
		else
			val = std::min(val, gm_->operator[](factorIndex_)(fullLabelIndex.begin()));
   } while( incrementIndex(outsideIndexVec,outsideNodes_) );
	return val;
#endif

#ifdef PBP_INVARIANT_TO_REPARAMETRIZATION
   OPENGM_ASSERT(insideLabeling_.size() == insideIndexVec.size());

   if(m == minMax::max) { // conforms to boundary conditions
      return 0;
   } else { // does not conform to boundary conditions
      ValueType val = std::numeric_limits<ValueType>::max();  
      std::vector<IndexType> fullLabelIndexCur(gm_->operator[](factorIndex_).numberOfVariables(),0); // inside with the current labeling
      std::vector<IndexType> fullLabelIndexCond(gm_->operator[](factorIndex_).numberOfVariables(),0); // inside with the boundary condition laeling
      std::vector<IndexType> outsideIndexVec(outsideNodes_.size(),0);
      do {
         for(IndexType node=0; node<insideNodes_.size(); node++) {
            fullLabelIndexCur[insideNodes_[node]] = insideIndexVec[node];
            fullLabelIndexCond[insideNodes_[node]] = insideLabeling_[node];
         }
         for(IndexType node=0; node<outsideNodes_.size(); node++) {
            fullLabelIndexCur[outsideNodes_[node]] = outsideIndexVec[node];
            fullLabelIndexCond[outsideNodes_[node]] = outsideIndexVec[node];
         }

         val = std::min( val, gm_->operator[](factorIndex_)(fullLabelIndexCur.begin()) - gm_->operator[](factorIndex_)(fullLabelIndexCond.begin()) );

      } while( incrementIndex(outsideIndexVec,outsideNodes_) );
      return val;
   }
#endif
}

template<class GM>
inline typename opengm::BoundaryViewFunction<GM>::ValueType
BoundaryViewFunction<GM>::boundaryCost
(
   std::vector<IndexType>& index
) const
{
	if( conformsToBoundaryConditions(index) ) {
		return boundaryCost(index, minMax::max);
	} else {
		return boundaryCost(index, minMax::min);
	}	
}

template<class GM>
template<class Iterator>
inline typename opengm::BoundaryViewFunction<GM>::ValueType
BoundaryViewFunction<GM>::operator()
(
   Iterator begin
) const
{
	std::vector<IndexType> indexVec(insideNodes_.size());
	IndexType i=0;
	for(IndexType i=0;i<insideNodes_.size();i++) {
		indexVec[i] = *begin;
		begin++;
	}
	return boundaryCost_[indexFromVector(indexVec,insideNodes_)];
}

template<class GM>
inline typename BoundaryViewFunction<GM>::LabelType
BoundaryViewFunction<GM>::shape(const size_t i) const
{
	OPENGM_ASSERT( i<insideNodes_.size() );
	return gm_->operator[](factorIndex_).shape( insideNodes_[i] );
}

template<class GM>
inline size_t BoundaryViewFunction<GM>::size() const
{
	size_t size = 1;
	for(int i=0; i<insideNodes_.size(); i++) 
		size *= gm_->numberOfLabels( gm_->variableOfFactor(factorIndex_,insideNodes_[i]) ); 
	return size;
}

template<class GM>
inline size_t BoundaryViewFunction<GM>::dimension() const
{
	return insideNodes_.size();
}


template<class GM>
inline bool
BoundaryViewFunction<GM>::operator==
(
 const BoundaryViewFunction & fb
 )const{
	throw;
	return true; // not implemented yet
}

} // namespace opengm

#endif // #ifndef OPENGM_BOUNDARYVIEWFUNCTION_HXX
