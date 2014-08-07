#pragma once
#ifndef OPENGM_IRI_POTENTIAL_PERM_HXX
#define OPENGM_IRI_POTENTIAL_PERM_HXX

//#define OPENGM_IRI_SUBMODULARIZATION // only makes sense for subset to one maps

#include <iostream>
#include <iterator>

#include "opengm/opengm.hxx"
#include "opengm/functions/function_properties_base.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/graphicalmodel_factor.hxx"

namespace opengm {
namespace IRI {

/// Function that uses a factor of another GraphicalModel and a labeling proposal
/// described in notes on partial optimality on label level
/// as base for a modified potential in persistency by pruning
/// Improving mapping is only allowed to be a permutation, not a general projection
///
/// \ingroup functions
/// \ingroup view_functions
template<class GM>
class Potential
: public FunctionBase<Potential<GM>,
            typename GM::ValueType,
            typename GM::IndexType,
            typename GM::LabelType>
{
public:
   typedef GM GraphicalModelType;
   typedef typename GM::ValueType ValueType;
   typedef typename GM::LabelType LabelType;
   typedef typename GM::FactorType FactorType;

   Potential(const FactorType& fac, std::vector<std::vector<LabelType> > im); 
   template<class Iterator> ValueType operator()(Iterator begin) const;
   size_t size() const {return fac_->size();}; 
   LabelType shape(const size_t i) const {return fac_->shape(i);}
   size_t dimension() const {return fac_->dimension();}

   void ComputeReducedPotential(const std::vector<std::vector<LabelType> >& im);
   template<class Iterator> ValueType OrigPotentialValue(Iterator begin) const;
   void UpdateImprovingMapping(const std::vector<std::vector<LabelType> >& im); 

   bool operator==(const Potential& ) const {throw; return false;} // do zrobienia

private:
   const FactorType* fac_;
   std::vector<std::vector<LabelType> > im_;
   std::vector<LabelType> origIt, permIt;
#ifdef OPENGM_IRI_SUBMODULARIZATION
   std::vector<ValueType> Delta1, Delta2;
#endif
};

template<class GM>
Potential<GM>::Potential
( 
 const FactorType& fac,
 const std::vector<std::vector<LabelType> > im
) :
   fac_(&fac),
   im_(im)
{
   origIt.resize(dimension());
   permIt.resize(dimension());

   OPENGM_ASSERT(im_.size() == dimension());
   for(size_t v=0; v<dimension(); v++)
      OPENGM_ASSERT(im_[v].size() == shape(v));

   ComputeReducedPotential(im_);
}

template <class GM>
void
Potential<GM>::ComputeReducedPotential(const std::vector<std::vector<typename GM::LabelType> >& im)
{
#ifdef OPENGM_IRI_SUBMODULARIZATION
   OPENGM_ASSERT(dimension() <= 2);
   if(dimension() == 2) {
      Delta1.resize(im[0].size(),std::numeric_limits<LabelType>::max());
      Delta2.resize(im[1].size(),std::numeric_limits<LabelType>::max());
      for(size_t i=0; i<im[0].size(); i++) {
         for(size_t j=0; j<im[1].size(); j++) {
            std::vector<LabelType> it(2);
            it[0] = i; 
            it[1] = j;
            if(im[1][j] == j) // j \in U_t
               Delta1[i] = std::min(Delta1[i],OrigPotentialValue(it.begin()));
            if(im[0][i] == i) // i \in U_s
               Delta2[j] = std::min(Delta2[j],OrigPotentialValue(it.begin()));
         }
      }
      for(size_t i=0; i<im[0].size(); i++) {
         OPENGM_ASSERT(Delta1[i] < std::numeric_limits<LabelType>::max());
         if(Delta1[i] == std::numeric_limits<LabelType>::max()) throw;
      }
      for(size_t j=0; j<im[1].size(); j++) {
         OPENGM_ASSERT(Delta2[j] < std::numeric_limits<LabelType>::max());
         if(Delta2[j] == std::numeric_limits<LabelType>::max()) throw;
      }
   }
#endif
}

template<class GM>
template<class Iterator> 
typename GM::ValueType
Potential<GM>::OrigPotentialValue(Iterator begin) const
{
   // due to const-ness cannot use origIt and permIt directly. However for performance reasons I do not want to allocate Iterators so often but hold memory ready.
   LabelType* origItTmp = const_cast<LabelType *>(origIt.data());
   LabelType* permItTmp = const_cast<LabelType *>(permIt.data());

   // construct second iterator which will permute by im_
   for(size_t v=0; v<dimension(); v++) {
      origItTmp[v] = *begin;
      permItTmp[v] = im_[v][*begin];
      begin++;
   }

   return fac_->operator()(origItTmp) - fac_->operator()(permItTmp); 
}

template<class GM>
template<class Iterator> 
inline typename GM::ValueType
Potential<GM>::operator()(Iterator begin) const 
{
   ValueType g = OrigPotentialValue(begin);
#ifdef OPENGM_IRI_SUBMODULARIZATION
   if(dimension()==2) {
      size_t i = *begin;
      size_t j = *(++begin);
      //std::cout << "i = " << i << ", j = " << j << ", Delta1[i] = " << Delta1[i] << ", Delta2[j] = " << Delta2[j] << std::endl;
      if(im_[0][i] == i && im_[1][j] == j)      // i \in U_s, j \in U_t
         return 0.0;
      else if(im_[0][i] == i && im_[1][j] != j) // i \in U_s, j \notin U_t
         return Delta2[j];
      else if(im_[0][i] != i && im_[1][j] == j) // i \notin U_s, j \in U_t
         return Delta1[i];
      else if(im_[0][i] != i && im_[1][j] != j) // i \notin U_s, j \notin U_t
         return std::min(Delta1[i] + Delta2[j], g);
      else
         throw;
   } else {
      return g;
   }
#else  
   return g;
#endif
}

template<class GM>
void 
Potential<GM>::UpdateImprovingMapping(const std::vector<std::vector<LabelType> >& im)
{
   im_ = im; 
   ComputeReducedPotential(im);
   OPENGM_ASSERT(im.size() == im_.size());
   /*
   for(size_t v=0; v<im.size(); v++) {
      OPENGM_ASSERT(im[v].size() == im_[v].size());
      for(size_t i=0; i<im[v].size(); i++) {
         if(im[v][i] != im_[v][i]) {
            im_ = im; 
            ComputeReducedPotential(im);
            return;
         }
      }
   }*/
}

} // namespace IRI

template <class GM>
struct FunctionRegistration< IRI::Potential<GM> >{
	/// Id  of Potential
	enum ID {
		Id = opengm::FUNCTION_TYPE_ID_OFFSET - 1791
	};
};

} // end namespace opengm

#endif

