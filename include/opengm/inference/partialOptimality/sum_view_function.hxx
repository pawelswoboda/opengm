#pragma once
#ifndef OPENGM_SUM_VIEW_FUNCTION_HXX
#define OPENGM_SUM_VIEW_FUNCTION_HXX

#include <limits>
#include <iostream>

#include "opengm/functions/function_properties_base.hxx"
#include "opengm/functions/function_registration.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/graphicalmodel_factor.hxx"

namespace opengm {

/// Function that holds a list of other functions, whose values are summed up
/// needed for pbp, as trws cannot handle multiple factors for the same variable
///
/// \tparam GM type of the graphical model which we want to view
///
/// \ingroup functions
/// \ingroup view_functions
template<class GM>
class SumViewFunction
: public FunctionBase<SumViewFunction<GM>,
            typename GM::ValueType,
            typename GM::IndexType,
            typename GM::LabelType>
{
public:
   typedef GM GraphicalModelType;
   typedef typename GM::ValueType          ValueType;
   typedef typename GM::IndexType          IndexType;
   typedef typename GM::LabelType          LabelType;

   SumViewFunction(const std::vector<LabelType>& boundaryShape);

   template<class Iterator> ValueType operator()(Iterator begin) const;
   size_t size() const;
   LabelType shape(const size_t i) const;
   size_t dimension() const;
   bool operator==(const SumViewFunction& ) const;

   void AddFunction(const marray::Marray<ValueType>& f); 

private:
   std::vector<LabelType> shape_;
   marray::Marray<ValueType> function_; 
};

template <class GM>
struct FunctionRegistration< SumViewFunction<GM> >{
	/// Id  of SumViewFunction
	enum ID {
		Id = opengm::FUNCTION_TYPE_ID_OFFSET - 1772
	};
};

/// Constructor
/// \param shape of the factor
/// \param 
template<class GM>
SumViewFunction<GM>::SumViewFunction
(
 const std::vector<LabelType> & shape
)
:  shape_(shape),
   function_(shape.begin(), shape.end(), 0.0)
{
   OPENGM_ASSERT(shape.size() >= 1);
   OPENGM_ASSERT(function_.size() == size());
}

template<class GM>
template<class Iterator>
inline typename opengm::SumViewFunction<GM>::ValueType
SumViewFunction<GM>::operator()( Iterator begin ) const
{
   //OPENGM_ASSERT(dimension() == 1);
   return function_(*begin);//, begin+shape_.size());
}

template<class GM>
inline typename SumViewFunction<GM>::LabelType
SumViewFunction<GM>::shape(const size_t i) const
{
	return shape_[i];
}

template<class GM>
inline size_t SumViewFunction<GM>::size() const
{
	int size = 1;
	for(int i=0; i<shape_.size(); i++) 
		size *= shape_[i]; 
	return size;
}

template<class GM>
inline size_t SumViewFunction<GM>::dimension() const
{
	return shape_.size();
}


template<class GM>
void SumViewFunction<GM>::AddFunction(const marray::Marray<ValueType>& f) 
{ 
   OPENGM_ASSERT(f.size() == size());
   for(size_t i=0; i<function_.size(); i++) {
      function_(i) = function_(i) + f(i);
   }
   //function_ += f;
}


template<class GM>
inline bool
SumViewFunction<GM>::operator==
(
 const SumViewFunction & fb
 )const{
	throw RuntimeError("== not implemented yet in SumViewFunction");
	return true; 
}

} // namespace opengm

#endif // OPENGM_SUM_VIEW_FUNCTION_HXX

