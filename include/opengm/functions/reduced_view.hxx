#pragma once
#ifndef OPENGM_REDUCED_VIEW_FUNCTION_HXX
#define OPENGM_REDUCED_VIEW_FUNCTION_HXX

#include "opengm/functions/function_properties_base.hxx"

namespace opengm {

/// reference to a Factor of a GraphicalModel
///
/// \ingroup functions
template<class GM>
class ReducedViewFunction
: public FunctionBase<ReducedViewFunction<GM>,
    typename GM::ValueType,typename GM::IndexType, typename GM::LabelType>
{
public:
   typedef typename GM::ValueType ValueType;
   typedef ValueType value_type;
   typedef typename GM::FactorType FactorType;
   typedef typename GM::OperatorType OperatorType;
   typedef typename GM::IndexType IndexType;
   typedef typename GM::IndexType LabelType;

   ReducedViewFunction();
   ReducedViewFunction(const FactorType &, const std::vector<std::vector<opengm::Tribool> >& partialOptimality);
   template<class Iterator>
      ValueType operator()(Iterator begin)const;
   LabelType shape(const IndexType)const;
   IndexType dimension()const;
   IndexType size()const;

private:
   FactorType const* factor_;
   std::vector<std::vector<opengm::Tribool> > partialOptimality_;
};

template<class GM>
inline
ReducedViewFunction<GM>::ReducedViewFunction()
:  factor_(NULL)
{}

template<class GM>
inline
ReducedViewFunction<GM>::ReducedViewFunction
(
   const typename ReducedViewFunction<GM>::FactorType & factor,
   const std::vector<std::vector<opengm::Tribool> >& partialOptimality
)
:  factor_(&factor),
   partialOptimality_(&partialOptimality)
{}

template<class GM>
template<class Iterator>
inline typename ReducedViewFunction<GM>::ValueType
ReducedViewFunction<GM>::operator()
(
   Iterator begin
) const {
   LabelType label [begin.size()];

   for (IndexType v = 0; v < factor_->numberOfVariables(); v++)
   {
       IndexType variable = factor_->variableIndex(v);
       label[v] = 0;
       for (size_t i = 0; i < begin[v]+1; i++)
       {
            while(partialOptimality_[variable][label[v]] == opengm::Tribool::False)
                label[v]++;

            label[v]++;
       }
       label[v]--;
   }
   return factor_->operator()(label);
}

template<class GM>
inline typename ReducedViewFunction<GM>::LabelType
ReducedViewFunction<GM>::shape
(
   const typename ReducedViewFunction<GM>::IndexType index
) const{
    LabelType numLabels = 0;
            for(IndexType i = 0; i < factor_->numberOfLabels(index); i++)
            {
                if(!(partialOptimality_[factor_->variableIndex(index)][i] == opengm::Tribool::False))
                    numLabels++;
            }
   return numLabels;
}

template<class GM>
inline typename ReducedViewFunction<GM>::IndexType
ReducedViewFunction<GM>::dimension() const {
   return factor_->numberOfVariables();
}

template<class GM>
inline typename ReducedViewFunction<GM>::IndexType
ReducedViewFunction<GM>::size() const {
    IndexType factorSize = 1;

    for (IndexType i = 0; i < factor_->numberOfVariables(); i++ ){
        factorSize *= this.shape(i);
    }
   return factorSize;
}

} // namespace opengm

#endif // #ifndef OPENGM_REDUCED_VIEW_FUNCTION_HXX

