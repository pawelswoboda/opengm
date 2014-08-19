#pragma once
#ifndef OPENGM_POPT_DATA_HXX
#define OPENGM_POPT_DATA_HXX

#include <vector>
//#include <string>
//#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/utilities/metaprogramming.hxx>
#include <opengm/utilities/tribool.hxx>
//#include <opengm/functions/reduced_view.hxx>
#include <opengm/functions/explicit_function.hxx>


namespace opengm {
//! [class popt_data]
/// Partial Optimaility Data (-Container)
///
/// Corresponding author: Joerg Hendrik Kappes
///
///\ingroup inference
template<class GM>
class POpt_Data {
public:
    typedef GM GraphicalModelType;
    typedef GM GmType;
    OPENGM_GM_TYPE_TYPEDEFS;

//    typedef ReducedViewFunction<GraphicalModelType>                  ReducedViewType;
    typedef opengm::ExplicitFunction<ValueType, LabelType, IndexType> ExplicitFunction;
    typedef GraphicalModel<ValueType, OperatorType, ExplicitFunction> ReducedGmType;

    POpt_Data(const GmType&);
    // Set
    void                                   setTrue(const IndexType, const LabelType);
    void                                   setFalse(const IndexType, const LabelType);
    void                                   setFalse(const IndexType factor, const std::vector<LabelType>& labeling);
    // Get Partial Optimality
    Tribool                                getPOpt(const IndexType, const LabelType) const;
    std::vector<opengm::Tribool>           getPOpt(const IndexType) const;
    ValueType                              getPOpt() const;
    // Get Optimal Certificates
    bool                                   getOpt(const IndexType) const;
    const std::vector<bool>&               getOpt() const;
    // Get Optimal Labels
    LabelType                              get(const IndexType) const;
    void                                   get(std::vector<LabelType>&) const;

    const GmType&                          graphicalModel() const {return gm_;}
    void                                   reducedGraphicalModel(ReducedGmType&) const;

    void                                   OriginalToReducedLabeling(const std::vector<LabelType>& lOrig, std::vector<LabelType>& lRed) const;
    void                                   ReducedToOriginalLabeling(std::vector<LabelType>& lOrig, const std::vector<LabelType>& lRed) const;

private:
    const GmType& gm_;
    std::vector<std::vector<opengm::Tribool> >   partialOptimality_;
    std::vector<bool>                            optimal_;
    std::vector<LabelType>                       labeling_;
    std::vector<IndexType>                       countFalse_;
    std::vector<std::vector<std::vector<LabelType> > >  excludedLabelings_; // for each factor the excluded labelings are pushed
    std::vector<std::vector<std::vector<LabelType> > > excludedCount_;
    // count for number of excluded labelings in which (variable,label) partakes
    // used for elimination of variables
};
//! [class popt_data]

//********************************************

template<class GM>
POpt_Data<GM>::POpt_Data
(
    const GmType& gm
)
    :  gm_(gm),
    partialOptimality_(std::vector<std::vector<opengm::Tribool> >(gm.numberOfVariables()) ),
    optimal_(std::vector<bool>(gm.numberOfVariables(),false)),
    labeling_(std::vector<LabelType>(gm.numberOfVariables(),0)),
    countFalse_(std::vector<IndexType>(gm.numberOfVariables(),0)),
    excludedLabelings_(gm.numberOfFactors()),
    excludedCount_(gm.numberOfFactors()) {

    for(IndexType var=0; var<gm_.numberOfVariables(); ++var) {
        partialOptimality_[var].resize(gm_.numberOfLabels(var),opengm::Tribool::Maybe);
    }
    for(size_t f=0; f<gm_.numberOfFactors(); f++) {
        excludedCount_[f].resize(gm_[f].numberOfVariables());
        for(size_t var=0; var<gm_[f].numberOfVariables(); var++) {
            excludedCount_[f][var].resize( gm_.numberOfVariables( gm_.variableOfFactor(f,var) ), 0);
        }
    }
}

//********************************************

// Set
template<class GM>
void POpt_Data<GM>:: setTrue(const IndexType var, const LabelType label) {
    OPENGM_ASSERT( var   < gm_.numberOfVariables() );
    OPENGM_ASSERT( label < gm_.numberOfLabels(var) );

    optimal_[var] = true;
    labeling_[var]  = label;

    OPENGM_ASSERT(partialOptimality_[var][label] != false);
    partialOptimality_[var][label] = true;

    for(size_t i=0; i<label; ++i) {
        OPENGM_ASSERT(partialOptimality_[var][i] != true);
        partialOptimality_[var][i] = false;
    }
    for(size_t i=label+1; i<partialOptimality_[var].size(); ++i) {
        OPENGM_ASSERT(partialOptimality_[var][i] != true);
        partialOptimality_[var][i] = false;
    }
    countFalse_[var] = partialOptimality_[var].size()-1;
}

template<class GM>
void POpt_Data<GM>::setFalse(IndexType var, LabelType label) {
    OPENGM_ASSERT( var   < gm_.numberOfVariables() );
    OPENGM_ASSERT( label < gm_.numberOfLabels(var) );

    if( optimal_[var] ) {
        OPENGM_ASSERT(labeling_[var] != label);
    } else {
        OPENGM_ASSERT(partialOptimality_[var][label] != true);
        partialOptimality_[var][label] = false;
        if(++countFalse_[var] == partialOptimality_[var].size()-1) {
            for(size_t i=0; i<partialOptimality_[var].size(); ++i) {
                if( partialOptimality_[var][i] != false ) {
                    partialOptimality_[var][i] = true;
                    optimal_[var]              = true;
                    labeling_[var]             = i;
                    break;
                }
            }
        }
    }
}

template<class GM>
void POpt_Data<GM>::setFalse(IndexType f, const std::vector<LabelType>& labeling) {
    OPENGM_ASSERT( f < gm_.numberOfFactors() );
    OPENGM_ASSERT( labeling.size() == gm_[f].numberOfVariables() );
    bool labelingTrue = true;
    for(size_t var=0; var<gm_[f].numberOfVariables(); var++) {
        OPENGM_ASSERT( labeling[var] < gm_.numberOfVariables( gm_.variableOfFactor(f,var) ) );
        OPENGM_ASSERT( getPOpt(gm_.variableOfFactor(f,var), labeling[var]) != Tribool::False );
        getPOpt( gm_.variableOfFactor(f,var), labeling[var] ) == Tribool::True ? labelingTrue= true : labelingTrue = false;
    }
    OPENGM_ASSERT( labelingTrue == false);
    for(size_t e=0; e<excludedLabelings_[f].size(); e++) {
        OPENGM_ASSERT( std::equal(labeling.begin(), labeling.end(), excludedLabelings_[f][e].begin()) );
    }

    excludedLabelings_[f].push(labeling);

    // check for implications: e.g. if all labels (i,1), ... (i,n) are excluded, exclude label i for the first variables as well.
    for(size_t var=0; var<gm_[f].numberOfVariables(); var++) {
        excludedCount_[f][var][ labeling[var] ]++;
    }
    for(size_t var=0; var<gm_[f].numberOfVariables(); var++) {
        if(excludedCount_[f][var][ labeling[var] ] == gm_[f].size()/gm_.numberOfLabels(gm_.variableOfFactor(f,var)) ) {
            // label can be excluded
            setFalse(gm_.variableOfFactor(f,var), labeling[var]);
        }
    }
}

// Get Partial Optimality
template<class GM>
Tribool POpt_Data<GM>::getPOpt(const IndexType var, const LabelType label) const {
    OPENGM_ASSERT( var   < gm_.numberOfVariables() );
    OPENGM_ASSERT( label < gm_.numberOfLabels(var) );
    return partialOptimality_[var][label];
}

template<class GM>
std::vector<opengm::Tribool> POpt_Data<GM>::getPOpt(const IndexType var) const {
    OPENGM_ASSERT( var<gm_.numberOfVariables() );
    return partialOptimality_[var];
}

template<class GM>
typename GM::ValueType POpt_Data<GM>::getPOpt() const {
   size_t noExcludedLabels = 0;
   size_t noLabels = 0;
   for(size_t var=0; var<gm_.numberOfVariables(); var++) {
      noLabels += gm_.numberOfLabels(var);
      for(size_t i=0; i<gm_.numberOfLabels(var); i++) {
         if(getPOpt(var,i) == false)
            noExcludedLabels++;
      }
   }
   OPENGM_ASSERT(noLabels > 0);
   OPENGM_ASSERT(noExcludedLabels < noLabels);
   ValueType p = ValueType(noExcludedLabels) / ValueType(noLabels - gm_.numberOfVariables());

   OPENGM_ASSERT(p <= 1.0);
   return p;
}

// Get Optimality
template<class GM>
bool POpt_Data<GM>::getOpt(const IndexType var) const {
    OPENGM_ASSERT( var<gm_.numberOfVariables() );
    return optimal_[var];
}

template<class GM>
const std::vector<bool>& POpt_Data<GM>::getOpt() const {
    return optimal_;
}

// Get Optimal Labels
template<class GM>
typename GM::LabelType POpt_Data<GM>::get(const IndexType var) const {
    OPENGM_ASSERT( var<gm_.numberOfVariables() );
    return labeling_[var];
}

template<class GM>
void POpt_Data<GM>::get(std::vector<LabelType>& labeling) const {
    OPENGM_ASSERT(labeling.size() == gm_.numberOfVariables());
    for(IndexType var=0; var<gm_.numberOfVariables(); ++var) {
        if( optimal_[var] )
            labeling[var] = labeling_[var];
    }
}

template<class GM>
void POpt_Data<GM>::reducedGraphicalModel(ReducedGmType& reducedGm) const
{
   //Variables and their labels are included in case they are not optimal yet.

   IndexType newVariable[gm_.numberOfVariables()];
   IndexType variable = 0;

   for (IndexType v = 0; v < gm_.numberOfVariables(); v++) {
      if(!optimal_[v]) {
         newVariable[v] = variable;
         variable++;

         LabelType numLabels = 0;

         for(IndexType i = 0; i < gm_.numberOfLabels(v); i++) {
            if(!(partialOptimality_[v][i] == opengm::Tribool::False))
               numLabels++;
         }
         OPENGM_ASSERT(numLabels>1);
         reducedGm.addVariable(numLabels);
      }
   }


   //factors will be included in case one of their variables is not optimal yet.
   for (IndexType f = 0; f < gm_.numberOfFactors(); f++) {
      bool insert_f = false;
      size_t numVariables = 0;
      std::vector<IndexType> variablesOfFactor;
      std::vector<IndexType> numLabels;
      IndexType numLabelComb = 1;

      for(IndexType v = 0; v < gm_[f].numberOfVariables(); v++){
         if(!optimal_[gm_[f].variableIndex(v)]){
            insert_f = true;
            numVariables++;
            variablesOfFactor.std::vector<IndexType>::push_back(newVariable[gm_[f].variableIndex(v)]);
            numLabels.std::vector<IndexType>::push_back(reducedGm.numberOfLabels(newVariable[gm_[f].variableIndex(v)]));
            numLabelComb *= numLabels.std::vector<IndexType>::back();
         }
      }

      if(insert_f){
         ExplicitFunction g(numLabels.std::vector<IndexType>::begin(), numLabels.std::vector<IndexType>::end());
         LabelType reducedShape [numVariables];
         LabelType shape [gm_[f].numberOfVariables()];

         for(size_t v = 0; v < numVariables; v++){
            reducedShape[v] = 0;
         }

         for(size_t v = 0; v < gm_[f].numberOfVariables(); v++){
            if(optimal_[gm_[f].variableIndex(v)]){
               shape[v] = labeling_[gm_[f].variableIndex(v)];
            }else{
               shape[v] = 0;
            }
         }
         g(reducedShape) = gm_[f](shape);

         for(size_t comb = 1; comb < numLabelComb; comb++){
            //next label combination
            IndexType nextStep = 0;
            while(reducedShape[nextStep] = numLabels[nextStep] && nextStep < numVariables-1){
               nextStep++;
            }
            reducedShape[nextStep]++;

            size_t index = 0;
            for(size_t v = 0; v < nextStep; v++){
               reducedShape[v] = 0;

               while(optimal_[gm_[f].variableIndex(index)])
                  index++;

               size_t label = 0;
               while(partialOptimality_[gm_[f].variableIndex(index)][label] == opengm::Tribool::False)
                  label++;

               shape[index] = label;
               index++;
            }

            while(optimal_[gm_[f].variableIndex(index)]){
               index++;
            }
            shape[index]++;
            while(partialOptimality_[gm_[f].variableIndex(index)][shape[index]] == opengm::Tribool::False)
               shape[index]++;

            g(reducedShape) = gm_[f](shape);
         }

         typename ReducedGmType::FunctionIdentifier id = reducedGm.addFunction(g);

         reducedGm.addFactor(id, variablesOfFactor.std::vector<IndexType>::begin(), variablesOfFactor.std::vector<IndexType>::end());
      }
   }
}

template<class GM>
void POpt_Data<GM>::OriginalToReducedLabeling(const std::vector<LabelType>& lOrig, std::vector<LabelType>& lRed) const
{
}

template<class GM>
void POpt_Data<GM>::ReducedToOriginalLabeling(std::vector<LabelType>& lOrig, const std::vector<LabelType>& lRed) const
{
}


} // namespace opengm

#endif // #ifndef OPENGM_POPT_DATA_HXX
