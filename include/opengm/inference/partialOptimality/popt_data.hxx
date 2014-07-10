#pragma once
#ifndef OPENGM_POPT_DATA_HXX
#define OPENGM_POPT_DATA_HXX

#include <vector>
//#include <string>
//#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/utilities/metaprogramming.hxx>
#include <opengm/utilities/tribool.hxx>
#include <opengm/functions/reduced_view.hxx>


namespace opengm {
   //! [class popt_data]
   /// Partial Optimaility Data (-Container)
   ///
   /// Corresponding author: Joerg Hendrik Kappes
   ///
   ///\ingroup inference
   template<class GM>
   class POpt_Data
   {
   public:
      typedef GM GraphicalModelType;
      typedef GM GmType;
      OPENGM_GM_TYPE_TYPEDEFS;

      typedef ReducedViewFunction<GraphicalModelType>                  ReducedViewType;
      typedef GraphicalModel<ValueType, OperatorType, ReducedViewType> ReducedGmType;

      POpt_Data(const GmType&);
      // Set
      void                                   setTrue(const IndexType, const LabelType);
      void                                   setFalse(const IndexType, const LabelType);
      void                                   setFalse(const IndexType factor, const std::vector<LabelType>& labeling);
      // Get Partial Optimality
      Tribool                                getPOpt(const IndexType, const LabelType) const;
      std::vector<opengm::Tribool>           getPOpt(const IndexType) const;
      // Get Optimal Certificates
      bool                                   getOpt(const IndexType) const;
      const std::vector<bool>&               getOpt() const;
       // Get Optimal Labels
      LabelType                              get(const IndexType) const;
      void                                   get(std::vector<LabelType>&) const;

      const GmType&                          graphicalModel() const {return gm_;}
      const ReducedGmType&                   reducedGraphicalModel() {

        DiscreteSpace<IndexType,LabelType> s;

        for (IndexType v = 0; v < gm_.numberOfVariables(); v++)
        {
            LabelType numLabels = 0;
            for(IndexType i = 0; i < gm_.numberOfLabels(v); i++)
            {
                if(!(partialOptimality_[v][i] == opengm::Tribool::False))
                    numLabels++;
            }
            // Hallo Anne: Kannst du bitte ein reduziertes Modell bauen ohne diejenigen Variablen, die nur 1 Label haben?
            OPENGM_ASSERT(numLabels>1);
            s.addVariable(numLabels);
        }

        ReducedGmType viewGM_(s);

        for (IndexType f = 0; f < gm_.numberOfFactors(); f++)
        {
                ReducedViewType g(gm_[f],partialOptimality_);
                typename ReducedGmType::FunctionIdentifier id = viewGM_.addFunction(g);
                viewGM_.addFactor(id, gm_[f].variableIndicesBegin(), gm_[f].variableIndicesEnd());
        }
        return viewGM_;
      }

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
      excludedCount_(gm.numberOfFactors())
   {
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
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
   void POpt_Data<GM>:: setTrue(const IndexType var, const LabelType label){
      OPENGM_ASSERT( var   < gm_.numberOfVariables() );
      OPENGM_ASSERT( label < gm_.numberOfLabels(var) );

      optimal_[var] = true;
      labeling_[var]  = label;

      OPENGM_ASSERT(partialOptimality_[var][label] != false);
      partialOptimality_[var][label] = true;

      for(size_t i=0; i<label; ++i){
         OPENGM_ASSERT(partialOptimality_[var][i] != true);
         partialOptimality_[var][i] = false;
      }
      for(size_t i=label+1; i<partialOptimality_[var].size(); ++i){
         OPENGM_ASSERT(partialOptimality_[var][i] != true);
         partialOptimality_[var][i] = false;
      }
      countFalse_[var] = partialOptimality_[var].size()-1;
   }

   template<class GM>
   void POpt_Data<GM>::setFalse(IndexType var, LabelType label){
      OPENGM_ASSERT( var   < gm_.numberOfVariables() );
      OPENGM_ASSERT( label < gm_.numberOfLabels(var) );

      if( optimal_[var] ){
         OPENGM_ASSERT(labeling_[var] != label);
      }else{
         OPENGM_ASSERT(partialOptimality_[var][label] != true);
         partialOptimality_[var][label] = false;
         if(++countFalse_[var] == partialOptimality_[var].size()-1){
            for(size_t i=0; i<partialOptimality_[var].size(); ++i){
               if( partialOptimality_[var][i] != false ){
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
   void POpt_Data<GM>::setFalse(IndexType f, const std::vector<LabelType>& labeling)
   {
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
   Tribool POpt_Data<GM>::getPOpt(const IndexType var, const LabelType label) const{
      OPENGM_ASSERT( var   < gm_.numberOfVariables() );
      OPENGM_ASSERT( label < gm_.numberOfLabels(var) );
      return partialOptimality_[var][label];
   }

   template<class GM>
   std::vector<opengm::Tribool> POpt_Data<GM>::getPOpt(const IndexType var) const{
      OPENGM_ASSERT( var<gm_.numberOfVariables() );
      return partialOptimality_[var];
   }

   // Get Optimality
   template<class GM>
   bool POpt_Data<GM>::getOpt(const IndexType var) const{
      OPENGM_ASSERT( var<gm_.numberOfVariables() );
      return optimal_[var];
   }

   template<class GM>
   const std::vector<bool>& POpt_Data<GM>::getOpt() const{
      return optimal_;
   }

   // Get Optimal Labels
   template<class GM>
   typename GM::LabelType POpt_Data<GM>::get(const IndexType var) const{
      OPENGM_ASSERT( var<gm_.numberOfVariables() );
      return labeling_[var];
   }

   template<class GM>
   void POpt_Data<GM>::get(std::vector<LabelType>& labeling) const{
      OPENGM_ASSERT(labeling.size() == gm_.numberOfVariables());
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
         if( optimal_[var] )
            labeling[var] = labeling_[var];
      }
   }


} // namespace opengm

#endif // #ifndef OPENGM_POPT_DATA_HXX
