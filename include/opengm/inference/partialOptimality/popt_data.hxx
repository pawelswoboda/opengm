#pragma once
#ifndef OPENGM_POPT_DATA_HXX
#define OPENGM_POPT_DATA_HXX

#include <vector>
//#include <string>
//#include <iostream>

#include <opengm/opengm.hxx>
#include <opengm/utilities/metaprogramming.hxx>
#include <opengm/utilities/tribool.hxx>


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
      typedef GM GmType;
      OPENGM_GM_TYPE_TYPEDEFS;
  
      POpt_Data(const GmType&);
      // Set
      void                                   setTrue(const IndexType, const LabelType);
      void                                   setFalse(const IndexType, const LabelType);
      // Get Partial Optimality
      Tribool                                getPOpt(const IndexType, const LabelType) const;
      const std::vector<opengm::Tribool>&    getPOpt(const IndexType) const;
      // Get Optimal Certificates
      bool                                   getOpt(const IndexType) const;
      const std::vector<bool>&               getOpt() const;
       // Get Optimal Labels 
      LabelType                              get(const IndexType) const;
      void                                   get(std::vector<LabelType>&) const; 

      const GmType&                          graphicalModel() const {return gm_;}

   private: 
      const GmType& gm_;
      std::vector<std::vector<opengm::Tribool> >   partialOptimality_;
      std::vector<bool>                            optimal_;
      std::vector<LabelType>                       labeling_;
      std::vector<IndexCount>                      countFalse_;
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
      countFalse_(std::vector<IndexType>(gm.numberOfVariables(),0))  
   {
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){
        partialOptimality_[var].resize(gm_.numberOfLabels(var),opengm::Tribool::Maybe); 
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
         OPENGM_ASSERT(label_[var] == label);
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

   // Get Partial Optimality 
   template<class GM>
   Tribool POpt_Data<GM>::getPOpt(const IndexType var, const LabelType label) const{ 
      OPENGM_ASSERT( var   < gm_.numberOfVariables() );
      OPENGM_ASSERT( label < gm_.numberOfLabels(var) );
      return partialOptimality_[var][label]
   }

   template<class GM>
   const std::vector<opengm::Tribool>& POpt_Data<GM>::getPOpt::getPOpt(const IndexType var) const{
      OPENGM_ASSERT( var<gm_.numberOfVariables() );
      return partialOptimality_[var];
   }

   // Get Optimality
   template<class GM>
   bool POpt_Data<GM>::getPOpt::getOpt(const IndexType var) const{
      OPENGM_ASSERT( var<gm_.numberOfVariables() );
      return optimal_[var];
   }

   template<class GM>
   const std::vector<bool>& POpt_Data<GM>::getPOpt::getOpt() const{
      return optimal_;
   }

   // Get Optimal Labels  
   template<class GM>
   LabelType POpt_Data<GM>::getPOpt::get(const IndexType var) const{
      OPENGM_ASSERT( var<gm_.numberOfVariables() );
      return labeling_[var];
   }

   template<class GM>   
   void POpt_Data<GM>::getPOpt::get(std::vector<LabelType>& labeling) const{
      OPENGM_ASSERT(labeling.size() == gm_.numberOfVariables());
      for(IndexType var=0; var<gm_.numberOfVariables(); ++var){ 
         if( optimal_[var] )
            labeling[var] = labeling_[var];
      } 
   }


} // namespace opengm

#endif // #ifndef OPENGM_POPT_DATA_HXX
