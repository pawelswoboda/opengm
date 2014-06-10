#pragma once
#ifndef OPENGM_POPT_KOVTUN_HXX
#define OPENGM_POPT_KOVTUN_HXX

#include <vector>
#include "popt_data.hxx"

#include <opengm/inference/external/qpbo.hxx>


namespace opengm { 

   //! [class popt_kovtun]
   /// Partial Optimaility by Kovtuns' Method
   ///
   /// Corresponding author: Joerg Hendrik Kappes
   ///
   ///\ingroup inference
   template<class DATA, class ACC>
   class POpt_Kovtun
   {
   public:
      typedef ACC AccumulationType;
      typedef typename DATA::GmType GmType;
      typedef GM GraphicalModelType;
      OPENGM_GM_TYPE_TYPEDEFS;
      typedef ValueType GraphValueType; 

      POpt_Data(DATA&);
      ~POpt_Kovtun();

      void pOptPotts(const LabelType);
      void pOptPotts();

   private: 
      DATA&   data_;
      const GmType& gm_;
      kolmogorov::qpbo::QPBO<GraphValueType>* qpbo_;
   };

   template<class DATA, class ACC>
   POpt_Kovtun<DATA,ACC>::POpt_Kovtun
   ( DATA& data ) : data_(data), gm_(data_.graphicalModel())
   {
      size_t numEdges = 0;
      for(size_t j = 0; j < gm_.numberOfFactors(); ++j) {
         if(gm_[j].numberOfVariables() == 2){
            ++numEdges ;
         }else if(gm_[j].numberOfVariables() > 2) {
            throw RuntimeError("This implementation of Kovtuns method supports only factors of order <= 2.");
         }
      }

      //preallocate mem for max flow
      qpbo_ = new kolmogorov::qpbo::QPBO<GraphValueType > (gm_.numberOfVariables(), numEdges);
   }

   template<class DATAS, class ACC>
   POpt_Kovtun<DATA,ACC>::~POpt_Kovtun(){
      delete qpbo_;
   }

   template<class DATAS, class ACC>
   void POpt_Kovtun<DATA,ACC>::pOptPotts(const LabelType)
   {
      qpbo_->Reset();
      qpbo_->AddNode(gm_.numberOfVariables());
      for(size_t f = 0; f < gm_.numberOfFactors(); ++f) {
         if(gm_[f].numberOfVariables() == 0) {
            ;
         }
         else if(gm_[f].numberOfVariables() == 1) {
            const LabelType numLabels =  gm_[f].numberOfLabels(0);
            const IndexType var = gm_[f].variableIndex(0);
            
            ValueType v0 = gm_[f](&guess);
            ValueType v1; ACC::neutral(v1);
            for(LabelType i=0; i<guess; ++i)
               ACC::op(gm_[f](&i),v1);
            for(LabelType i=guess+1; i<numLabels; ++i)
               ACC::op(gm_[f](&i),v1);
            qpbo_->AddUnaryTerm(var, v0, v1);
         }
         else if(gm_[f].numberOfVariables() == 2) {
            const IndexType var0 = gm_[f].variableIndex(0);
            const IndexType var1 = gm_[f].variableIndex(1); 
            
            LabelType c[2] = {guess,guess};
            LabelType c2[2] = {0,1};
            
            ValueType v00 = gm_[f](c);
            ValueType v01 = gm_[f](c2);
            ValueType v10 = v01; 
            ValueType v11 = std::min(v00,v01);
            qpbo_->AddPairwiseTerm(var0, var1, v00, v01, v10, v11);
         }
      }
      qpbo_->MergeParallelEdges();
      qpbo_->Solve();
      for(IndexType var=0; var<gm_.numberOfVariables();++var){
         if(qpbo_->GetLabel(var)==0){
            data_.setTrue(var,guess)
         }
      }
   }

   template<class DATAS, class ACC>
   void POpt_Kovtun<DATA,ACC>::pOptPotts()
   {
      for(LabelType l=0 ; l<gm_[f].numberOfLabels(0); ++l){
         pOptPotts(l);
      }
   }
      
} // namespace opengm

#endif // #ifndef OPENGM_POPT_KOVTUN_HXX
