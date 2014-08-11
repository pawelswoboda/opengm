#pragma once
#ifndef POPT_INFER_HXX
#define POPT_INFER_HXX

#include "popt_inference_base.hxx"
#include "popt_data.hxx"
#include "popt_kovtun.hxx"
#include "popt_dee.hxx"
#include "popt_iterative_relaxed_inf.hxx"
#include "popt_iri_trws.hxx"
#include "popt_iri_cplex.hxx"

namespace opengm {

template<class GM, class ACC> 
class POpt_infer : public Inference<GM, ACC>
{  
public:
   struct Parameter {
      enum Method {DEE1,DEE2,DEE3,DEE4,Kovtun,MQPBO,IRI_TRWS,IRI_CPLEX};
      std::vector<Method> methodSequence_;
   };

   typedef ACC AccumulationType;
   typedef GM GraphicalModelType;
   OPENGM_GM_TYPE_TYPEDEFS;
   typedef visitors::VerboseVisitor<POpt_infer<GM,ACC> > VerboseVisitorType;
   typedef visitors::EmptyVisitor<POpt_infer<GM,ACC> >   EmptyVisitorType;
   typedef visitors::TimingVisitor<POpt_infer<GM,ACC> >  TimingVisitorType;

   POpt_infer(const GM& gm, const Parameter& p);
   virtual std::string name() const {return "POpt_infer";}
   const GraphicalModelType& graphicalModel() const {return gm_;};
   InferenceTermination infer() { EmptyVisitorType visitor; return infer(visitor);}
   template<class VISITOR>
      InferenceTermination infer(VISITOR &);
   InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
   virtual ValueType value() const; 


private:
   POpt_Data<GM> d_;
   const GraphicalModelType gm_;
   const Parameter parameter_;
};

template<class GM, class ACC>
POpt_infer<GM,ACC>::POpt_infer
(
 const GM& gm,
 const Parameter& p
)
    :  gm_(gm),
    parameter_(p),
    d_(POpt_Data<GM>(gm))
{
}

template<class GM, class ACC>
template<class Visitor>
InferenceTermination
POpt_infer<GM,ACC>::infer(Visitor& visitor)
{
   visitor.begin(*this);
   for(size_t i=0; i<parameter_.methodSequence_.size(); i++) {
      InferenceTermination infReturnValue;
      if(parameter_.methodSequence_[i] == Parameter::DEE1) {
         DEE<POpt_Data<GM>, opengm::Minimizer> dee(d_);
         infReturnValue = dee.dee1();
      } else if(parameter_.methodSequence_[i] == Parameter::DEE3  ) {
         DEE<POpt_Data<GM>, opengm::Minimizer> dee(d_);
         infReturnValue = dee.dee3();
      } else if(parameter_.methodSequence_[i] == Parameter::DEE4 ) {
         DEE<POpt_Data<GM>, opengm::Minimizer> dee(d_);
         infReturnValue = dee.dee4();
#ifdef WITH_QPBO
      } else if(parameter_.methodSequence_[i] == Parameter::Kovtun ) {
         POpt_Kovtun<POpt_Data<GM>, opengm::Minimizer> kovtun(d_);
         infReturnValue = kovtun.infer();
#endif
      } else if(parameter_.methodSequence_[i] == Parameter::IRI_TRWS  ) {
         IRI::IRI<POpt_Data<GM>,opengm::Minimizer,POpt_IRI_TRWS> iri(d_);
         infReturnValue = iri.infer();
#ifdef WITH_CPLEX
      } else if(parameter_.methodSequence_[i] == Parameter::IRI_CPLEX) {
         IRI::IRI<POpt_Data<GM>,opengm::Minimizer,POpt_IRI_CPLEX> iri(d_);
         infReturnValue = iri.infer();
#endif
      } else {
         std::cout << "method not implemented yet" << std::endl;
         throw; 
      }
      std::cout << "Percentage partial optimality after " << i << "th iteration" << ": " << d_.getPOpt() << std::endl;
      if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
         break;
      }
      if(infReturnValue != NORMAL) {
         std::cout << "POpt_infer: return value in " << i << "th persistency method not NORMAL, exiting" << std::endl;
         return infReturnValue;
      }
   }
   visitor.end(*this);
   return NORMAL;
}

template<class GM, class ACC>
InferenceTermination 
POpt_infer<GM,ACC>::arg(
      std::vector<typename GM::LabelType>& l,
      const size_t T
      ) const
{
   // returns the number of excluded labels
   l.resize(gm_.numberOfVariables(),0);
   for(size_t i=0; i<gm_.numberOfVariables(); i++)
      for(size_t x_i=0; x_i<gm_.numberOfLabels(i); x_i++)
         if(d_.getPOpt(i,x_i) == false)
            l[i]++;
   return NORMAL;
}

/// \brief return the percentage partial optimality
template<class GM, class ACC>
typename GM::ValueType
POpt_infer<GM, ACC>::value() const 
{
   return d_.getPOpt();
} 



} // end namespace opengm

#endif // POPT_INFER_HXX
