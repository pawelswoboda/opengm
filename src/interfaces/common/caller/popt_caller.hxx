#pragma once
#ifndef POPT_CALLER_HXX_
#define POPT_CALLER_HXX_

#include <opengm/inference/partialOptimality/popt_infer.hxx>
#include <opengm/utilities/tribool.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

#include <sstream>

namespace opengm {
namespace interface {

template <class IO, class GM, class ACC>
class POptCaller : public InferenceCallerBase<IO, GM, ACC, POptCaller<IO, GM, ACC> >
{
public:
   typedef POpt_infer<GM, ACC> POpt_inferType;
   typedef typename POpt_infer<GM, ACC>::Parameter POpt_ParameterType;
   typedef InferenceCallerBase<IO, GM, ACC, POptCaller<IO, GM, ACC> > BaseClass;
   typedef typename POpt_inferType::VerboseVisitorType VerboseVisitorType;
   typedef typename POpt_inferType::EmptyVisitorType EmptyVisitorType;
   typedef typename POpt_inferType::TimingVisitorType TimingVisitorType;
   typedef typename BaseClass::OutputBase OutputBase;

   const static std::string name_;
   const static size_t noPOpt_=20; // number of partial optimality methods which can be called.
   POptCaller(IO& ioIn);
   template <class OUTPUTTYPE>
   void run(GM& model, OUTPUTTYPE& outputfile, const bool verbose);
   //template <class INF, class VISITOR, class PARAMETER>
   //   void infer(GM& model, OutputBase& output, const bool verbose, const PARAMETER& param) const;

protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typename POpt_inferType::Parameter POpt_parameter_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

private:
   template<class Parameter> void setParameter(Parameter& p);
   std::vector<std::string> POptSequence_; // call partial optimality methods sequentially to accumulate persistent labels
   std::vector<std::string> POptMethods_; 
   std::string outputFilename;
};

template <class IO, class GM, class ACC>
const std::string POptCaller<IO, GM, ACC>::name_ = "POPT";

template <class IO, class GM, class ACC>
inline POptCaller<IO, GM, ACC>::POptCaller(IO& ioIn)
   : BaseClass("POPT", "detailed description of Partial Optimality Parser...", ioIn)
{
   POptMethods_.push_back("NONE");
#ifdef WITH_QPBO
   POptMethods_.push_back("Kovtun");
#endif
   POptMethods_.push_back("DEE1");
   POptMethods_.push_back("DEE2");
   POptMethods_.push_back("DEE3");
   POptMethods_.push_back("DEE4");
   POptMethods_.push_back("IRI_TRWS");
   POptMethods_.push_back("IRI_ADSAL");
#ifdef WITH_CPLEX
   POptMethods_.push_back("IRI_CPLEX");
#endif
   POptMethods_.push_back("PBP_TRWS");
   POptMethods_.push_back("PBP_ADSAL");
#ifdef WITH_CPLEX
   POptMethods_.push_back("PBP_CPLEX");
#endif

   POptSequence_.resize(noPOpt_);
   // make list of partial optimality methods to call
   for(size_t i=0; i<noPOpt_; i++) {
      std::string method;
      std::string argName("POPT");
      std::stringstream ss;
      ss << i;
      argName.append(ss.str());
      addArgument(StringArgument<>(POptSequence_[i],
               "", argName, "Algorithm used for partial optimality", POptMethods_[0], POptMethods_));

   }
}

template <class IO, class GM, class ACC>
template<class Parameter>
void POptCaller<IO, GM, ACC>::setParameter(Parameter& p)
{
}

template <class IO, class GM, class ACC>
inline void POptCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose)
{
   std::cout << "running Partial Optimality caller" << std::endl;
   typedef typename GM::ValueType ValueType;
   POpt_ParameterType parameter_;
   for(size_t i=0; i<noPOpt_; i++) {
      if(POptSequence_[i].compare("NONE") == 0 ) {
#ifdef WITH_QPBO
      } else if(POptSequence_[i].compare("Kovtun") == 0 ) {
         parameter_.methodSequence_.push_back(POpt_ParameterType::Kovtun);
#endif
      } else if(POptSequence_[i].compare("DEE1") == 0 ) {
         parameter_.methodSequence_.push_back(POpt_ParameterType::DEE1);
      } else if(POptSequence_[i].compare("DEE3") == 0 ) {
         parameter_.methodSequence_.push_back(POpt_ParameterType::DEE3);
      } else if(POptSequence_[i].compare("DEE4") == 0 ) {
         parameter_.methodSequence_.push_back(POpt_ParameterType::DEE4);
      } else if(POptSequence_[i].compare("IRI_TRWS") == 0 ) {
         parameter_.methodSequence_.push_back(POpt_ParameterType::IRI_TRWS);
      } else if(POptSequence_[i].compare("IRI_ADSal") == 0 ) {
         parameter_.methodSequence_.push_back(POpt_ParameterType::IRI_ADSal);
      } else if(POptSequence_[i].compare("IRI_CPLEX") == 0 ) {
         parameter_.methodSequence_.push_back(POpt_ParameterType::IRI_CPLEX);
      } else if(POptSequence_[i].compare("PBP_TRWS") == 0 ) {
         parameter_.methodSequence_.push_back(POpt_ParameterType::PBP_TRWS);
      } else if(POptSequence_[i].compare("PBP_ADSAL") == 0 ) {
         parameter_.methodSequence_.push_back(POpt_ParameterType::PBP_ADSal);
      } else if(POptSequence_[i].compare("PBP_CPLEX") == 0 ) {
         parameter_.methodSequence_.push_back(POpt_ParameterType::PBP_CPLEX);
      } else {
         std::cout << "method not implemented yet" << std::endl;
         throw; // no other methods implemented yet
      }
   }

   POpt_infer<GM,ACC> inference(model, parameter_);
   if((inference.infer() == UNKNOWN)) {
      std::string error("partial optimality method failed.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }

   std::vector<typename GM::LabelType> states;
   inference.arg(states);
   output.storeStates(states);
   std::vector<typename GM::ValueType> partialOptimality(model.numberOfVariables());
   for(size_t i=0; i<model.numberOfVariables(); i++) {
      for(size_t x_i=0; x_i<model.numberOfLabels(i); x_i++)
         if(inference.getPOpt_Data().getPOpt(i,x_i) == opengm::Tribool::False)
            partialOptimality.push_back(0.0);
         else if(inference.getPOpt_Data().getPOpt(i,x_i) == opengm::Tribool::True)
            partialOptimality.push_back(1.0);
         else
            partialOptimality.push_back(0.5);
   }

   std::string storage = outputFilename;
   storage += ":partialOptimality";
   io_.storeVector(storage, partialOptimality);

   storage = outputFilename;
   storage += ":partialOptimalityPercentage";
   partialOptimality =  std::vector<double>(1,inference.getPOpt_Data().getPOpt());
   io_.storeVector(storage, partialOptimality);

   //this-> template infer<POpt_inferType, TimingVisitorType, typename POpt_inferType::Parameter>(model, output, verbose, parameter_);
}

template <class IO, class GM, class ACC>
template <class OUTPUTTYPE>
void POptCaller<IO, GM, ACC>::run(GM& model, OUTPUTTYPE& outputfile, const bool verbose) {
   BaseClass::argumentContainer_.read();
   io_.read(outputfile);
   outputFilename = outputfile.getValue();
   if(meta::Compare<OUTPUTTYPE, StringArgument<> >::value) {
      typename BaseClass::HDF5Output output(io_, outputfile);
      runImpl(model, output, verbose);
   } else
#ifdef WITH_MATLAB
      if(meta::Compare<OUTPUTTYPE, mxArrayArgument<> >::value) {
         typename BaseClass::MatlabOutput output(io_, outputfile);
         runImpl(model, output, verbose);
      } else
#endif
   {
      throw RuntimeError("Unsupported Outputtype");
   }
}


/*
template <class IO, class GM, class ACC>
template <class INF, class VISITOR, class PARAMETER>
inline void POptCaller<IO, GM, ACC>::infer(GM& model, OutputBase& output, const bool verbose, const PARAMETER& param) const {
   INF* inference = NULL;

   if(protocolate_->isSet()) {
      if(protocolate_->getValue() != 0) {
         VISITOR visitor(protocolate_->getValue(), 0, verbose, true, timeLimit_, gapLimit_);
         inference = new INF(model, param);
         if((inference->infer(visitor) == UNKNOWN)) {
            std::string error(inference->name() + " did not solve the problem.");
            io_.errorStream() << error << std::endl;
            delete inference;
            throw RuntimeError(error);
         }
         protocolate(visitor, output);
      } else {
         inference = new INF(model, param);
         if((inference->infer() == UNKNOWN)) {
            std::string error(inference->name() + " did not solve the problem.");
            io_.errorStream() << error << std::endl;
            delete inference;
            throw RuntimeError(error);
         }
      }
   } else {
      inference = new INF(model, param);
      if((inference->infer() == UNKNOWN)) {
         std::string error(inference->name() + " did not solve the problem.");
         io_.errorStream() << error << std::endl;
         delete inference;
         throw RuntimeError(error);
      }
   }

   std::vector<typename GM::LabelType> states;
   if(!(inference->arg(states) == NORMAL)) {
      std::string error(inference->name() + " could not return optimal argument.");
      io_.errorStream() << error << std::endl;
      delete inference;
      throw RuntimeError(error);
   }

   output.storeStates(states);
   std::vector<opengm::Tribool> partialOptimality(model.numberOfVariables());
   for(size_t i=0; i<model.numberOfVariables(); i++) {
      partialOptimality.push_back(inference.getPOpt_Data().getPOpt(i));
   output.store(partialOptimality,*output,"partialOptimality");

   delete inference;
}
*/


} // namespace interface
} // namespace opengm

#endif /* POPT_CALLER_HXX_ */
