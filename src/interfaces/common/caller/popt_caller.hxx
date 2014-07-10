#pragma once
#ifndef POPT_CALLER_HXX_
#define POPT_CALLER_HXX_

#include <opengm/inference/partialOptimality/popt_data.hxx>
#include <opengm/inference/partialOptimality/popt_inference.hxx>
#include <opengm/inference/partialOptimality/popt_kovtun.hxx>
#include <opengm/inference/partialOptimality/popt_dee.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

#include <sstream>

namespace opengm {
   namespace interface {

      template <class IO, class GM, class ACC>
      class POptCaller : public InferenceCallerBase<IO, GM, ACC, POptCaller<IO, GM, ACC> >
      {
         //enum POptMethod {Kovtun, DEE1, DEE2, DEE3, DEE4, MQPBO, IterativePruning, None}; // noch nicht alle implementiert

      protected:
         typedef InferenceCallerBase<IO, GM, ACC,  POptCaller<IO, GM, ACC> > BaseClass;
         typedef typename BaseClass::OutputBase OutputBase;

         using BaseClass::addArgument;
         using BaseClass::io_;
         using BaseClass::infer;

         virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

      private:
         template<class Parameter> void setParameter(Parameter& p);
         std::vector<std::string> POptMethods_;
         std::vector<std::string> POptSequence_; // call partial optimality methods sequentially to accumulate persistent labels
      public:
         const static std::string name_;
         const static size_t noPOpt_=20; // number of partial optimality methods which can be called.
         POptCaller(IO& ioIn);
      };

      template <class IO, class GM, class ACC>
      const std::string POptCaller<IO, GM, ACC>::name_ = "POPT";

      template <class IO, class GM, class ACC>
      inline POptCaller<IO, GM, ACC>::POptCaller(IO& ioIn)
         : BaseClass("POPT", "detailed description of POPT Parser...", ioIn)
      {
         POptMethods_.push_back("NONE");
         POptMethods_.push_back("Kovtun");
         POptMethods_.push_back("DEE1");
         POptMethods_.push_back("DEE2");
         POptMethods_.push_back("DEE3");
         POptMethods_.push_back("DEE4");
         POptMethods_.push_back("MQPBO");
         POptMethods_.push_back("IterativePruning");

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
         POpt_Data<GM> D(model);
         for(size_t i=0; i<noPOpt_; i++) {
            if(POptSequence_[i].compare("NONE") == 0 ) {
            } else if(POptSequence_[i].compare("Kovtun") == 0 ) {
               POpt_Kovtun<POpt_Data<GM>, opengm::Minimizer> p(D);
               p.infer();
            } else if(POptSequence_[i].compare("DEE1") == 0 ) {
               DEE<POpt_Data<GM>, opengm::Minimizer> d(D);
               d.dee1();
            }
//            else if(POptSequence_[i].compare("DEE2") == 0 ) {
//               DEE<POpt_Data<GM>, opengm::Minimizer> d(D);
//               d.dee2();
//            }
            else if(POptSequence_[i].compare("DEE3") == 0 ) {
               DEE<POpt_Data<GM>, opengm::Minimizer> d(D);
               d.dee3();
            } else if(POptSequence_[i].compare("DEE4") == 0 ) {
               DEE<POpt_Data<GM>, opengm::Minimizer> d(D);
               d.dee4();
            } else {
               std::cout << "method not implemented yet" << std::endl;
               throw; // no other methods implemented yet
            }
         }
      }

   } // namespace interface
} // namespace opengm

#endif /* POPT_CALLER_HXX_ */
