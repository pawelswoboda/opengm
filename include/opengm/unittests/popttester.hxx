#pragma once
#ifndef OPENGM_TEST_PARTIAL_OPTIMALITY_HXX
#define OPENGM_TEST_PARTIAL_OPTIMALITY_HXX

#include <vector>
#include <typeinfo>

#include <opengm/opengm.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestbase.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/integrator.hxx>

#include <opengm/inference/partialOptimality/popt_data.hxx>
#include <opengm/inference/partialOptimality/popt_inference.hxx>

/// \cond HIDDEN_SYMBOLS

namespace opengm {
   template<class GM>
   class PartialOptimalityTester
   {
   public:
      typedef GM GraphicalModelType;
      template<class POPT> void test(const typename POPT::Parameter&, bool tValue=true, bool tArg=false, bool tMarg=false, bool tFacMarg=false);
      void addTest(BlackBoxTestBase<GraphicalModelType>*);
      ~PartialOptimalityTester();

   private:
      std::vector<BlackBoxTestBase<GraphicalModelType>*> testList;
   };

   //***************
   //IMPLEMENTATION
   //***************
   
   /*
   template class<GM>
   bool PartialOptimalityTester<GM>::noOptimalLabelsEliminated(GM& gm, POpt_Data<GM>& p)
   {
      // minimize original model
      opengm::Bruteforce<GraphicalModelType, AccType> bf(gm);
      OPENGM_TEST(bf.infer()==opengm::NORMAL);
      OPENGM_TEST(bf.arg(optimalState)==opengm::NORMAL);
      OPENGM_TEST(optimalState.size()==gm.numberOfVariables());
      for(size_t i = 0; i < gm.numberOfVariables(); ++i) {
         OPENGM_TEST(optimalState[i]<gm.numberOfLabels(i));
      }

      //minimize reduced model
      typedef typename POpt_Data::ReducedGmType ReducedGmType;
      ReducedGmType gmr = p.reducedGraphicalModel();
      opengm::Bruteforce<ReducedGraphicalModelType, AccType> bfr(gmr);
      OPENGM_TEST(bfr.infer()==opengm::NORMAL);
      OPENGM_TEST(bfr.arg(optimalState)==opengm::NORMAL);
      OPENGM_TEST(optimalStateR.size()==gmr.numberOfVariables());
      for(size_t i = 0; i < gmr.numberOfVariables(); ++i) {
         OPENGM_TEST(optimalState[i]<gmr.numberOfLabels(i));
      }

      // compare energy of both minima
      OPENGM_TEST_EQUAL_TOLERANCE(gmr.evaluate(optimalStateR), gm.evaluate(optimalState), 0.00001);

   }*/

   template<class GM>
   PartialOptimalityTester<GM>::~PartialOptimalityTester()
   {
      for(size_t testId = 0; testId < testList.size(); ++testId) {
         delete testList[testId];
      }
   }

   template<class GM>
   template<class POPT>
   void PartialOptimalityTester<GM>::test(const typename POPT::Parameter& infPara, bool tValue, bool tArg, bool tMarg, bool tFacMarg)
   {
      typedef typename GraphicalModelType::ValueType ValueType;
      typedef typename GraphicalModelType::OperatorType OperatorType;
      typedef typename POPT::AccumulationType AccType;
      typedef typename POpt_Data<GM>::ReducedGmType ReducedGmType;

      for(size_t testId = 0; testId < testList.size(); ++testId) {
         size_t numTests = testList[testId]->numberOfTests();
         BlackBoxBehaviour behaviour = testList[testId]->behaviour();
         std::cout << testList[testId]->infoText();
         std::cout << " " << std::flush;
         for(size_t n = 0; n < numTests; ++n) {
            std::cout << "*" << std::flush;
            GraphicalModelType gm = testList[testId]->getModel(n);
            // Create partial optimality object
            POpt_Data<GraphicalModelType> D(gm);
            // Run Algorithm
            bool exceptionFlag = false;
            std::vector<typename GM::LabelType> state;
            try{      
               POPT inf(D);
               InferenceTermination returnValue=inf.infer();
               OPENGM_TEST((returnValue==opengm::NORMAL) || (returnValue==opengm::CONVERGENCE)); 
               if(typeid(AccType) == typeid(opengm::Minimizer) || typeid(AccType) == typeid(opengm::Maximizer)) {
                  if(behaviour == opengm::OPTIMAL) {
                     std::vector<typename GM::LabelType> optimalStateOrig;
                     opengm::Bruteforce<GraphicalModelType, AccType> bfOrig(gm);
                     OPENGM_TEST(bfOrig.infer()==opengm::NORMAL);
                     OPENGM_TEST(bfOrig.arg(optimalStateOrig)==opengm::NORMAL);
                     OPENGM_TEST(optimalStateOrig.size()==gm.numberOfVariables());
                     for(size_t i = 0; i < gm.numberOfVariables(); ++i) {
                        OPENGM_TEST(optimalStateOrig[i]<gm.numberOfLabels(i));
                        OPENGM_TEST(D.getPOpt(i,optimalStateOrig[i]) != false);
                     }

                     ReducedGmType gmRed = D.reducedGraphicalModel();
                     std::vector<typename GM::LabelType> optimalStateRed;
                     opengm::Bruteforce<ReducedGmType, AccType> bfRed(gmRed);
                     OPENGM_TEST(bfRed.infer()==opengm::NORMAL);
                     OPENGM_TEST(bfRed.arg(optimalStateRed)==opengm::NORMAL);
                     OPENGM_TEST(optimalStateRed.size()==gmRed.numberOfVariables());

                     OPENGM_TEST_EQUAL_TOLERANCE(gm.evaluate(optimalStateOrig), gmRed.evaluate(optimalStateRed), 0.00001);
                  }
               }
               } catch(std::exception& e) {
                 exceptionFlag = true;
                 std::cout << e.what() <<std::endl;
               }
            if(behaviour == opengm::FAIL) {
               OPENGM_TEST(exceptionFlag);
            }else{
               OPENGM_TEST(!exceptionFlag);
            }
         }
         if(behaviour == opengm::OPTIMAL) {
            std::cout << " OPTIMAL!" << std::endl;
         }else if(behaviour == opengm::PASS) {
            std::cout << " PASS!" << std::endl;
         }else{
            std::cout << " OK!" << std::endl;
         }
      }
   }

   template<class GM>
   void PartialOptimalityTester<GM>::addTest(BlackBoxTestBase<GraphicalModelType>* test)
   {
      testList.push_back(test);
   }

}

/// \endcond

#endif // #ifndef OPENGM_TEST_PARTIAL_OPTIMALITY_HXX


