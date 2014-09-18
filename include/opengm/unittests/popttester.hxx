#pragma once
#ifndef OPENGM_TEST_PARTIAL_OPTIMALITY_HXX
#define OPENGM_TEST_PARTIAL_OPTIMALITY_HXX

#include <vector>
#include <typeinfo>

#include <opengm/opengm.hxx>
#include <opengm/unittests/test.hxx>
#include <opengm/inference/bruteforce.hxx>
#ifdef WITH_CPLEX
#include <opengm/inference/lpcplex.hxx>
#endif
#include <opengm/unittests/blackboxtests/blackboxtestbase.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/integrator.hxx>

#include <opengm/inference/partialOptimality/popt_data.hxx>
#include <opengm/inference/partialOptimality/popt_infer.hxx>

/// \cond HIDDEN_SYMBOLS

namespace opengm {
namespace test {
   template<class GM>
   class PartialOptimalityTester
   {
   public:
      typedef GM GraphicalModelType;
      template<class POPT> void test(const typename POPT::Parameter& param = typename POPT::Parameter()); //, bool tValue=true, bool tArg=false, bool tMarg=false, bool tFacMarg=false);
      void addTest(BlackBoxTestBase<GraphicalModelType>*);
      ~PartialOptimalityTester();

   private:
      std::vector<BlackBoxTestBase<GraphicalModelType>*> testList;
   };

   //***************
   //IMPLEMENTATION
   //***************

   template<class GM>
   PartialOptimalityTester<GM>::~PartialOptimalityTester()
   {
      for(size_t testId = 0; testId < testList.size(); ++testId) {
         delete testList[testId];
      }
   }

   template<class GM>
   template<class POPT>
   void PartialOptimalityTester<GM>::test(const typename POPT::Parameter& infPara) //, bool tValue, bool tArg, bool tMarg, bool tFacMarg)
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
               POPT inf(D,infPara);
               InferenceTermination returnValue=inf.infer();
               OPENGM_TEST((returnValue==opengm::NORMAL) || (returnValue==opengm::CONVERGENCE));
               if(typeid(AccType) == typeid(opengm::Minimizer) || typeid(AccType) == typeid(opengm::Maximizer)) {
                  if(behaviour == opengm::OPTIMAL) {
                     std::vector<typename GM::LabelType> optimalStateOrig, optimalStateOrig2;
#ifdef WITH_CPLEX
                     typename opengm::LPCplex<GraphicalModelType,AccType>::Parameter origParam;
                     origParam.integerConstraint_ = false;
                     opengm::LPCplex<GraphicalModelType,AccType> origSolver(gm,origParam);
#else
                     opengm::Bruteforce<GraphicalModelType, AccType> origSolver(gm);
#endif
                     OPENGM_TEST(origSolver.infer()==opengm::NORMAL);
                     OPENGM_TEST(origSolver.arg(optimalStateOrig)==opengm::NORMAL);
                     OPENGM_TEST(origSolver.arg(optimalStateOrig2,2)==opengm::NORMAL);
                     OPENGM_TEST(optimalStateOrig.size()==gm.numberOfVariables());
                     // to do: check if solution unique, otherwise test may not be correct
                     //if(gm.evaluate(optimalStateOrig.begin()) < gm.evaluate(optimalStateOrig2.begin()) - 0.00001) {
                     //   for(size_t var = 0; var < gm.numberOfVariables(); ++var) {

                           //std::cout << "node = " << var << ", optimal label = " << optimalStateOrig[var] << ", partial Optimality = ";
                           //if(D.getPOpt(var,optimalStateOrig[var]) == false) std::cout << "false";
                           //else if(D.getPOpt(var,optimalStateOrig[var]) == true) std::cout << "true";
                           //else std::cout << "maybe";
                           //std::cout << std::endl;

                     //      OPENGM_TEST(optimalStateOrig[var]<gm.numberOfLabels(var));
                     //      OPENGM_TEST(D.getPOpt(var,optimalStateOrig[var]) != false);
                     //   }
                     //} else {
                     //   OPENGM_TEST_EQUAL_TOLERANCE(gm.evaluate(optimalStateOrig), gm.evaluate(optimalStateOrig2), 0.00001);
                     //   std::cout << "graphical model possesses more than one minimum" << std::endl;
                     //}


                     // to do: this test is always applicable, even when solution is not unique
                     ReducedGmType gmRed;
                     D.reducedGraphicalModel(gmRed);
                     std::vector<typename GM::LabelType> optimalStateRed;
#ifdef WITH_CPLEX
                     typename opengm::LPCplex<ReducedGmType,AccType>::Parameter redParam;
                     redParam.integerConstraint_ = false;
                     opengm::LPCplex<ReducedGmType,AccType> redSolver(gmRed,redParam);
#else
                     opengm::Bruteforce<ReducedGmType, AccType> redSolver(gmRed);
#endif
                     OPENGM_TEST(redSolver.infer()==opengm::NORMAL);
                     OPENGM_TEST(redSolver.arg(optimalStateRed)==opengm::NORMAL);
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

} // end namespace test
} // end namespace opengm

/// \endcond

#endif // #ifndef OPENGM_TEST_PARTIAL_OPTIMALITY_HXX


