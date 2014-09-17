#include <iostream>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

#include <opengm/unittests/popttester.hxx>

#include <opengm/functions/potts.hxx>

#include <opengm/inference/bruteforce.hxx>
#include <opengm/inference/partialOptimality/popt_iterative_relaxed_inf.hxx>
#include <opengm/inference/partialOptimality/popt_dee.hxx>

int main(){ 
   typedef double ValueType;
   typedef opengm::meta::TypeListGenerator
   <
   opengm::PottsFunction<ValueType>,
   opengm::ExplicitFunction<ValueType>
   >::type FunctionTypeList;
   

   typedef opengm::GraphicalModel<ValueType,opengm::Adder,FunctionTypeList>  GmType;
   typedef opengm::POpt_Data<GmType> POpt_DataType;

   typedef opengm::DEE<POpt_DataType,opengm::Minimizer> DEEType;

   typedef opengm::BlackBoxTestGrid<GmType> GridTest;
   typedef opengm::BlackBoxTestFull<GmType> FullTest;
   typedef opengm::BlackBoxTestStar<GmType> StarTest;
   
   opengm::test::PartialOptimalityTester<GmType> tester;
   tester.addTest(new GridTest(3, 3, 3, false, true, GridTest::POTTS, opengm::OPTIMAL, 10));
   tester.addTest(new GridTest(5, 3, 5, false, true, GridTest::POTTS, opengm::OPTIMAL, 10));
   tester.addTest(new StarTest(6,    5, false, true, StarTest::POTTS, opengm::OPTIMAL, 10));
   tester.addTest(new FullTest(5,    5, false, 3,    FullTest::POTTS, opengm::OPTIMAL, 10));
#ifdef WITH_CPLEX
   tester.addTest(new GridTest(15, 15, 5, false, true, GridTest::POTTS, opengm::OPTIMAL, 10));
   tester.addTest(new StarTest(30,     5, false, true, StarTest::POTTS, opengm::OPTIMAL, 10));
   tester.addTest(new FullTest(30,     5, false, 3,    FullTest::POTTS, opengm::OPTIMAL, 10));
#endif

   tester.addTest(new GridTest(3, 3, 3, false, true, GridTest::RANDOM, opengm::OPTIMAL, 10));
   tester.addTest(new GridTest(5, 3, 5, false, true, GridTest::RANDOM, opengm::OPTIMAL, 10));
   tester.addTest(new StarTest(6,    5, false, true, StarTest::RANDOM, opengm::OPTIMAL, 10));
   tester.addTest(new FullTest(5,    5, false, 3,    FullTest::RANDOM, opengm::OPTIMAL, 10));
#ifdef WITH_CPLEX
   tester.addTest(new GridTest(15, 15, 5, false, true, GridTest::RANDOM, opengm::OPTIMAL, 10));
   tester.addTest(new StarTest(30,     5, false, true, StarTest::RANDOM, opengm::OPTIMAL, 10));
   tester.addTest(new FullTest(30,     5, false, 3,    FullTest::RANDOM, opengm::OPTIMAL, 10));
#endif

   typename opengm::DEE::Parameter param;

   std::cout << "Test dee 1" << std::endl;
   param.method_ = DEE1;
   tester.test<DEEType>(param);

   std::cout << "Test dee 3" << std::endl;
   param.method_ = DEE3;
   tester.test<DEEType>(param);

   std::cout << "Test dee 4" << std::endl;
   param.method_ = DEE4;
   tester.test<DEEType>(param);

   return 0;
};

