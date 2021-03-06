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
#include <opengm/inference/partialOptimality/popt_kovtun.hxx>


int main(){ 
   typedef double ValueType;
   typedef opengm::meta::TypeListGenerator
   <
   opengm::PottsFunction<ValueType>,
   opengm::ExplicitFunction<ValueType>
   //opengm::PottsNFunction<ValueType>,
   //opengm::PottsGFunction<ValueType>,
   //opengm::AbsoluteDifferenceFunction<ValueType>,
   //opengm::SquaredDifferenceFunction<ValueType>,
   //opengm::TruncatedAbsoluteDifferenceFunction<ValueType>,
   //opengm::TruncatedSquaredDifferenceFunction<ValueType>
   >::type FunctionTypeList;
   

   typedef opengm::GraphicalModel<ValueType,opengm::Adder,FunctionTypeList>  GmType;
   typedef opengm::POpt_Data<GmType> POpt_DataType;

   typedef opengm::POpt_Kovtun<POpt_DataType,opengm::Minimizer> POptType;

   typedef opengm::BlackBoxTestGrid<GmType> GridTest;
   typedef opengm::BlackBoxTestFull<GmType> FullTest;
   typedef opengm::BlackBoxTestStar<GmType> StarTest;
   
   opengm::test::PartialOptimalityTester<GmType> tester;
   tester.addTest(new GridTest(3, 3, 3, false, true, GridTest::POTTS, opengm::OPTIMAL, 10));
   tester.addTest(new GridTest(5, 3, 5, false, true, GridTest::POTTS, opengm::OPTIMAL, 10));
   tester.addTest(new StarTest(6,    5, false, true, StarTest::POTTS, opengm::OPTIMAL, 10));
   tester.addTest(new FullTest(5,    5, false, 3,    FullTest::POTTS, opengm::OPTIMAL, 10));
#ifdef WITH_CPLEX // bigger models can be checked for optimality if cplex solves them in tester
   tester.addTest(new GridTest(15, 15, 5, false, true, GridTest::POTTS, opengm::OPTIMAL, 10));
   tester.addTest(new StarTest(30,     5, false, true, StarTest::POTTS, opengm::OPTIMAL, 10));
   tester.addTest(new FullTest(30,     5, false, 3,    FullTest::POTTS, opengm::OPTIMAL, 10));
#endif

#ifdef WITH_QPBO
   std::cout << "Test Kovtun" << std::endl;
   tester.test<POptType>();
#else
   std::cout <<"Cannot test Kovtun, no QPBO installed" << std::endl;
#endif
  
   return 0;
};
