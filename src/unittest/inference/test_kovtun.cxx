#include <stdlib.h>
#include <vector>
#include <set>
#include <functional>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/maximizer.hxx>

#include <opengm/unittests/popttester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

#include <opengm/inference/partialOptimality/popt_kovtun.hxx>

int main() {
   typedef opengm::GraphicalModel<double, opengm::Adder > AdderGmType;
   typedef opengm::BlackBoxTestGrid<AdderGmType> AdderGridTest;
   typedef opengm::BlackBoxTestFull<AdderGmType> AdderFullTest;
   typedef opengm::BlackBoxTestStar<AdderGmType> AdderStarTest;
   std::cout << "Kovtun Tests" << std::endl;
   {
      opengm::PartialOptimalityTester<AdderGmType> adderTester;
      adderTester.addTest(new AdderFullTest(1, 90, false, 1, AdderFullTest::POTTS, opengm::OPTIMAL, 20));
      adderTester.addTest(new AdderGridTest(5, 4, 5, false, true, AdderGridTest::POTTS, opengm::PASS, 2));
      adderTester.addTest(new AdderGridTest(4, 4, 5, false, false, AdderGridTest::POTTS, opengm::PASS, 2));
      adderTester.addTest(new AdderStarTest(10, 2, false, true, AdderStarTest::POTTS, opengm::OPTIMAL, 20));
      adderTester.addTest(new AdderStarTest(5, 5, false, true, AdderStarTest::POTTS, opengm::OPTIMAL, 20));
      adderTester.addTest(new AdderStarTest(10, 10, false, true, AdderStarTest::POTTS, opengm::PASS, 2));
      adderTester.addTest(new AdderGridTest(3, 3, 3, false, true, AdderGridTest::POTTS, opengm::OPTIMAL, 20));
      adderTester.addTest(new AdderGridTest(5, 5, 4, false, true, AdderGridTest::POTTS, opengm::PASS, 2));
      adderTester.addTest(new AdderStarTest(10, 4, false, true, AdderStarTest::POTTS, opengm::PASS, 20));
      adderTester.addTest(new AdderStarTest(10, 2, false, false, AdderStarTest::POTTS, opengm::OPTIMAL, 20));
      adderTester.addTest(new AdderStarTest(10, 4, false, true, AdderStarTest::POTTS, opengm::PASS, 20));
      adderTester.addTest(new AdderStarTest(6, 4, false, true, AdderStarTest::POTTS, opengm::PASS, 4));
      adderTester.addTest(new AdderFullTest(5, 3, false, 3, AdderFullTest::POTTS, opengm::PASS, 20));
      //adderTester.addTest(new AdderGridTest(400, 400, 5, false, false, AdderGridTest::POTTS, opengm::PASS, 1));
   
      {
         std::cout << "  * Minimization/Adder ..." << std::endl;
         typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
         typedef opengm::POpt_Data<GmType> DataType;
         typedef opengm::POpt_Kovtun<DataType, opengm::Minimizer> KovtunType;
         KovtunType::Parameter para;
         adderTester.test<KovtunType>(para);
         std::cout << " OK!" << std::endl;
      }

    }
   return 0;
}



