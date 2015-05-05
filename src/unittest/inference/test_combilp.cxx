#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>

#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/inference/combilp.hxx>


int main() {
	typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;
	typedef opengm::BlackBoxTestGrid<GraphicalModelType> GridTest;
	typedef opengm::BlackBoxTestFull<GraphicalModelType> FullTest;
	typedef opengm::BlackBoxTestStar<GraphicalModelType> StarTest;

	opengm::InferenceBlackBoxTester<GraphicalModelType> tester;
	tester.addTest(new GridTest(1,  4,  4, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 2000));
	tester.addTest(new GridTest(1,  4,  8, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 100));
	tester.addTest(new GridTest(1,  4, 20,  true,  true, GridTest::RANDOM, opengm::OPTIMAL, 10));
	tester.addTest(new GridTest(1, 10,  3, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 10));
	tester.addTest(new GridTest(1,  5,  8, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 10));

	tester.addTest(new GridTest(3,  3,  5, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 5));
	tester.addTest(new StarTest(5,      6, false,  true, StarTest::RANDOM, opengm::OPTIMAL, 20));

	std::cout << "Test CombiLP ..." << std::endl;
	{
		typedef opengm::TRWSi<GraphicalModelType,opengm::Minimizer> LPSolverType;
		typedef opengm::Bruteforce<opengm::CombiLP_ILP_TypeGen<LPSolverType>::GraphicalModelType,opengm::Minimizer> ILPSolverType;
		typedef opengm::CombiLP<GraphicalModelType,opengm::Minimizer,LPSolverType,ILPSolverType> CombiLPType;
		CombiLPType::Parameter param;
		param.lpsolverParameter_.maxNumberOfIterations_=100;
		tester.test<CombiLPType>(param);
	}
}
