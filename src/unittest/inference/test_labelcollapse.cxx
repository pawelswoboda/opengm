#include <iostream>
#include <vector>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/inference/bruteforce.hxx>
#include <opengm/inference/labelcollapse.hxx>
#include <opengm/inference/lpcplex.hxx>

#include <opengm/unittests/blackboxtester.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestgrid.hxx>
#include <opengm/unittests/blackboxtests/blackboxtestfull.hxx>
#include <opengm/unittests/blackboxtests/blackboxteststar.hxx>


int main()
{
	srand(time(0));

	typedef opengm::GraphicalModel<double, opengm::Adder> OriginalModelType;
	typedef opengm::LabelCollapseAuxTypeGen<OriginalModelType>::GraphicalModelType AuxiliaryModelType;
	typedef opengm::BlackBoxTestGrid<OriginalModelType> GridTest;
	typedef opengm::BlackBoxTestFull<OriginalModelType> FullTest;
	typedef opengm::BlackBoxTestStar<OriginalModelType> StarTest;

	opengm::InferenceBlackBoxTester<OriginalModelType> tester;

	tester.addTest(new GridTest(1,  4,  4, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 2000));
	tester.addTest(new GridTest(1,  4,  8, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 100));
	tester.addTest(new GridTest(1,  4, 20,  true,  true, GridTest::RANDOM, opengm::OPTIMAL, 10));
	tester.addTest(new GridTest(1, 10,  3, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 10));
	tester.addTest(new GridTest(1,  5,  8, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 10));
	tester.addTest(new GridTest(3,  3,  5, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 5));
	tester.addTest(new StarTest(5,      6, false,  true, StarTest::RANDOM, opengm::OPTIMAL, 20));

	std::cout << "Test LabelCollapse ..." << std::endl;

	std::cout << "  * Test Min-Sum with Bruteforce" << std::endl;
	{
		typedef opengm::Bruteforce<AuxiliaryModelType, opengm::Minimizer> Proxy;
		typedef opengm::LabelCollapse<OriginalModelType, Proxy> Inference;
		Inference::Parameter parameter;
		tester.test<Inference>(parameter);
	}

#ifdef WITH_CPLEX
	std::cout << "  * Test Min-Sum with CPLEX" << std::endl;
	{
		typedef opengm::LPCplex<AuxiliaryModelType, opengm::Minimizer> Proxy;
		typedef opengm::LabelCollapse<OriginalModelType, Proxy> Inference;

		Proxy::Parameter proxy_parameter;
		proxy_parameter.integerConstraint_ = true;

		Inference::Parameter parameter(proxy_parameter);
		tester.test<Inference>(parameter);
	}
#endif
}
