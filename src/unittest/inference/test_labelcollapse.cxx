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
	typedef opengm::GraphicalModel<double, opengm::Adder> OriginalModelType;
	typedef opengm::LabelCollapseAuxTypeGen<OriginalModelType>::GraphicalModelType AuxiliaryModelType;
	typedef opengm::BlackBoxTestGrid<OriginalModelType> GridTest;
	typedef opengm::BlackBoxTestFull<OriginalModelType> FullTest;
	typedef opengm::BlackBoxTestStar<OriginalModelType> StarTest;

	// We really need more than two states per variable, otherwise no labels
	// can get collapsed.
	opengm::InferenceBlackBoxTester<OriginalModelType> tester;

	// At first some rather small amount of variables (chain like) with
	// different settings and increasing state space.
	//
	// The first pass is run with Bruteforce as proxy inference and the second
	// pass uses CombiLP. CombiLP does not handle problems without unary terms.
	// So these problems will only be used for the first pass. (Problems
	// without a unary term are a good test case, because most of the bugs were
	// found using this kind of problem and they are simpler to debug.)
	tester.addTest(new GridTest(1,  2,   4, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 1000));
	tester.addTest(new GridTest(1,  2,   8, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 100));
	tester.addTest(new GridTest(1,  2, 100,  true,  true, GridTest::RANDOM, opengm::OPTIMAL, 3));
	tester.addTest(new GridTest(1, 10,   3, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 10));
	tester.addTest(new GridTest(1,  5,   8, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 1));

	// Now some more variables, but less states and iterations. This is very
	// time consuming to brute force. :-(
	tester.addTest(new GridTest(2,  2,   6, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 20));
	tester.addTest(new StarTest(5,       6, false,  true, StarTest::RANDOM, opengm::OPTIMAL, 20));

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
