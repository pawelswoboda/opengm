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
	typedef opengm::labelcollapse::AuxiliaryModelTypeGenerator<OriginalModelType>::GraphicalModelType AuxiliaryModelType;
	typedef opengm::BlackBoxTestGrid<OriginalModelType> GridTest;
	typedef opengm::BlackBoxTestFull<OriginalModelType> FullTest;
	typedef opengm::BlackBoxTestStar<OriginalModelType> StarTest;

	// We really need more than two states per variable, otherwise no labels
	// can get collapsed.
	opengm::InferenceBlackBoxTester<OriginalModelType> allTester, bruteforceTester;

	// At first some rather small amount of variables (chain like) with
	// different settings and increasing state space.
	//
	// The first pass is run with Bruteforce as proxy inference and the second
	// pass uses CombiLP. CombiLP does not handle problems without unary terms.
	// So these problems will only be used for the first pass. (Problems
	// without a unary term are a good test case, because most of the bugs were
	// found using this kind of problem and they are simpler to debug.)
	allTester.addTest(new GridTest(1,  2,   4, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 1000));
	allTester.addTest(new GridTest(1,  2,   8, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 100));
	allTester.addTest(new GridTest(1,  2, 100,  true,  true, GridTest::RANDOM, opengm::OPTIMAL, 3));
	allTester.addTest(new GridTest(1, 10,   3, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 10));
	allTester.addTest(new GridTest(1,  5,   8, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 1));

	bruteforceTester.addTest(new GridTest(1,  2,   4, false, false, GridTest::RANDOM, opengm::OPTIMAL, 1000));
	bruteforceTester.addTest(new GridTest(1,  2,   8, false, false, GridTest::RANDOM, opengm::OPTIMAL, 100));
	bruteforceTester.addTest(new GridTest(1,  2, 100, false, false, GridTest::RANDOM, opengm::OPTIMAL, 30));

	// Now some more variables, but less states and iterations. This is very
	// time consuming to brute force. :-(
	allTester.addTest(new GridTest(2,  2,   6, false,  true, GridTest::RANDOM, opengm::OPTIMAL, 20));
	allTester.addTest(new StarTest(5,       6, false,  true, StarTest::RANDOM, opengm::OPTIMAL, 20));

	bruteforceTester.addTest(new GridTest(2,  2,   6, false, false, GridTest::RANDOM, opengm::OPTIMAL, 20));
	bruteforceTester.addTest(new StarTest(5,       6, false, false, StarTest::RANDOM, opengm::OPTIMAL, 20));
	bruteforceTester.addTest(new FullTest(3,       6, false,     2, FullTest::RANDOM, opengm::OPTIMAL, 20));
	// Note above: FullTest also uses no unary terms. Does not work with CombiLP.

	std::cout << "Test LabelCollapse ..." << std::endl;

	std::cout << "  * Test Min-Sum with Bruteforce" << std::endl;
	{
		typedef opengm::Bruteforce<AuxiliaryModelType, opengm::Minimizer> Proxy;
		typedef opengm::labelcollapse::Inference<OriginalModelType, Proxy> Inference;
		Inference::Parameter parameter;
		allTester.test<Inference>(parameter);
		bruteforceTester.test<Inference>(parameter);
	}

#ifdef WITH_CPLEX
	std::cout << "  * Test Min-Sum with CPLEX" << std::endl;
	{
		typedef opengm::LPCplex<AuxiliaryModelType, opengm::Minimizer> Proxy;
		typedef opengm::labelcollapse::Inference<OriginalModelType, Proxy> Inference;

		Proxy::Parameter proxy_parameter;
		proxy_parameter.integerConstraint_ = true;

		Inference::Parameter parameter(proxy_parameter);
		allTester.test<Inference>(parameter);
		bruteforceTester.test<Inference>(parameter);
	}
#endif
}
