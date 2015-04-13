#define TRWS_DEBUG_OUTPUT
#undef COMBILP_STOP_AFTER_REPARAMETRIZATION
#undef CPLEX_DUMP_SEQUENTIALLY

#include <boost/chrono.hpp>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/inference/combilp_default.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>

int main(int argc, char **argv)
{
	typedef boost::chrono::steady_clock Clock;
	Clock::time_point begin, end;
	boost::chrono::duration<double> duration;

	typedef opengm::GraphicalModel<double, opengm::Adder> GraphicalModelType;
	GraphicalModelType gm;

	if (argc != 2) {
		std::cerr << "Missing hdf5 filename argument." << std::endl;
		return EXIT_FAILURE;
	}
	opengm::hdf5::load(gm, argv[1], "gm");

	std::cout << ":: Benchmarking CombiLP + TRWSi + CPLEX ..." << std::endl;
	begin = Clock::now();
	{
		typedef opengm::CombiLP_TRWSi_Gen<GraphicalModelType, opengm::Minimizer> Generator;
		typedef Generator::CombiLPType CombiLPType;
		CombiLPType::Parameter param;
		param.verbose_ = true;
		param.lpsolverParameter_.verbose_ = true;
		param.lpsolverParameter_.setTreeAgreeMaxStableIter(100);
		param.lpsolverParameter_.maxNumberOfIterations_=1000;
		param.ilpsolverParameter_.integerConstraint_ = true;
		param.ilpsolverParameter_.timeLimit_ = 3600;
		param.ilpsolverParameter_.workMem_= 1024*6;
		CombiLPType inference(gm, param);
		inference.infer();
	}
	end = Clock::now();
	duration = end - begin;
	std::cout << "=> Total elapsed time: " << duration << std::endl;

	begin = Clock::now();
	std::cout << "Benchmarking CombiLP + TRWSi + LabelCollpse + CPLEX ..." << std::endl;
	{
		typedef opengm::CombiLP_TRWSi_LC_Gen<GraphicalModelType, opengm::Minimizer> Generator;
		typedef Generator::CombiLPType CombiLPType;
		CombiLPType::Parameter param;
		param.verbose_ = true;
		param.lpsolverParameter_.verbose_ = true;
		param.lpsolverParameter_.setTreeAgreeMaxStableIter(100);
		param.lpsolverParameter_.maxNumberOfIterations_=1000;
		param.ilpsolverParameter_.proxy.integerConstraint_ = true;
		param.ilpsolverParameter_.proxy.timeLimit_ = 3600;
		param.ilpsolverParameter_.proxy.workMem_= 1024*6;
		CombiLPType inference(gm, param);
		inference.infer();
	}
	end = Clock::now();
	duration = end - begin;
	std::cout << "=> Total elapsed time: " << duration << std::endl;
}
