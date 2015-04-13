#define TRWS_DEBUG_OUTPUT
#undef COMBILP_STOP_AFTER_REPARAMETRIZATION
#undef CPLEX_DUMP_SEQUENTIALLY

#include <boost/chrono.hpp>

#include <opengm/functions/explicit_function.hxx>
#include "opengm/functions/fieldofexperts.hxx"
#include <opengm/functions/pottsg.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/combilp_default.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/utilities/metaprogramming.hxx>

int main(int argc, char **argv)
{
	typedef double ValueType;
	typedef size_t IndexType;
	typedef size_t LabelType;
	typedef opengm::Adder OperatorType;
	typedef opengm::Minimizer AccumulatorType;
	typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;

	typedef opengm::meta::TypeListGenerator<
		opengm::ExplicitFunction<ValueType, IndexType, LabelType>,
		opengm::PottsFunction<ValueType, IndexType, LabelType>,
		opengm::PottsNFunction<ValueType, IndexType, LabelType>,
		opengm::PottsGFunction<ValueType, IndexType, LabelType>,
		opengm::TruncatedSquaredDifferenceFunction<ValueType, IndexType, LabelType>,
		opengm::TruncatedAbsoluteDifferenceFunction<ValueType, IndexType, LabelType>,
		opengm::FoEFunction<ValueType, IndexType, LabelType>
	>::type FunctionTypes;

	typedef opengm::GraphicalModel<ValueType, OperatorType, FunctionTypes> GraphicalModelType;

	typedef boost::chrono::steady_clock Clock;
	Clock::time_point begin, end;
	boost::chrono::duration<double> duration;

	if (argc != 2) {
		std::cerr << "Missing hdf5 filename argument." << std::endl;
		return EXIT_FAILURE;
	}

	GraphicalModelType gm;
	opengm::hdf5::load(gm, argv[1], "gm");

	std::cout << ":: Benchmarking CombiLP + TRWSi + CPLEX ..." << std::endl;
	begin = Clock::now();
	{
		typedef opengm::CombiLP_TRWSi_Gen<GraphicalModelType, AccumulatorType> Generator;
		typedef Generator::CombiLPType CombiLPType;
		CombiLPType::Parameter param;
		param.verbose_ = true;
		param.singleReparametrization_ = true;
		param.lpsolverParameter_.verbose_ = true;
		param.lpsolverParameter_.setTreeAgreeMaxStableIter(100);
		param.lpsolverParameter_.maxNumberOfIterations_= 10000;
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
	std::cout << ":: Benchmarking CombiLP + TRWSi + LabelCollapse + CPLEX ..." << std::endl;
	{
		typedef opengm::CombiLP_TRWSi_LC_Gen<GraphicalModelType, AccumulatorType> Generator;
		typedef Generator::CombiLPType CombiLPType;
		CombiLPType::Parameter param;
		param.verbose_ = true;
		param.singleReparametrization_ = true;
		param.lpsolverParameter_.verbose_ = true;
		param.lpsolverParameter_.setTreeAgreeMaxStableIter(100);
		param.lpsolverParameter_.maxNumberOfIterations_= 10000;
		param.ilpsolverParameter_.proxy.integerConstraint_ = true;
		param.ilpsolverParameter_.proxy.timeLimit_ = 3600;
		param.ilpsolverParameter_.proxy.workMem_= 1024*6;
		CombiLPType inference(gm, param);
		inference.infer();
	}
	end = Clock::now();
	duration = end - begin;
	std::cout << "=> Total elapsed time: " << duration << std::endl;

	std::cout << ":: Benchmarking Dense CombiLP + TRWSi + CPLEX ..." << std::endl;
	begin = Clock::now();
	{
		typedef opengm::CombiLP_TRWSi_Gen<GraphicalModelType, AccumulatorType> Generator;
		typedef Generator::CombiLPType CombiLPType;
		CombiLPType::Parameter param;
		param.verbose_ = true;
		param.singleReparametrization_ = false;
		param.lpsolverParameter_.verbose_ = true;
		param.lpsolverParameter_.setTreeAgreeMaxStableIter(100);
		param.lpsolverParameter_.maxNumberOfIterations_= 10000;
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
	std::cout << ":: Benchmarking Dense CombiLP + TRWSi + LabelCollapse + CPLEX ..." << std::endl;
	{
		typedef opengm::CombiLP_TRWSi_LC_Gen<GraphicalModelType, AccumulatorType> Generator;
		typedef Generator::CombiLPType CombiLPType;
		CombiLPType::Parameter param;
		param.verbose_ = true;
		param.lpsolverParameter_.verbose_ = true;
		param.lpsolverParameter_.setTreeAgreeMaxStableIter(100);
		param.lpsolverParameter_.maxNumberOfIterations_= 10000;
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
