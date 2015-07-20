#define TRWS_DEBUG_OUTPUT
#define OPENGM_COMBILP_DEBUG
#undef ALREADY_REPARAMETRIZED
#undef TEST_LABELCOLLAPSE_POPULATION
#undef PRINT_LABELING

#include <stdexcept>
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
		std::cerr << "Wrong arguments." << std::endl;
		return EXIT_FAILURE;
	}

	GraphicalModelType gm;
	opengm::hdf5::load(gm, argv[1], "gm");
#ifdef TEST_LABELCOLLAPSE_POPULATION
	std::vector<LabelType> labeling;
#endif

	begin = Clock::now();
	{
		typedef opengm::CombiLP_TRWSi_LC_Gen<
			GraphicalModelType, AccumulatorType,
			opengm::labelcollapse::ReparameterizationTRWS
		> Generator;

		typedef Generator::CombiLPType CombiLPType;
		CombiLPType::Parameter param;
		param.verbose_ = true;
		param.singleReparametrization_ = false;
		param.lpsolverParameter_.verbose_ = true;
#ifdef ALREADY_REPARAMETRIZED
		param.lpsolverParameter_.setTreeAgreeMaxStableIter(0);
		param.lpsolverParameter_.maxNumberOfIterations_= 1;
#else
		param.lpsolverParameter_.setTreeAgreeMaxStableIter(100);
		param.lpsolverParameter_.maxNumberOfIterations_= 10000;
#endif
		param.ilpsolverParameter_.proxy.integerConstraint_ = true;
		param.ilpsolverParameter_.proxy.timeLimit_ = 3600;
		param.ilpsolverParameter_.proxy.workMem_= 1024*10;
		param.ilpsolverParameter_.proxy.numberOfThreads_ = 4;

		CombiLPType::TimingVisitorType visitor(1, 0, true, false, std::numeric_limits<double>::infinity(), 0.0, 2);
		CombiLPType inference(gm, param);
		opengm::InferenceTermination result = inference.infer(visitor);
		if (result != opengm::NORMAL && result != opengm::CONVERGENCE) {
			std::cout << "ERROR: INFERENCE FAILED" << std::endl;
			throw std::runtime_error("Inference failed!");
		}
#if defined(TEST_LABELCOLLAPSE_POPULATION) || defined(PRINT_LABELING)
		inference.arg(labeling);
#endif
	}
	end = Clock::now();
	duration = end - begin;

#ifdef PRINT_LABELING
	std::cout << "LABELING:" << std::endl;
	for (IndexType i = 0; i < gm.numberOfLabels(); ++i) {
		std::cout << " " << labeling[i];
	}
	std::cout << std::endl;
#endif

#ifdef TEST_LABELCOLLAPSE_POPULATION
	{
		using namespace opengm;
		typedef typename LabelCollapseAuxTypeGen<GraphicalModelType>::GraphicalModelType AuxiliaryModelType;
		typedef LPCplex<AuxiliaryModelType, AccumulatorType> Cplex;
		typedef LabelCollapse<GraphicalModelType, Cplex> Inference;

		std::vector<LabelType> targetShape(gm.numberOfVariables());
		labelcollapse::Reordering<GraphicalModelType, AccumulatorType> reordering(gm);
		for (IndexType i = 0; i < gm.numberOfVariables(); ++i) {
			std::vector<LabelType> mapping(gm.numberOfLabels(i));
			reordering.reorder(i);
			reordering.getMapping(mapping.begin());
			targetShape[i] = mapping[ labeling[i] ];
		}

		Inference::Parameter param;
		param.proxy.verbose_ = true;
		param.proxy.integerConstraint_ = true;
		param.proxy.numberOfThreads_ = 4;

		Inference inf(gm, param);
		inf.populate(targetShape.begin());
		inf.infer();
	}
#endif

	// Scripts expect this to be the last line of the log file.
	std::cout << "=> Elapsed ILP time: " << boost::chrono::duration_cast< boost::chrono::duration<double> >(opengm::hack::elapsed) << std::endl;
	std::cout << "=> Total elapsed time: " << duration << std::endl;
}
