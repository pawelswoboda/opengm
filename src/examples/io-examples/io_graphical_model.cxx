#undef NDEBUG

#define TRWS_DEBUG_OUTPUT

#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/fieldofexperts.hxx>
#include <opengm/functions/pottsg.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/functions/truncated_absolute_difference.hxx>
#include <opengm/functions/truncated_squared_difference.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_manipulator.hxx>
#include <opengm/inference/combilp_default.hxx>
#include <opengm/inference/trws/trws_base.hxx>
#include <opengm/inference/trws/trws_reparametrization.hxx>
#include <opengm/inference/trws/trws_temporary.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/utilities/indexing.hxx>
#include <opengm/utilities/metaprogramming.hxx>

int
main
(
	int argc,
	char **argv
)
{
	typedef size_t IndexType;
	typedef size_t LabelType;
	typedef double ValueType;
	typedef opengm::meta::TypeListGenerator<
		opengm::ExplicitFunction<ValueType, IndexType, LabelType>,
		opengm::PottsFunction<ValueType, IndexType, LabelType>,
		opengm::PottsNFunction<ValueType, IndexType, LabelType>,
		opengm::PottsGFunction<ValueType, IndexType, LabelType>,
		opengm::TruncatedSquaredDifferenceFunction<ValueType, IndexType, LabelType>,
		opengm::TruncatedAbsoluteDifferenceFunction<ValueType, IndexType, LabelType>,
		opengm::FoEFunction<ValueType, IndexType, LabelType>
	>::type FunctionTypes;
	typedef opengm::Minimizer AccumulationType;
	typedef opengm::GraphicalModel<ValueType, opengm::Adder, FunctionTypes> GraphicalModelType;
	typedef opengm::trws_base::DecompositionStorage<GraphicalModelType> DecompositionStorageType;

	GraphicalModelType gm;

	opengm::hdf5::load(gm, argv[1], "gm");
	DecompositionStorageType storage(gm, DecompositionStorageType::GENERALSTRUCTURE, NULL);

	typedef opengm::hack::SequenceGeneratorIterator<GraphicalModelType, GraphicalModelType> GeneratorType;
	typename GeneratorType::Iterators its = GeneratorType::makeIterators(storage);

#if 0
	{
		typedef opengm::hack::CanonicalReparametrizer<GraphicalModelType> ReparametrizerType;
		ReparametrizerType::reparametrizeAll(gm, its.first, its.second);
	}

	{
		typedef opengm::hack::UniformReparametrizer<GraphicalModelType> ReparametrizerType;
		ReparametrizerType::reparametrizeAll(gm, its.first, its.second);
	}
#endif

	{
		typedef opengm::hack::CustomReparametrizer<GraphicalModelType> ReparametrizerType;
		ReparametrizerType::reparametrizeAll(gm, its.first, its.second);
	}
}
