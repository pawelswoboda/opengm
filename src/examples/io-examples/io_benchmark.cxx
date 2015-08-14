#include <iostream>
#include <stdexcept>

#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/pottsg.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/functions/truncated_absolute_difference.hxx>
#include <opengm/functions/truncated_squared_difference.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/lpcplex.hxx>
#include <opengm/inference/partialOptimality/popt_infer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/utilities/metaprogramming.hxx>

template<class GM>
std::vector<typename GM::LabelType>
runCPLEX
(
	const GM &gm
)
{
	typedef opengm::LPCplex<GM, opengm::Minimizer> Inference;

	typename Inference::Parameter param;
	param.integerConstraint_ = true;
	param.timeLimit_ = 3600;
	param.workMem_= 1024*10;
	param.numberOfThreads_ = 4;

	Inference inference(gm, param);
	opengm::InferenceTermination result = inference.infer();
	if (result != opengm::NORMAL && result != opengm::CONVERGENCE) {
		std::cout << "ERROR: INFERENCE FAILED" << std::endl;
		throw std::runtime_error("Inference failed!");
	}

	std::vector<typename GM::LabelType> labeling;
	inference.arg(labeling);
	return labeling;
}

template<class T>
void
printLabeling
(
	const std::vector<T> &labeling,
	const std::string &prefix = ""
)
{
	if (! prefix.empty())
		std::cout << prefix;

	typename std::vector<T>::const_iterator it = labeling.begin();
	if (it != labeling.end())
		std::cout << *it;

	for (++it; it != labeling.end(); ++it)
		std::cout << " " << *it;

	std::cout << std::endl;
}

int main(int argc, char **argv)
{
	typedef double ValueType;
	typedef size_t IndexType;
	typedef size_t LabelType;
	typedef opengm::Adder OperatorType;
	typedef opengm::Minimizer AccumulationType;
	typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;

	typedef opengm::meta::TypeListGenerator<
		opengm::ExplicitFunction<ValueType, IndexType, LabelType>,
		opengm::PottsFunction<ValueType, IndexType, LabelType>,
		opengm::PottsNFunction<ValueType, IndexType, LabelType>,
		opengm::PottsGFunction<ValueType, IndexType, LabelType>,
		opengm::TruncatedSquaredDifferenceFunction<ValueType, IndexType, LabelType>,
		opengm::TruncatedAbsoluteDifferenceFunction<ValueType, IndexType, LabelType>
	>::type FunctionTypes;

	typedef opengm::GraphicalModel<ValueType, OperatorType, FunctionTypes> GraphicalModelType;

	if (argc != 2) {
		std::cerr << "Wrong number of arguments." << std::endl;
		return EXIT_FAILURE;
	}

	GraphicalModelType gm;
	opengm::hdf5::load(gm, argv[1], "gm");

	//
	// Calculate reduced model.
	//

	typedef opengm::POpt_infer<GraphicalModelType, opengm::Minimizer> POptInferType;
	typedef opengm::POpt_Data<GraphicalModelType> POptDataType;
	typedef typename POptDataType::ReducedGmType ReducedModelType;

	ReducedModelType rm;
	typename POptInferType::Parameter poptparam;
	poptparam.methodSequence_.push_back(POptInferType::Parameter::IRI_SHEKHOVTSOV);
	POptInferType popt(gm, poptparam);
	popt.infer();
	popt.getPOpt_Data().reducedGraphicalModel(rm);

	//
	// Run inference on original and reduced model.
	//

	std::vector<LabelType> labelingOriginal = runCPLEX(gm);
	std::vector<LabelType> labelingReduced = runCPLEX(rm);

	//
	// Convert labeling of reduced model and check if the solutions match.
	//

	std::vector<LabelType> labelingConvertetd;
	popt.getPOpt_Data().ReducedToOriginalLabeling(labelingConvertetd, labelingReduced);

	printLabeling(labelingOriginal,   "CPLEX labeling: ");
	printLabeling(labelingConvertetd, "POpt labeling:  ");

	ValueType solOriginal = gm.evaluate(labelingOriginal.begin());
	ValueType solConverted = gm.evaluate(labelingConvertetd.begin());

	std::cout << std::endl
	          << "CPLEX solution: " << solOriginal << std::endl
	          << "POpt solution:  " << solConverted << std::endl;

	if (! opengm::isNumericEqual(solOriginal, solConverted)) {
		std::cout << std::endl
		          << "!!" << std::endl << "!!" << std::endl
		          << "!! SOLUTIONS DO NOT MATCH" << std::endl
		          << "!!" << std::endl << "!!" << std::endl;
	}
}
