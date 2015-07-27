#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/operations/multiplier.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>

#include "opengm/inference/partialOptimality/popt_inference_base.hxx"
#include "opengm/inference/partialOptimality/popt_data.hxx"

#include <string>
#include "debug/logs.h"
#include "part_opt_opengm.h"
#include "files/xfs.h"
#include "dynamic/options.h"

#include <string>
#include <iostream>

using namespace opengm;

int main(int argc, char *argv[]){
	const char * file;
	std::string root;
	root = xfs::getPath(argv[0]);
	printf("my path: %s\n", root.c_str());
	if (argc < 2){
		file = "../../data/pdb1kwh.h5";
		printf("Usage: test_part_opt_opengm <problem> options\n");
		printf("No arguments provided, running test: %s\n", file);
	} else{
		file = argv[1];
	};

	//parese options
	options ops;
	for (int i = 2; i < argc; ++i){
		std::stringstream o(std::string(argv[i]).c_str());
		std::string o_name;
		std::string o_val;
		std::getline(o, o_name, '=');
		std::getline(o, o_val, '=');
		debug::stream << "option " << o_name << " = " << o_val << "\n";
		ops[o_name] = atof(o_val.c_str());
	};

	//file = "../../data/PBP-bug.h5";
	//file = "Z:/work/dev/matlab/part_opt/datasets/color-seg-n4/pfau-small.h5";
	//file = "Z:/work/dev/matlab/part_opt/datasets/mrf-stereo/tsu-gm.h5";
	//file = "Z:/work/dev/matlab/part_opt/datasets/protein-folding/pdb1iqc.h5";
	//typedef d_type ValueType;
	//typedef int ValueType;
	typedef double ValueType;
	typedef opengm::meta::TypeListGenerator
		<
		opengm::PottsFunction<ValueType>,
		opengm::PottsNFunction<ValueType>,
		opengm::ExplicitFunction<ValueType>,
		opengm::TruncatedSquaredDifferenceFunction<ValueType>,
		opengm::TruncatedAbsoluteDifferenceFunction<ValueType>
		> ::type FunctionTypeList;


	typedef opengm::GraphicalModel<ValueType, opengm::Adder, FunctionTypeList>  GmType;
	typedef opengm::POpt_Data<GmType> POpt_DataType;

	// load a test model
	//
	std::string modelFilename = file;
	GmType * gm = new GmType();
	debug::stream << "Loading model " << modelFilename << "\n";
	opengm::hdf5::load(*gm, modelFilename, "gm");
	debug::stream << "Loading model done\n";

	typedef POpt_Data<GmType> DATA;
	typedef opengm::Minimizer ACC;
	DATA data(*gm);
	// invoke my solver
	part_opt_opengm<DATA, opengm::Adder> alg(data);
	alg.instance_name = modelFilename;

	(*alg.ops) << ops;
	alg.infer();
		
	std::cout << "time_total: " << alg.time_total << "\n";
	std::cout << "time_init: " << alg.time_init << "\n";
	std::cout << "time_po: " << alg.time_po << "\n";
	std::cout << "iter_init_trws: " << alg.iter_init_trws << "\n";
	std::cout << "iter_po: " << alg.iter_po << "\n";
	std::cout << "iter_po_trws: " << alg.iter_po_trws << "\n";
	delete gm;

	std::cin.get();
	return 0;
};