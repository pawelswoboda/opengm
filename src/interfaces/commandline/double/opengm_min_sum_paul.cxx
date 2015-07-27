#include <iostream>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/functions/pottsg.hxx>
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#include "../cmd_interface.hxx"

//inference caller
#include "../../common/caller/popt_caller.hxx"


using namespace opengm;

int main(int argc, char** argv) {
   if(argc < 2) {
      std::cerr << "At least one input argument required" << std::endl;
      std::cerr << "try \"-h\" for help" << std::endl;
      return 1;
   }

   typedef double ValueType;
   typedef size_t IndexType;
   typedef size_t LabelType;
   typedef Adder OperatorType;
   typedef Minimizer AccumulatorType;
   typedef interface::IOCMD InterfaceType;
   typedef DiscreteSpace<IndexType, LabelType> SpaceType;

   // Set functions for graphical model

   typedef meta::TypeListGenerator<
      opengm::ExplicitFunction<ValueType, IndexType, LabelType>,
      opengm::PottsFunction<ValueType, IndexType, LabelType>,
      opengm::PottsNFunction<ValueType, IndexType, LabelType>,
      opengm::PottsGFunction<ValueType, IndexType, LabelType>,
      opengm::TruncatedSquaredDifferenceFunction<ValueType, IndexType, LabelType>,
      opengm::TruncatedAbsoluteDifferenceFunction<ValueType, IndexType, LabelType> 
      >::type FunctionTypeList;


   typedef opengm::GraphicalModel<
      ValueType,
      OperatorType,
      FunctionTypeList,
      SpaceType
   > GmType;

   typedef meta::TypeListGenerator < 
      interface::POptCaller<InterfaceType, GmType, AccumulatorType>,
      opengm::meta::ListEnd
   >::type NativeInferenceTypeList;

   typedef meta::TypeListGenerator <
      opengm::meta::ListEnd
      >::type ExternalInferenceTypeList;


   typedef meta::TypeListGenerator <
      opengm::meta::ListEnd
   >::type ExternalILPInferenceTypeList;

   typedef meta::MergeTypeLists<NativeInferenceTypeList, ExternalInferenceTypeList>::type InferenceTypeList_T1;
   typedef meta::MergeTypeLists<ExternalILPInferenceTypeList, InferenceTypeList_T1>::type InferenceTypeList;
   interface::CMDInterface<GmType, InferenceTypeList> interface(argc, argv);
   interface.parse();

   return 0;
}
