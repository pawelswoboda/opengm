/*
 * combiLP.hxx
 *
 *  Created on: Sep 16, 2013
 *		  Author: bsavchyn
 */

#ifndef OPENGM_COMBILP_HXX
#define OPENGM_COMBILP_HXX

// To enable detailed debug output enable the following preprocessor macro:
// #define OPENGM_COMBILP_DEBUG

#include <boost/scoped_ptr.hpp>

#include <opengm/graphicalmodel/graphicalmodel_manipulator.hxx>
#include <opengm/inference/lpcplex.hxx>
#include <opengm/inference/auxiliary/lp_reparametrization.hxx>
#include <opengm/inference/trws/output_debug_utils.hxx>
#include <opengm/inference/trws/trws_base.hxx>

namespace opengm{

namespace combilp {
	template<class GM>
	void DilateMask(const GM&, typename GM::IndexType, std::vector<bool>&);

	template<class GM>
	void DilateMask(const GM&,const std::vector<bool>&, std::vector<bool>&);

	template<class GM>
	bool LabelingMatching(const std::vector<typename GM::LabelType>&,const std::vector<typename GM::LabelType>&, const std::vector<bool>&, std::list<typename GM::IndexType>&);

	template<class GM>
	bool LabelingMatching(const GM&, const std::vector<typename GM::LabelType>&, const std::vector<typename GM::LabelType>&, const std::vector<bool>&, std::list<typename GM::IndexType>&, typename GM::ValueType&);

	template<class GM>
	void GetMaskBoundary(const GM&, const std::vector<bool>&, std::vector<bool>&);

	template<class LP, class REPA>
	class Parameter;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// \brief CombiLP\n\n
/// Savchynskyy, B. and Kappes, J. H. and Swoboda, P. and Schnoerr, C.:
/// "Global MAP-Optimality by Shrinking the Combinatorial Search Area with Convex Relaxation".
/// In NIPS, 2013.
/// \ingroup inference
template<class GM, class ACC, class LPSOLVER>//TODO: remove default ILP solver
class CombiLP : public Inference<GM, ACC>
{
public:
	//
	// Types
	//
	typedef typename LPSOLVER::ReparametrizerType ReparametrizerType;

	typedef ACC AccumulationType;
	typedef GM GraphicalModelType;
	OPENGM_GM_TYPE_TYPEDEFS;

	typedef visitors::VerboseVisitor<CombiLP<GM, ACC, LPSOLVER> > VerboseVisitorType;
	typedef visitors::EmptyVisitor<CombiLP<GM, ACC, LPSOLVER> >   EmptyVisitorType;
	typedef visitors::TimingVisitor<CombiLP<GM, ACC, LPSOLVER> >  TimingVisitorType;

	typedef combilp::Parameter<typename LPSOLVER::Parameter,typename ReparametrizerType::Parameter> Parameter;
	typedef typename ReparametrizerType::MaskType MaskType;
	typedef typename opengm::GraphicalModelManipulator<typename ReparametrizerType::ReparametrizedGMType> GMManipulatorType;

	typedef LPCplex<typename GMManipulatorType::MGM, ACC> LPCPLEX;//TODO: move to template parameters

	//
	// Methods
	//
	CombiLP(const GraphicalModelType& gm, const Parameter& param);
	std::string name() const{ return "CombiLP"; }
	const GraphicalModelType& graphicalModel() const { return lpsolver_.graphicalModel(); }

	InferenceTermination infer();
	template<class VISITOR> InferenceTermination infer(VISITOR &visitor);
	template<class VISITORWRAPPER> InferenceTermination infer(MaskType& mask,const std::vector<LabelType>& lp_labeling,VISITORWRAPPER& vis,ValueType value, ValueType bound);
	InferenceTermination arg(std::vector<LabelType>& out, const size_t = 1) const;
	ValueType bound() const{return bound_;};
	ValueType value() const{return value_;};

private:
	//
	// Methods
	//
	void ReparametrizeAndSave();
	void Reparametrize_(typename ReparametrizerType::ReparametrizedGMType& pgm,const MaskType& mask);
	InferenceTermination PerformILPInference_(GMManipulatorType& modelManipulator,std::vector<LabelType>& plabeling);

	//
	// Members
	//
	Parameter parameter_;
	LPSOLVER lpsolver_;
	boost::scoped_ptr<ReparametrizerType> plpparametrizer_;
	std::vector<LabelType> labeling_;
	ValueType value_;
	ValueType bound_;
};

template<class GM, class ACC, class LPSOLVER>
CombiLP<GM, ACC, LPSOLVER>::CombiLP
(
	const GraphicalModelType& gm,
	const Parameter& param
)
: parameter_(param)
, lpsolver_(gm,param.lpsolverParameter_)
, plpparametrizer_(lpsolver_.getReparametrizer(parameter_.repaParameter_))//TODO: parameters of the reparametrizer come here
, labeling_(gm.numberOfVariables(),std::numeric_limits<LabelType>::max())
, value_(lpsolver_.value())
, bound_(lpsolver_.bound())
// FROM BASE CLASS:
#if 0
, value_(ACC::template neutral<ValueType>())
, bound_(ACC::template ineutral<ValueType>())
#endif
{
#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "Parameters of the " << name() << " algorithm:" << std::endl;
	param.print();
#endif
};

template<class GM, class ACC, class LPSOLVER>
InferenceTermination
CombiLP<GM, ACC, LPSOLVER>::infer()
{
	EmptyVisitorType visitor;
	return infer(visitor);
};

template<class GM, class ACC, class LPSOLVER>
template<class VISITOR>
InferenceTermination
CombiLP<GM, ACC, LPSOLVER>::infer
(
	VISITOR &visitor
)
{
#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "Running LP solver "<< lpsolver_.name() << std::endl;
#endif
	visitor.begin(*this);

	lpsolver_.infer();
	value_=lpsolver_.value();
	bound_=lpsolver_.bound();
	lpsolver_.arg(labeling_);

	if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
		visitor.end(*this);
		return NORMAL;
	}

	std::vector<LabelType> labeling_lp;
	MaskType initialmask;
	plpparametrizer_->reparametrize();
	//plpparametrizer_->getArcConsistency(&initialmask,&labeling_lp);
	lpsolver_.getTreeAgreement(initialmask,&labeling_lp);

#ifdef OPENGM_COMBILP_DEBUG
	std::cout << "Energy of the labeling consistent with the arc consistency =" << lpsolver_.graphicalModel().evaluate(labeling_lp) << std::endl;
	std::cout << "Arc inconsistent set size =" << std::count(initialmask.begin(),initialmask.end(),false) << std::endl;
	std::cout << "Trivializing." << std::endl;
#endif

#ifdef  WITH_HDF5
	if (parameter_.reparametrizedModelFileName_.compare("")!=0)
	{
#ifdef OPENGM_COMBILP_DEBUG
		std::cout << "Saving reparametrized model..." << std::endl;
#endif
		if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
			visitor.end(*this);
			return NORMAL;
		}
		ReparametrizeAndSave();
		if( visitor(*this) != visitors::VisitorReturnFlag::ContinueInf ){
			visitor.end(*this);
			return NORMAL;
		}
	}
#endif

	if (std::count(initialmask.begin(),initialmask.end(),false)==0)
		return NORMAL;

	trws_base::transform_inplace(initialmask.begin(),initialmask.end(),std::logical_not<bool>());

	MaskType mask;
	if (parameter_.singleReparametrization_) //BSD: do not need to dilate it in the new approach
		combilp::DilateMask(lpsolver_.graphicalModel(),initialmask,mask);
	else mask=initialmask;

	visitors::VisitorWrapper<VISITOR,CombiLP<GM,ACC,LPSOLVER> > vis(&visitor,this);
	InferenceTermination terminationVal = infer(mask,labeling_lp,vis,value(),bound());
	if ( (terminationVal==NORMAL) || (terminationVal==CONVERGENCE) )
	{
		// FIXME: This looks fishy.
		value_= value();
		bound_= bound();
		arg(labeling_);
	}

	visitor.end(*this);
	return NORMAL;
}

template<class GM, class ACC, class LPSOLVER>
InferenceTermination
CombiLP<GM, ACC, LPSOLVER>::arg(
	std::vector<LabelType>& labeling,
	const size_t idx
) const
{
	if (idx != 1)
		return UNKNOWN;

	labeling = labeling_;
	return NORMAL;
}

template<class GM, class ACC, class LPSOLVER>
InferenceTermination
CombiLP<GM, ACC, LPSOLVER>::PerformILPInference_
(
	GMManipulatorType& modelManipulator,
	std::vector<LabelType>& plabeling
)
{
	InferenceTermination terminationILP=NORMAL;
	modelManipulator.buildModifiedSubModels();

	// FIXME: Introduce typedef for labeling
	std::vector< std::vector<LabelType> > submodelLabelings(modelManipulator.numberOfSubmodels());
	for (size_t modelIndex=0; modelIndex < modelManipulator.numberOfSubmodels(); ++modelIndex) {
		const typename GMManipulatorType::MGM& model = modelManipulator.getModifiedSubModel(modelIndex);
		submodelLabelings[modelIndex].resize(model.numberOfVariables());
		typename LPCPLEX::Parameter param;
		param.integerConstraint_=true;
		param.numberOfThreads_= parameter_.threads_;

		// FIXME: Make this parameters
		// Even better! Introduce proxy parameters!!
		param.timeLimit_ = 3600;
		param.workMem_= 1024*6;
		LPCPLEX ilpSolver(model,param);
		terminationILP=ilpSolver.infer();

		if ((terminationILP!=NORMAL) && (terminationILP!=CONVERGENCE)) {
			return terminationILP;
		} else {
			ilpSolver.arg(submodelLabelings[modelIndex]);
		}
	}

	modelManipulator.modifiedSubStates2OriginalState(submodelLabelings, plabeling);

	return terminationILP;
}

template<class GM, class ACC, class LPSOLVER>
template <class VISITORWRAPPER>
InferenceTermination
CombiLP<GM, ACC, LPSOLVER>::infer(
	MaskType& mask,
	const std::vector<LabelType>& lp_labeling,
	VISITORWRAPPER& vis,
	ValueType value_,
	ValueType bound_
)
{
	value_=value_;
	bound_=bound_;

#ifdef OPENGM_COMBILP_DEBUG
	if (!parameter_.singleReparametrization_)
		std::cout << "Applying reparametrization for each ILP run ..." << std::endl;
	else
		std::cout << "Applying a single uniform reparametrization..." << std::endl;
	std::cout << "Switching to ILP." << std::endl;
#endif

	bool startILP=true;
	typename ReparametrizerType::ReparametrizedGMType gm;
	bool reparametrizedFlag=false;
	InferenceTermination terminationId=TIMEOUT;

	for (size_t i=0; (startILP && (i < parameter_.maxNumberOfILPCycles_)); ++i) {
		if(vis() != visitors::VisitorReturnFlag::ContinueInf)
			return TIMEOUT;

#ifdef OPENGM_COMBILP_DEBUG
		std::cout << "Subproblem " << i << " size=" << std::count(mask.begin(), mask.end(), true) << std::endl;
#endif

		MaskType boundmask(mask.size());
		combilp::GetMaskBoundary(plpparametrizer_->graphicalModel(),mask, boundmask);

#ifdef OPENGM_COMBILP_DEBUG
		if (parameter_.saveProblemMasks_) {
			OUT::saveContainer(std::string(parameter_.maskFileNamePre_+"-mask-"+trws_base::any2string(i)+".txt"),mask.begin(),mask.end());
			OUT::saveContainer(std::string(parameter_.maskFileNamePre_+"-boundmask-"+trws_base::any2string(i)+".txt"),boundmask.begin(),boundmask.end());
		}
#endif

		if (parameter_.singleReparametrization_ && (!reparametrizedFlag)) {
#ifdef OPENGM_COMBILP_DEBUG
			std::cout << "Reparametrizing..." << std::endl;
#endif

			Reparametrize_(gm,MaskType(mask.size(), true));
			reparametrizedFlag=true;
		} else if (!parameter_.singleReparametrization_) {
#ifdef OPENGM_COMBILP_DEBUG
			std::cout << "Reparametrizing..." << std::endl;
#endif
			Reparametrize_(gm,mask);
		}

		OPENGM_ASSERT_OP(mask.size(), ==, gm.numberOfVariables());

		GMManipulatorType modelManipulator(gm,GMManipulatorType::DROP);
		modelManipulator.unlock();
		modelManipulator.freeAllVariables();

		for (IndexType varId=0;varId<mask.size();++varId)
			if (mask[varId]==0)
				modelManipulator.fixVariable(varId,lp_labeling[varId]);

		modelManipulator.lock();

		InferenceTermination terminationILP;
		std::vector<LabelType> labeling;
		terminationILP=PerformILPInference_(modelManipulator, labeling);
		if ((terminationILP != NORMAL) && (terminationILP != CONVERGENCE)) {
#ifdef OPENGM_COMBILP_DEBUG
			std::cout << "ILP solver failed to solve the problem. Best attained results will be saved." << std::endl;
#endif
			// TODO: BSD: check that in this case the resulting labeling is the
			// best one attained and not obligatory lp_labeling
			if (parameter_.singleReparametrization_)
				labeling_=lp_labeling;

			return terminationILP;
		}

#ifdef OPENGM_COMBILP_DEBUG
		std::cout << "Boundary size=" << std::count(boundmask.begin(),boundmask.end(),true) << std::endl;
#endif

		std::list<IndexType> result;
		bool optimalityFlag;

		ValueType gap=0;
		if (parameter_.singleReparametrization_) {
			optimalityFlag=combilp::LabelingMatching<GM>(lp_labeling, labeling, boundmask, result);
		} else {
			optimalityFlag=combilp::LabelingMatching(gm, lp_labeling, labeling, mask, result, gap);
			ValueType newvalue=gm.evaluate(labeling);

			std::vector<bool> imask(mask.size());
			std::transform(mask.begin(),mask.end(),imask.begin(),std::logical_not<bool>());
			ValueType newbound=gm.evaluate(labeling,mask)+gm.evaluate(labeling,imask);

			if (ACC::bop(newvalue,value_)) {
				value_=newvalue;
				labeling_=labeling;
			}

			ACC::iop(bound_,newbound,bound_);

#ifdef OPENGM_COMBILP_DEBUG
			std::cout << "newvalue=" << newvalue << "; best value=" << value_ << std::endl;
			std::cout << "newbound=" << newbound << "; best bound=" << bound_ << std::endl;
			std::cout << "new gap=" << gap << std::endl;
#endif
		}

		if (optimalityFlag || (fabs(value_-bound_)<= std::numeric_limits<ValueType>::epsilon()*value_)) {
			startILP=false;
			labeling_=labeling;
			value_=bound_=plpparametrizer_->graphicalModel().evaluate(labeling_);
			terminationId=NORMAL;
#ifdef OPENGM_COMBILP_DEBUG
			std::cout << "Solved! Optimal energy=" << value() << std::endl;
#endif
		} else {
#ifdef OPENGM_COMBILP_DEBUG
			std::cout << "Adding " << result.size() << " nodes." << std::endl;
			if (parameter_.saveProblemMasks_)
			OUT::saveContainer(std::string(parameter_.maskFileNamePre_+"-added-"+trws_base::any2string(i)+".txt"),result.begin(),result.end());
#endif
			for (typename std::list<IndexType>::const_iterator it=result.begin();it!=result.end();++it) {
				if (parameter_.singleReparametrization_) //BSD: expanding the mask
					combilp::DilateMask(gm, *it, mask);
				else
					mask[*it]=true;
			}
		}
	}

	return terminationId;
}

template<class GM, class ACC, class LPSOLVER>
void
CombiLP<GM, ACC, LPSOLVER>::Reparametrize_(
	typename ReparametrizerType::ReparametrizedGMType& pgm,
	const MaskType& mask
)
{
	plpparametrizer_->reparametrize(&mask);
	plpparametrizer_->getReparametrizedModel(pgm);
}

template<class GM, class ACC, class LPSOLVER>
void
CombiLP<GM, ACC, LPSOLVER>::ReparametrizeAndSave()
{
	typename ReparametrizerType::ReparametrizedGMType gm;
	Reparametrize_(gm,MaskType(plpparametrizer_->graphicalModel().numberOfVariables(),true));
	store_into_explicit(gm, parameter_.reparametrizedModelFileName_);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace combilp {

	template<class LPSOLVERPARAMETERS, class REPARAMETRIZERPARAMETERS>
	struct Parameter
	{
		Parameter
		(
			LPSOLVERPARAMETERS lpsolverParameter = LPSOLVERPARAMETERS(),
			REPARAMETRIZERPARAMETERS repaParameter = REPARAMETRIZERPARAMETERS(),
			size_t maxNumberOfILPCycles = 100,
			bool verbose = false,
			std::string reparametrizedModelFileName = "",
			bool singleReparametrization = true,
			bool saveProblemMasks = false,
			std::string maskFileNamePre = "",
			size_t threads = 1
		)
		: maxNumberOfILPCycles_(maxNumberOfILPCycles)
		, verbose_(verbose)
		, reparametrizedModelFileName_(reparametrizedModelFileName)
		, singleReparametrization_(singleReparametrization)
		, saveProblemMasks_(saveProblemMasks)
		, maskFileNamePre_(maskFileNamePre)
		, threads_(threads)
		{
		}

		size_t maxNumberOfILPCycles_;
		bool verbose_;
		std::string reparametrizedModelFileName_;
		bool singleReparametrization_;
		bool saveProblemMasks_;
		std::string maskFileNamePre_;
		size_t threads_;

		LPSOLVERPARAMETERS lpsolverParameter_;
		REPARAMETRIZERPARAMETERS repaParameter_;

#ifdef OPENGM_COMBILP_DEBUG
		void
		print() const
		{
			std::cout << "maxNumberOfILPCycles=" << maxNumberOfILPCycles_ << std::endl;
			std::cout << "verbose" << verbose_ << std::endl;
			std::cout << "reparametrizedModelFileName=" << reparametrizedModelFileName_ << std::endl;
			std::cout << "singleReparametrization=" << singleReparametrization_ << std::endl;
			std::cout << "saveProblemMasks=" << saveProblemMasks_ << std::endl;
			std::cout << "maskFileNamePre=" << maskFileNamePre_ << std::endl;
			std::cout << "== lpsolverParameters: ==" << std::endl;
			lpsolverParameter_.print(std::cout);
		}
#endif
	};
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

	namespace combilp{

		template<class FACTOR>
		void MakeFactorVariablesTrue(const FACTOR& f,std::vector<bool>& pmask)
		{
			for (typename FACTOR::VariablesIteratorType it=f.variableIndicesBegin();
				  it!=f.variableIndicesEnd();++it)
				pmask[*it]=true;
		}

		template<class GM>
		void DilateMask(const GM& gm,typename GM::IndexType varId,std::vector<bool>& pmask)
		{
			typename GM::IndexType numberOfFactors=gm.numberOfFactors(varId);
			for (typename GM::IndexType localFactorId=0;localFactorId<numberOfFactors;++localFactorId)
			{
				const typename GM::FactorType& f=gm[gm.factorOfVariable(varId,localFactorId)];
				if (f.numberOfVariables()>1)
					MakeFactorVariablesTrue(f, pmask);
			}
		}

/*
 * inmask and poutmask should be different objects!
 */
		template<class GM>
		void DilateMask(const GM& gm,const std::vector<bool>& inmask, std::vector<bool>& poutmask)
		{
			poutmask=inmask;
			for (typename GM::IndexType varId=0;varId<inmask.size();++varId)
				if (inmask[varId]) DilateMask(gm,varId,poutmask);

		}

		template<class GM>
		bool LabelingMatching(const std::vector<typename GM::LabelType>& labeling1,const std::vector<typename GM::LabelType>& labeling2,
									 const std::vector<bool>& mask,std::list<typename GM::IndexType>& presult)
		{
			OPENGM_ASSERT(labeling1.size()==mask.size());
			OPENGM_ASSERT(labeling2.size()==mask.size());
			presult.clear();
			for (typename GM::IndexType varId=0;varId<mask.size();++varId)
				if ((mask[varId]) && (labeling1[varId]!=labeling2[varId]))
					presult.push_back(varId);

			return presult.empty();
		}


		template<class GM>
		bool LabelingMatching(const GM& gm, const std::vector<typename GM::LabelType>& labeling_out,const std::vector<typename GM::LabelType>& labeling_in,
									 const std::vector<bool>& mask_in,std::list<typename GM::IndexType>& presult, typename GM::ValueType& pgap)
		{
			OPENGM_ASSERT(labeling_in.size()==mask_in.size());
			OPENGM_ASSERT(labeling_out.size()==mask_in.size());
			presult.clear();

			//go over all border p/w potentials and check that the corresponding edge 0

		  std::vector<std::pair<typename GM::IndexType,typename GM::IndexType> > borderFactors;
		  std::vector<typename GM::IndexType> borderFactorCounter(gm.numberOfVariables(),0);//!< is not needed below, just to fit function parameters list
		  LPReparametrizer<GM,Minimizer>::getGMMaskBorder(gm,mask_in,&borderFactors,&borderFactorCounter);//!< Minimizer does not play any role in this code, just to instantiate the template

		  pgap=0;
		  std::vector<typename GM::LabelType> ind(2,0);
		  for (typename std::vector<std::pair<typename GM::IndexType,typename GM::IndexType> >::const_iterator fit=borderFactors.begin();
								fit!=borderFactors.end();++fit)
		  {
					 typename GM::IndexType var_out=gm[fit->first].variableIndex(fit->second);
					 typename GM::IndexType var_in=gm[fit->first].variableIndex(1-fit->second);

					 ind[fit->second]=labeling_out[var_out];
					 ind[1-fit->second]=labeling_in[var_in];

					 if (fabs(gm[fit->first](ind.begin())) > 1e-15)//BSD: improve this line to get an optimal edge and be independent on numerical issues
					 {
								pgap += gm[fit->first](ind.begin());
								presult.push_back(var_out);
					 }
		  }

			return presult.empty();
		}

		template<class GM>
		void GetMaskBoundary(const GM& gm,const std::vector<bool>& mask,std::vector<bool>& boundmask)
		{
			boundmask.assign(mask.size(),false);
			for (typename GM::IndexType varId=0;varId<mask.size();++varId)
			{
				if (!mask[varId]) continue;

				typename GM::IndexType numberOfFactors=gm.numberOfFactors(varId);
				for (typename GM::IndexType localFactorId=0;localFactorId<numberOfFactors;++localFactorId)
				{
					if (boundmask[varId]) break;

					const typename GM::FactorType& f=gm[gm.factorOfVariable(varId,localFactorId)];
					if (f.numberOfVariables()>1)
					{
						for (typename GM::FactorType::VariablesIteratorType it=f.variableIndicesBegin();
							  it!=f.variableIndicesEnd();++it)
							if (!mask[*it])
							{
								boundmask[varId]=true;
								break;
							}
					}
				}
			}
		}
	}
}

#endif
