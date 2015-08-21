//
// File: labelcollapse.hxx
//
// This file is part of OpenGM.
//
// Copyright (C) 2015 Stefan Haller
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//

////////////////////////////////////////////////////////////////////////////////
//
// TODO: This is not fucking true. It is not obvious were to call this
//       function. I suggest calling it anywhere where we are relying on
//       the model (e.g. population changes the model, etc.)
//
//  // Building the model is not really necessary (should be already done
//  // at this point).
//  builder_.buildAuxiliaryModel();
//
////////////////////////////////////////////////////////////////////////////////

#pragma once
#ifndef OPENGM_LABELCOLLAPSE_HXX
#define OPENGM_LABELCOLLAPSE_HXX

#include <algorithm>
#include <vector>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/visitors/visitors.hxx>
#include <opengm/inference/auxiliary/fusion_move/fusion_mover.hxx>
#include <opengm/utilities/metaprogramming.hxx>

#include "labelcollapse/modelbuilder.hxx"
#include "labelcollapse/reparametrization.hxx"
#include "labelcollapse/utils.hxx"
#include "labelcollapse/visitor.hxx"


namespace opengm {

// Main class implementing the inference method. This class is intended to be
// used by the user.
template<
	class GM, class INF,
	labelcollapse::ReparametrizationKind REPA = labelcollapse::ReparametrizationNone
> class LabelCollapse;

// This is a type generator for generating the template parameter for
// the underlying proxy inference method.
//
// Access is possible by “LabelCollapseAuxTypeGen<GM>::GraphicalModelType”.
template<class GM, class ACC>
struct LabelCollapseAuxTypeGen;

////////////////////////////////////////////////////////////////////////////////
//
// struct LabelCollapseAuxTypeGen
//
////////////////////////////////////////////////////////////////////////////////

template<class GM, class ACC>
struct LabelCollapseAuxTypeGen {
	typedef typename LPReparametrizer<GM, ACC>::ReparametrizedGMType ReparametrizedModelType;
	typedef typename labelcollapse::ModelBuilderAuxTypeGen<ReparametrizedModelType, ACC>::GraphicalModelType GraphicalModelType;
};

////////////////////////////////////////////////////////////////////////////////
//
// class LabelCollapse
//
////////////////////////////////////////////////////////////////////////////////

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
class LabelCollapse : public opengm::Inference<GM, typename INF::AccumulationType>
{
public:
	//
	// Types
	//
	struct Proxy {
		// This namespace contains all the types of the underlying inference
		// method. Many are identical with our types, but for sake of
		// completeness all are mentioned here.
		typedef INF Inference;
		typedef typename INF::AccumulationType AccumulationType;
		typedef typename INF::GraphicalModelType GraphicalModelType;
		OPENGM_GM_TYPE_TYPEDEFS;

		typedef typename INF::EmptyVisitorType EmptyVisitorType;
		typedef typename INF::VerboseVisitorType VerboseVisitorType;
		typedef typename INF::TimingVisitorType TimingVisitorType;
		typedef typename std::vector<LabelType>::const_iterator LabelIterator;
		typedef typename INF::Parameter Parameter;
	};
	typedef LabelCollapse<GM, INF, KIND> MyType;

	typedef typename INF::AccumulationType AccumulationType;
	typedef GM GraphicalModelType;
	typedef LabelCollapseAuxTypeGen<GraphicalModelType, AccumulationType> AuxTypeGen;
	typedef typename AuxTypeGen::ReparametrizedModelType ReparametrizedModelType;
	typedef typename AuxTypeGen::GraphicalModelType AuxiliaryModelType;
	typedef typename labelcollapse::ReparametrizerTypeGen<GraphicalModelType, AccumulationType, KIND>::Type ReparametrizerType;
	typedef typename labelcollapse::ModelBuilder<ReparametrizedModelType, AccumulationType> ModelBuilderType;

	OPENGM_GM_TYPE_TYPEDEFS;

#if 0
	typedef visitors::EmptyVisitor<MyType> EmptyVisitorType;
	typedef visitors::VerboseVisitor<MyType> VerboseVisitorType;
	typedef visitors::TimingVisitor<MyType> TimingVisitorType;
	typedef visitors::LabelCollapseStatisticsVisitor<MyType> StatisticsVisitorType;
#else
	typedef visitors::LabelCollapseStatisticsVisitor<MyType> EmptyVisitorType;
#endif

	typedef typename std::vector<LabelType>::const_iterator LabelIterator;

	struct Parameter {
		Parameter()
		: proxy()
		{}

		Parameter(const typename Proxy::Parameter &proxy)
		: proxy(proxy)
		{}

		typename Proxy::Parameter proxy;
	};


	//
	// Methods
	//
	LabelCollapse(const GraphicalModelType&, const Parameter& = Parameter());
	std::string name() const;
	const GraphicalModelType& graphicalModel() const { return *gm_; }
	const ReparametrizedModelType &reparametrizedModel() const { return repa_.reparametrizedModel(); }
	const AuxiliaryModelType& currentAuxiliaryModel() const { return builder_.getAuxiliaryModel(); }
	const ReparametrizerType& reparametrizer() const { return repa_; }

	InferenceTermination infer();
	template<class VISITOR> InferenceTermination infer(VISITOR&);
	template<class VISITOR, class PROXY_VISITOR> InferenceTermination infer(VISITOR&, PROXY_VISITOR&);
	InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
	ValueType bound() const { return bound_; }
	virtual ValueType value() const { return value_; }

	template<class INPUT_ITERATOR> void populateShape(INPUT_ITERATOR);
	template<class INPUT_ITERATOR> void populateLabeling(INPUT_ITERATOR);
	template<class INPUT_ITERATOR> void populateFusionMove(INPUT_ITERATOR);

	template<class OUTPUT_ITERATOR> void originalNumberOfLabels(OUTPUT_ITERATOR) const;
	template<class OUTPUT_ITERATOR> void currentNumberOfLabels(OUTPUT_ITERATOR) const;
	template<class IN_ITER, class OUT_ITER> void auxiliaryLabeling(IN_ITER, OUT_ITER) const;
	template<class OUTPUT_ITERATOR> void depth(OUTPUT_ITERATOR) const;
	template<class IN_ITER, class OUT_ITER> void calculateDepth(IN_ITER, OUT_ITER) const;

private:
	const GraphicalModelType *gm_;
	ReparametrizedModelType rgm;
	ReparametrizerType repa_;
	mutable ModelBuilderType builder_; // This is a hack. Sry.
	Parameter parameter_;
	InferenceTermination termination_;
	std::vector<LabelType> labeling_;
	ValueType value_;
	ValueType bound_;
};

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
LabelCollapse<GM, INF, KIND>::LabelCollapse
(
	const GraphicalModelType &gm,
	const Parameter &parameter
)
: gm_(&gm)
, repa_(gm)
, builder_(ReparametrizedModelType())
, parameter_(parameter)
{
	// FIXME: This is a bit clumsy. We first construct all the objects and
	// afterwards need to instantiate a *new* builder. This should be fixed.
	repa_.reparametrize();
	repa_.getReparametrizedModel(rgm);
	builder_ = ModelBuilderType(rgm);

	// Maybe user code wants to have a look at the model.
	builder_.buildAuxiliaryModel();
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
std::string
LabelCollapse<GM, INF, KIND>::name() const
{
	AuxiliaryModelType gm;
	typename Proxy::Inference inf(gm, parameter_.proxy);
	return "LabelCollapse(" + inf.name() + ")";
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
InferenceTermination
LabelCollapse<GM, INF, KIND>::infer()
{
	EmptyVisitorType visitor;
	return infer(visitor);
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
template<class VISITOR>
InferenceTermination
LabelCollapse<GM, INF, KIND>::infer
(
	VISITOR &visitor
)
{
	typename Proxy::EmptyVisitorType proxy_visitor;
	return infer(visitor, proxy_visitor);
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
template<class VISITOR, class PROXY_VISITOR>
InferenceTermination
LabelCollapse<GM, INF, KIND>::infer
(
	VISITOR& visitor,
	PROXY_VISITOR& proxy_visitor
)
{
	size_t cntLP = 0, cntILP = 0;

	termination_ = UNKNOWN;
	labeling_.resize(gm_->numberOfVariables());
	bound_ = AccumulationType::template neutral<ValueType>();
	value_ = AccumulationType::template neutral<ValueType>();

	visitor.begin(*this);
	std::vector<LabelType> labeling;

	bool again = true;
	while (again) {
		// Building the model is not really necessary (should be already done
		// at this point).
		builder_.buildAuxiliaryModel();
		const AuxiliaryModelType &gm = builder_.getAuxiliaryModel();
		OPENGM_ASSERT_OP(gm.numberOfVariables(), ==, gm_->numberOfVariables());
		std::vector<LabelType> labeling;

		// Run approximate inference.
		{
			std::cout << "-> Running TRWS... ";
			typedef TRWSi<AuxiliaryModelType, AccumulationType> InfType;
			typename InfType::Parameter param;
			param.setTreeAgreeMaxStableIter(100);
			param.maxNumberOfIterations_= 500;
			InfType inf(gm, param);
			termination_ = inf.infer();

			if (termination_ != NORMAL && termination_ != CONVERGENCE)
				break;

			inf.arg(labeling, 1);

			++cntLP;
		}

		// If there were no auxiliary labels selected during the approximate
		// inference, we check again with our “real” inference method.
		// (Otherwise we just uncollapse the labels and retry.)
		if (builder_.isValidLabeling(labeling.begin())){
			std::cout << "Seems good, run ILP..." << std::endl;

			// FIXME: Serious hack.
			parameter_.proxy.mipStartLabeling_ = labeling;

			typename Proxy::Inference inf(gm, parameter_.proxy);
			termination_ = inf.infer(proxy_visitor);

			if (termination_ != NORMAL && termination_ != CONVERGENCE)
				break;

			bound_ = inf.value();
			inf.arg(labeling, 1);

			++cntILP;
		} else {
			std::cout << "Improvement possible" << std::endl;
		}

		// Update the model. This will try to make more labels available where
		// the current labeling is invalid.
		// If no update was necessary (i.e. the labeling is valid), we are done.
		again = builder_.uncollapseLabeling(labeling.begin());

		if (! again) {
			termination_ = NORMAL;
			value_ = bound_;
			builder_.originalLabeling(labeling.begin(), labeling_.begin());
		} else {
			// In case user code looks at the model.
			builder_.buildAuxiliaryModel();
		}

		if (visitor(*this) != visitors::VisitorReturnFlag::ContinueInf) {
			// Visitor could also return StopInfBoundReached. But that does not
			// make any sense, because value_ is always plus or minus infinity.
			// There is no meaningful gap between value_ and bound_.
			termination_ = TIMEOUT;
			return termination_;
		}
	}

	std::cout << "-> cntLP = " << cntLP << " | cntILP = " << cntILP << std::endl;

	visitor.end(*this);
	return termination_;
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
template<class ITERATOR>
void
LabelCollapse<GM, INF, KIND>::populateShape
(
	ITERATOR it
)
{
	for (IndexType i = 0; i < gm_->numberOfVariables(); ++i, ++it)
		while (builder_.isUncollapsable(i) && builder_.numberOfLabels(i) < *it)
			builder_.uncollapse(i);
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
template<class ITERATOR>
void
LabelCollapse<GM, INF, KIND>::populateLabeling
(
	ITERATOR it
)
{
	std::vector<LabelType> auxiliaryLabeling(gm_->numberOfVariables());

	do {
		builder_.auxiliaryLabeling(it, auxiliaryLabeling.begin());
	} while (builder_.uncollapseLabeling(auxiliaryLabeling.begin()));
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
template<class ITERATOR>
void
LabelCollapse<GM, INF, KIND>::populateFusionMove
(
	ITERATOR it
)
{
	size_t cntMove = 0;

	// We need to populate the original labeling so that it is available in
	// the model.
	populateLabeling(it);

	// This is our reference labeling model.
	std::vector<LabelType> auxLabeling(gm_->numberOfVariables());
	builder_.auxiliaryLabeling(it, auxLabeling.begin());

	bool again = true;
	while (again) {
		++cntMove;
		std::cout << "(FusionMove)" << std::endl;
		builder_.buildAuxiliaryModel();
		const AuxiliaryModelType &model = builder_.getAuxiliaryModel();

		ValueType resultingEnergy;
		std::vector<LabelType> resultingLabeling(gm_->numberOfVariables());
		std::vector<LabelType> lowerBoundLabeling(gm_->numberOfVariables());

		// If we can’t move any label to a auxiliary label (function returns
		// false), both labelings are identical. It does not make any sense to
		// run FusionMove. (FusionMove would simply return false.)
		if (builder_.moveToAuxiliary(auxLabeling.begin(), lowerBoundLabeling.begin())) {
			// Default parameter of HlFusionMover uses QPBO if available and in
			// all other cases LazyFlipper.
			typedef HlFusionMover<AuxiliaryModelType, AccumulationType> FusionMover;
			typename FusionMover::Parameter param;
			FusionMover fusionMover(model, param);

			// TODO: Why do we need to pass the values of the labelings separately?
			bool success = fusionMover.fuse(
				auxLabeling, lowerBoundLabeling, resultingLabeling,
				model.evaluate(auxLabeling.begin()), model.evaluate(lowerBoundLabeling.begin()),
				resultingEnergy
			);

			// Only false if both labelings are identical. Can’t be the case,
			// see moveToAuxiliary above.
			OPENGM_ASSERT(success);

			again = builder_.uncollapseLabeling(resultingLabeling.begin());
		} else {
			again = false;
		}
	}

	std::cout << "-> cntMove = " << cntMove << std::endl;
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
InferenceTermination
LabelCollapse<GM, INF, KIND>::arg
(
	std::vector<LabelType>& label,
	const size_t idx
) const
{
	if (idx == 1) {
		label = labeling_;
		return termination_;
	} else {
		return UNKNOWN;
	}
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
template<class OUTPUT_ITERATOR>
void
LabelCollapse<GM, INF, KIND>::originalNumberOfLabels
(
	OUTPUT_ITERATOR it
) const
{
	builder_.buildAuxiliaryModel();
	for (IndexType i = 0; i < gm_->numberOfVariables(); ++i, ++it) {
		*it = gm_->numberOfLabels(i);
	}
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
template<class OUTPUT_ITERATOR>
void
LabelCollapse<GM, INF, KIND>::currentNumberOfLabels
(
	OUTPUT_ITERATOR it
) const
{
	builder_.buildAuxiliaryModel();
	for (IndexType i = 0; i < gm_->numberOfVariables(); ++i, ++it) {
		*it = builder_.numberOfLabels(i);
	}
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
template<class IN_ITER, class OUT_ITER>
void
LabelCollapse<GM, INF, KIND>::auxiliaryLabeling
(
	IN_ITER original,
	OUT_ITER auxiliary
) const
{
	builder_.buildAuxiliaryModel();
	builder_.auxiliaryLabeling(original, auxiliary);
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
template<class OUTPUT_ITERATOR>
void
LabelCollapse<GM, INF, KIND>::depth
(
	OUTPUT_ITERATOR depth
) const
{
	OPENGM_ASSERT(termination_ == CONVERGENCE || termination_ == NORMAL);
	calculateDepth(labeling_.begin(), depth);
}

template<class GM, class INF, labelcollapse::ReparametrizationKind KIND>
template<class IN_ITER, class OUT_ITER>
void
LabelCollapse<GM, INF, KIND>::calculateDepth
(
	IN_ITER labeling,
	OUT_ITER depth
) const
{
	builder_.buildAuxiliaryModel();
	builder_.calculateDepth(labeling, depth);
}

} // namespace opengm

#endif
