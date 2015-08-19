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

#pragma once
#ifndef OPENGM_LABELCOLLAPSE_HXX
#define OPENGM_LABELCOLLAPSE_HXX

#include <algorithm>
#include <vector>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/visitors/visitors.hxx>
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

	typedef visitors::EmptyVisitor<MyType> EmptyVisitorType;
	typedef visitors::VerboseVisitor<MyType> VerboseVisitorType;
	typedef visitors::TimingVisitor<MyType> TimingVisitorType;
	typedef visitors::LabelCollapseStatisticsVisitor<MyType> StatisticsVisitorType;

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

	template<class OUTPUT_ITERATOR> void originalNumberOfLabels(OUTPUT_ITERATOR) const;
	template<class OUTPUT_ITERATOR> void currentNumberOfLabels(OUTPUT_ITERATOR) const;
	template<class IN_ITER, class OUT_ITER> void auxiliaryLabeling(IN_ITER, OUT_ITER) const;
	template<class OUTPUT_ITERATOR> void depth(OUTPUT_ITERATOR) const;
	template<class IN_ITER, class OUT_ITER> void calculateDepth(IN_ITER, OUT_ITER) const;

private:
	const GraphicalModelType *gm_;
	ReparametrizedModelType rgm;
	ReparametrizerType repa_;
	ModelBuilderType builder_;
	const Parameter parameter_;
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
	termination_ = UNKNOWN;
	labeling_.resize(gm_->numberOfVariables());
	bound_ = AccumulationType::template neutral<ValueType>();
	value_ = AccumulationType::template neutral<ValueType>();

	visitor.begin(*this);

	bool exitInf = false;
	while (!exitInf) {
		// Building the model is not really necessary (should be already done
		// at this point).
		builder_.buildAuxiliaryModel();
		const AuxiliaryModelType &gm = builder_.getAuxiliaryModel();
		OPENGM_ASSERT_OP(gm.numberOfVariables(), ==, gm_->numberOfVariables());

		// Run inference on auxiliary model and cache the results.
		typename Proxy::Inference inf(gm, parameter_.proxy);
		InferenceTermination result = inf.infer(proxy_visitor);

		// If the proxy inference method returns an error, we pass it upwards.
		if (result != NORMAL) {
			termination_ = result;
			break;
		}

		bound_ = inf.value();
		std::vector<LabelType> labeling;
		inf.arg(labeling, 1); // FIXME: Check result value.

		// If the labeling is valid, we are done.
		if (builder_.isValidLabeling(labeling.begin())) {
			exitInf = true;
			termination_ = NORMAL;
			value_ = inf.value();

			builder_.originalLabeling(labeling.begin(), labeling_.begin());
		} else {
			// Update the model. This will try to make more labels available where
			// the current labeling is invalid.
			builder_.uncollapseLabeling(labeling.begin());
		}

		// In case user code looks at the model.
		builder_.buildAuxiliaryModel();

		if (visitor(*this) != visitors::VisitorReturnFlag::ContinueInf) {
			// Visitor could also return StopInfBoundReached. But that does not
			// make any sense, because value_ is always plus or minus infinity.
			// There is no meaningful gap between value_ and bound_.
			termination_ = TIMEOUT;
			return termination_;
		}
	}

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
	std::vector<LabelType> auxiliaryLabeling;
	builder_.auxiliaryLabeling(it, auxiliaryLabeling.begin());

	// We have an auxiliary labeling, but we need a label shape (remember that
	// there is the additional zeroth label which represents all collapsed
	// labels).
	std::transform(
		auxiliaryLabeling.begin(), auxiliaryLabeling.end(),
		auxiliaryLabeling.begin(),
		std::bind2nd(std::plus<LabelType>(), 1)
	);

	populateShape(auxiliaryLabeling.begin());
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
