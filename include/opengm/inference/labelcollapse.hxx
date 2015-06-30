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
#include "labelcollapse/reparameterization.hxx"
#include "labelcollapse/utils.hxx"
#include "labelcollapse/visitor.hxx"


namespace opengm {

// Main class implementing the inference method. This class is intended to be
// used by the user.
template<
	class GM, class INF,
	labelcollapse::UncollapsingBehavior UNCOLLAPSING = labelcollapse::Unary,
	labelcollapse::ReparameterizationKind REPA = labelcollapse::ReparameterizationNone
> class LabelCollapse;

// This is a type generator for generating the template parameter for
// the underlying proxy inference method.
//
// Access is possible by “LabelCollapseAuxTypeGen<GM>::GraphicalModelType”.
template<class GM>
struct LabelCollapseAuxTypeGen;

////////////////////////////////////////////////////////////////////////////////
//
// struct LabelCollapseAuxTypeGen
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
struct LabelCollapseAuxTypeGen {
	// HACK: We are only interested in the the type and so we pass an arbitrary
	// ACC (AccumulationType).
	typedef typename labelcollapse::Reparameterizer<GM, opengm::Minimizer>::ReparameterizedModelType ReparameterizedModelType;
	typedef typename labelcollapse::ModelBuilderAuxTypeGen<ReparameterizedModelType>::GraphicalModelType GraphicalModelType;
};

////////////////////////////////////////////////////////////////////////////////
//
// class LabelCollapse
//
////////////////////////////////////////////////////////////////////////////////

template<class GM, class INF, labelcollapse::UncollapsingBehavior UNCOLLAPSING, labelcollapse::ReparameterizationKind KIND>
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
	typedef LabelCollapse<GM, INF, UNCOLLAPSING, KIND> MyType;

	typedef typename INF::AccumulationType AccumulationType;
	typedef GM GraphicalModelType;
	typedef typename LabelCollapseAuxTypeGen<GM>::ReparameterizedModelType ReparameterizedModelType;
	typedef typename LabelCollapseAuxTypeGen<GM>::GraphicalModelType AuxiliaryModelType;
	typedef typename labelcollapse::Reparameterizer<GraphicalModelType, AccumulationType, KIND> ReparameterizerType;
	typedef typename labelcollapse::ModelBuilderTypeGen<ReparameterizedModelType, AccumulationType, UNCOLLAPSING>::Type ModelBuilderType;

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
	const ReparameterizedModelType &reparameterizedModel() const { return repa_.reparameterizedModel(); }
	const AuxiliaryModelType& currentAuxiliaryModel() const { return builder_.getAuxiliaryModel(); }

	InferenceTermination infer();
	template<class VISITOR> InferenceTermination infer(VISITOR&);
	template<class VISITOR, class PROXY_VISITOR> InferenceTermination infer(VISITOR&, PROXY_VISITOR&);
	InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
	ValueType bound() const { return bound_; }
	virtual ValueType value() const { return value_; }
	template<class INPUT_ITERATOR> void populate(INPUT_ITERATOR);

	template<class OUTPUT_ITERATOR> void originalNumberOfLabels(OUTPUT_ITERATOR) const;
	template<class OUTPUT_ITERATOR> void currentNumberOfLabels(OUTPUT_ITERATOR) const;
	template<class INOUT_ITERATOR> void calculateDepth(INOUT_ITERATOR) const;

private:
	const GraphicalModelType *gm_;
	ReparameterizerType repa_;
	ModelBuilderType builder_;
	const Parameter parameter_;

	InferenceTermination termination_;
	std::vector<LabelType> labeling_;
	ValueType value_;
	ValueType bound_;
};

template<class GM, class INF, labelcollapse::UncollapsingBehavior UNCOLLAPSING, labelcollapse::ReparameterizationKind KIND>
LabelCollapse<GM, INF, UNCOLLAPSING, KIND>::LabelCollapse
(
	const GraphicalModelType &gm,
	const Parameter &parameter
)
: gm_(&gm)
, repa_(gm)
, builder_(repa_.reparameterizedModel())
, parameter_(parameter)
{
	// FIXME: This is a bit clumsy. We first construct all the objects and
	// afterwards need to instantiate a new builder. This should be fixed.
	repa_.reparameterize();
	builder_ = ModelBuilderType(repa_.reparameterizedModel());
}

template<class GM, class INF, labelcollapse::UncollapsingBehavior UNCOLLAPSING, labelcollapse::ReparameterizationKind KIND>
std::string
LabelCollapse<GM, INF, UNCOLLAPSING, KIND>::name() const
{
	AuxiliaryModelType gm;
	typename Proxy::Inference inf(gm, parameter_.proxy);
	return "LabelCollapse(" + inf.name() + ")";
}

template<class GM, class INF, labelcollapse::UncollapsingBehavior UNCOLLAPSING, labelcollapse::ReparameterizationKind KIND>
InferenceTermination
LabelCollapse<GM, INF, UNCOLLAPSING, KIND>::infer()
{
	EmptyVisitorType visitor;
	return infer(visitor);
}

template<class GM, class INF, labelcollapse::UncollapsingBehavior UNCOLLAPSING, labelcollapse::ReparameterizationKind KIND>
template<class VISITOR>
InferenceTermination
LabelCollapse<GM, INF, UNCOLLAPSING, KIND>::infer
(
	VISITOR &visitor
)
{
	typename Proxy::EmptyVisitorType proxy_visitor;
	return infer(visitor, proxy_visitor);
}

template<class GM, class INF, labelcollapse::UncollapsingBehavior UNCOLLAPSING, labelcollapse::ReparameterizationKind KIND>
template<class VISITOR, class PROXY_VISITOR>
InferenceTermination
LabelCollapse<GM, INF, UNCOLLAPSING, KIND>::infer
(
	VISITOR& visitor,
	PROXY_VISITOR& proxy_visitor
)
{
	termination_ = UNKNOWN;
	labeling_.resize(0);
	bound_ = AccumulationType::template neutral<ValueType>();
	value_ = AccumulationType::template neutral<ValueType>();

	visitor.begin(*this);

	bool exitInf = false;
	while (!exitInf) {
		// Build auxiliary model.
		builder_.buildAuxiliaryModel();
		const AuxiliaryModelType &gm = builder_.getAuxiliaryModel();

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

			builder_.originalLabeling(labeling, labeling_);
		} else {
			// Update the model. This will try to make more labels available where
			// the current labeling is invalid.
			builder_.uncollapseLabeling(labeling.begin());
		}

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

template<class GM, class INF, labelcollapse::UncollapsingBehavior UNCOLLAPSING, labelcollapse::ReparameterizationKind KIND>
template<class INPUT_ITERATOR>
void
LabelCollapse<GM, INF, UNCOLLAPSING, KIND>::populate
(
	INPUT_ITERATOR it
)
{
	builder_.populate(it);
}

template<class GM, class INF, labelcollapse::UncollapsingBehavior UNCOLLAPSING, labelcollapse::ReparameterizationKind KIND>
InferenceTermination
LabelCollapse<GM, INF, UNCOLLAPSING, KIND>::arg
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

template<class GM, class INF, labelcollapse::UncollapsingBehavior UNCOLLAPSING, labelcollapse::ReparameterizationKind KIND>
template<class OUTPUT_ITERATOR>
void
LabelCollapse<GM, INF, UNCOLLAPSING, KIND>::originalNumberOfLabels
(
	OUTPUT_ITERATOR it
) const
{
	for (IndexType i = 0; i < gm_->numberOfVariables(); ++i, ++it) {
		*it = gm_->numberOfLabels(i);
	}
}

template<class GM, class INF, labelcollapse::UncollapsingBehavior UNCOLLAPSING, labelcollapse::ReparameterizationKind KIND>
template<class OUTPUT_ITERATOR>
void
LabelCollapse<GM, INF, UNCOLLAPSING, KIND>::currentNumberOfLabels
(
	OUTPUT_ITERATOR it
) const
{
	for (IndexType i = 0; i < gm_->numberOfVariables(); ++i, ++it) {
		*it = builder_.numberOfLabels(i);
	}
}

template<class GM, class INF, labelcollapse::UncollapsingBehavior UNCOLLAPSING, labelcollapse::ReparameterizationKind KIND>
template<class INOUT_ITERATOR>
void
LabelCollapse<GM, INF, UNCOLLAPSING, KIND>::calculateDepth
(
	INOUT_ITERATOR it
) const
{
	OPENGM_ASSERT(termination_ == CONVERGENCE || termination_ == NORMAL);
	builder_.calculateDepth(it);
}

} // namespace opengm

#endif
