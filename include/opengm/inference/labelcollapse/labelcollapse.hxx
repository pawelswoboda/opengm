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

#include <vector>

#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"

#include "labelcollapse_internal.hxx"
#include "labelcollapse_visitor.hxx"

namespace opengm {

////////////////////////////////////////////////////////////////////////////////
//
// struct LabelCollapseAuxTypeGen
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
struct LabelCollapseAuxTypeGen {
	typedef typename GM::OperatorType OperatorType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename GM::ValueType ValueType;

	typedef typename opengm::DiscreteSpace<IndexType, LabelType> SpaceType;
	typedef typename meta::TypeListGenerator< labelcollapse::EpsilonFunction<GM> >::type FunctionTypeList;

	typedef GraphicalModel<ValueType, OperatorType, FunctionTypeList, SpaceType>
	GraphicalModelType;
};

////////////////////////////////////////////////////////////////////////////////
//
// class LabelCollapse
//
////////////////////////////////////////////////////////////////////////////////

template<class GM, class INF>
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

	typedef typename INF::AccumulationType AccumulationType;
	typedef GM GraphicalModelType;
	typedef typename LabelCollapseAuxTypeGen<GM>::GraphicalModelType
	AuxiliaryModelType;

	OPENGM_GM_TYPE_TYPEDEFS;

	typedef visitors::EmptyVisitor< LabelCollapse<GM, INF> > EmptyVisitorType;
	typedef visitors::VerboseVisitor< LabelCollapse<GM, INF> > VerboseVisitorType;
	typedef visitors::TimingVisitor< LabelCollapse<GM, INF> > TimingVisitorType;
	typedef visitors::LabelCollapseStatisticsVisitor< LabelCollapse<GM, INF> > StatisticsVisitorType;
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
	LabelCollapse(const GraphicalModelType&);
	LabelCollapse(const GraphicalModelType&, const Parameter&);
	std::string name() const;
	const GraphicalModelType& graphicalModel() const { return gm_; }
	const AuxiliaryModelType& currentAuxiliaryModel() const { return builder_.getAuxiliaryModel(); }

	InferenceTermination infer();
	template<class VISITOR> InferenceTermination infer(VISITOR&);
	template<class VISITOR, class PROXY_VISITOR> InferenceTermination infer(VISITOR&, PROXY_VISITOR&);
	InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
	ValueType bound() const { return bound_; }
	virtual ValueType value() const { return value_; }
	template<class INPUT_ITERATOR> void populate(INPUT_ITERATOR);
	void reset();

	template<class OUTPUT_ITERATOR> void originalNumberOfLabels(OUTPUT_ITERATOR) const;
	template<class OUTPUT_ITERATOR> void currentNumberOfLabels(OUTPUT_ITERATOR) const;

private:
	const GraphicalModelType &gm_;
	labelcollapse::ModelBuilder<GraphicalModelType, AccumulationType> builder_;
	Parameter parameter_;

	InferenceTermination termination_;
	std::vector<LabelType> labeling_;
	ValueType value_;
	ValueType bound_;
};

template<class GM, class INF>
LabelCollapse<GM, INF>::LabelCollapse
(
	const GraphicalModelType &gm
)
: gm_(gm)
, builder_(gm)
{
}

template<class GM, class INF>
LabelCollapse<GM, INF>::LabelCollapse
(
	const GraphicalModelType &gm,
	const Parameter &parameter
)
: gm_(gm)
, builder_(gm)
, parameter_(parameter)
{
}

template<class GM, class INF>
std::string
LabelCollapse<GM, INF>::name() const
{
	AuxiliaryModelType gm;
	typename Proxy::Inference inf(gm, parameter_.proxy);
	return "LabelCollapse(" + inf.name() + ")";
}

template<class GM, class INF>
InferenceTermination
LabelCollapse<GM, INF>::infer()
{
	EmptyVisitorType visitor;
	return infer(visitor);
}

template<class GM, class INF>
template<class VISITOR>
InferenceTermination
LabelCollapse<GM, INF>::infer
(
	VISITOR &visitor
)
{
	typename Proxy::EmptyVisitorType proxy_visitor;
	return infer(visitor, proxy_visitor);
}

template<class GM, class INF>
template<class VISITOR, class PROXY_VISITOR>
InferenceTermination
LabelCollapse<GM, INF>::infer
(
	VISITOR& visitor,
	PROXY_VISITOR& proxy_visitor
)
{
	std::vector<LabelType> warmstarting;
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

		// FIXME: Serious hack.
		parameter_.proxy.mipStartLabeling_ = warmstarting;

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
		warmstarting = labeling;

		// If the labeling is valid, we are done.
		if (builder_.isValidLabeling(labeling.begin())) {
			exitInf = true;
			termination_ = NORMAL;
			value_ = inf.value();
			builder_.originalLabeling(labeling, labeling_);
		}

		if (visitor(*this) != visitors::VisitorReturnFlag::ContinueInf)
			exitInf = true;

		// Update the model. This will try to make more labels available where
		// the current labeling is invalid.
		builder_.uncollapseLabeling(labeling.begin());
	}

	visitor.end(*this);
	return termination_;
}

template<class GM, class INF>
void
LabelCollapse<GM, INF>::reset()
{
	builder_.reset();
}

template<class GM, class INF>
template<class INPUT_ITERATOR>
void
LabelCollapse<GM, INF>::populate(
	INPUT_ITERATOR it
)
{
	builder_.populate(it);
}

template<class GM, class INF>
InferenceTermination
LabelCollapse<GM, INF>::arg
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

template<class GM, class INF>
template<class OUTPUT_ITERATOR>
void
LabelCollapse<GM, INF>::originalNumberOfLabels
(
	OUTPUT_ITERATOR it
) const
{
	for (IndexType i = 0; i < gm_.numberOfVariables(); ++i, ++it) {
		*it = gm_.numberOfLabels(i);
	}
}

template<class GM, class INF>
template<class OUTPUT_ITERATOR>
void
LabelCollapse<GM, INF>::currentNumberOfLabels
(
	OUTPUT_ITERATOR it
) const
{
	for (IndexType i = 0; i < gm_.numberOfVariables(); ++i, ++it) {
		*it = builder_.numberOfLabels(i);
	}
}

} // namespace opengm

#endif
