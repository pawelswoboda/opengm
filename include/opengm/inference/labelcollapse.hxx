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
#include <utility>

#include "opengm/datastructures/fast_sequence.hxx"
#include "opengm/functions/function_properties_base.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/space/discretespace.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/labelcollapse_visitor.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/utilities/indexing.hxx"
#include "opengm/utilities/metaprogramming.hxx"

namespace opengm {

////////////////////////////////////////////////////////////////////////////////
//
// Forward declarations and a little typeclassopedia.
//
////////////////////////////////////////////////////////////////////////////////

// Main class implementing the inference method. This class is intended to be
// used by the user.
template<class GM, class INF>
class LabelCollapse;

// This is a type generator for generating the template parameter for
// the underlying proxy inference method.
//
// Access is possible by “LabelCollapseAuxTypeGen<GM>::GraphicalModelType”.
template<class GM>
struct LabelCollapseAuxTypeGen;

//
// Namespace for internal implementation details.
//
namespace labelcollapse {

// Builds the auxiliary model given the original model.
template<class GM, class INF>
class ModelBuilder;

// A view function which returns the values from the original model if the
// nodes are not collapsed. If they are, the view function will return the
// corresponding epsilon value.
template<class GM>
class EpsilonFunction;

} // namespace labelcollapse

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
	void reset();

	template<class OUTPUT_ITERATOR> void originalNumberOfLabels(OUTPUT_ITERATOR) const;
	template<class OUTPUT_ITERATOR> void currentNumberOfLabels(OUTPUT_ITERATOR) const;

private:
	const GraphicalModelType &gm_;
	labelcollapse::ModelBuilder<GraphicalModelType, AccumulationType> builder_;
	const Parameter parameter_;

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
	if (builder_.rebuildNecessary()) {
		// If the builder needs to rebuild the model, we did not perform any
		// inference iteration (“iterations == 0”). We just return the labels
		// of the original model.
		//
		// FIXME: Ensure that we are at iteration zero! The current
		// implementation of this aspect looks more like a hack.
		return originalNumberOfLabels(it);
	}

	const AuxiliaryModelType &gm = builder_.getAuxiliaryModel();
	for (IndexType i = 0; i < gm.numberOfVariables(); ++i, ++it) {
		*it = gm.numberOfLabels(i);
	}
}


// Namespace for implementation details.
namespace labelcollapse {

////////////////////////////////////////////////////////////////////////////////
//
// class ModelBuilder
//
////////////////////////////////////////////////////////////////////////////////

template<class GM, class ACC>
class ModelBuilder {
public:
	//
	// Types
	//
	typedef ACC AccumulationType;

	// Shared types (Original = Modified)
	typedef typename GM::OperatorType OperatorType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename GM::ValueType ValueType;

	// Original model types
	typedef GM OriginalModelType;

	// Auxiliary model types
	typedef typename LabelCollapseAuxTypeGen<GM>::GraphicalModelType
	AuxiliaryModelType;

	//
	// Methods
	//
	ModelBuilder(const OriginalModelType&);

	void buildAuxiliaryModel();
	bool rebuildNecessary() const { return rebuildNecessary_; }

	const OriginalModelType& getOriginalModel() const { return original_; }
	const AuxiliaryModelType& getAuxiliaryModel() const
	{
		OPENGM_ASSERT(!rebuildNecessary_);
		return auxiliary_;
	}

	template<class ITERATOR> bool isValidLabeling(ITERATOR) const;
	void originalLabeling(const std::vector<LabelType>&, std::vector<LabelType>&) const;
	template<class ITERATOR> void uncollapseLabeling(ITERATOR);
	void uncollapse(const IndexType);

	void reset();

private:
	typedef std::vector<LabelType> Stack;
	typedef std::vector<bool> Mapping;

	bool isFull(IndexType) const;
	LabelType numberOfLabels(IndexType) const;
	static bool compare(const std::pair<LabelType, ValueType>&, const std::pair<LabelType, ValueType>&);

	const OriginalModelType &original_;
	std::vector<Stack> uncollapsed_;
	std::vector<Stack> collapsed_;
	std::vector<Mapping> mappings_;
	std::vector<ValueType> epsilons_;
	bool rebuildNecessary_;
	AuxiliaryModelType auxiliary_;
};

template<class GM, class ACC>
ModelBuilder<GM, ACC>::ModelBuilder
(
	const OriginalModelType &gm
)
: original_(gm)
, uncollapsed_(gm.numberOfVariables())
, collapsed_(gm.numberOfVariables())
, mappings_(gm.numberOfVariables())
, rebuildNecessary_(true)
{
	reset();
}

template<class GM, class ACC>
void
ModelBuilder<GM, ACC>::buildAuxiliaryModel()
{
	if (!rebuildNecessary_)
		return;

	// Build space.
	std::vector<LabelType> shape(original_.numberOfVariables());
	for (IndexType i = 0; i < original_.numberOfVariables(); ++i) {
		shape[i] = numberOfLabels(i);
	}
	typename AuxiliaryModelType::SpaceType space(shape.begin(), shape.end());
	auxiliary_ = AuxiliaryModelType(space);

	// Build graphical models with all factors.
	for (IndexType i = 0; i < original_.numberOfFactors(); ++i) {
		typedef EpsilonFunction<OriginalModelType> ViewFunction;

		const typename OriginalModelType::FactorType &factor = original_[i];
		const ViewFunction func(factor, uncollapsed_, collapsed_, epsilons_[i]);

		auxiliary_.addFactor(
			auxiliary_.addFunction(func),
			factor.variableIndicesBegin(),
			factor.variableIndicesEnd()
		);
	}

	rebuildNecessary_ = false;
}

template<class GM, class ACC>
void
ModelBuilder<GM, ACC>::originalLabeling
(
	const std::vector<LabelType> &auxiliary,
	std::vector<LabelType> &original
) const
{
	OPENGM_ASSERT(isValidLabeling(auxiliary.begin()));
	OPENGM_ASSERT(auxiliary.size() == original_.numberOfVariables());

	original.assign(auxiliary.size(), 0);
	for (IndexType i = 0; i < original_.numberOfVariables(); ++i) {
		original[i] = isFull(i) ? auxiliary[i] : uncollapsed_[i][auxiliary[i] - 1];
	}
}

template<class GM, class ACC>
template<class ITERATOR>
bool
ModelBuilder<GM, ACC>::isValidLabeling
(
	ITERATOR it
) const
{
	OPENGM_ASSERT(!rebuildNecessary_);

	for (IndexType i = 0; i < original_.numberOfVariables(); ++i, ++it) {
		if (!isFull(i) && *it == 0)
			return false;
	}

	return true;
}

template<class GM, class ACC>
template<class ITERATOR>
void
ModelBuilder<GM, ACC>::uncollapseLabeling
(
	ITERATOR it
)
{
	OPENGM_ASSERT(!rebuildNecessary_);

	for (IndexType i = 0; i < original_.numberOfVariables(); ++i, ++it) {
		if (!isFull(i) && *it == 0)
			uncollapse(i);
	}
}

template<class GM, class ACC>
void
ModelBuilder<GM, ACC>::uncollapse
(
	const IndexType idx
)
{
	if (collapsed_[idx].size() == 0)
		return;

	LabelType label = collapsed_[idx].back();
	collapsed_[idx].pop_back();
	uncollapsed_[idx].push_back(label);
	mappings_[idx][label] = true;

	for (IndexType f2 = 0; f2 < original_.numberOfFactors(idx); ++f2) {
		typedef typename OriginalModelType::FactorType FactorType;
		const IndexType f = original_.factorOfVariable(idx, f2);
		const FactorType factor = original_[f];

		if (factor.numberOfVariables() == 1) {
			// If there are no collapsed labels then the unary epsilon
			// value will not be used. So we only update it if there are
			// more uncollapsed labels.
			if (collapsed_[idx].size() > 0) {
				opengm::FastSequence<LabelType> labeling(1);
				labeling[0] = collapsed_[idx].back();
				epsilons_[f] = factor(labeling.begin());
			}
		} else if (factor.numberOfVariables() > 1) {
			// Use ShapeWalker to iterate over all the factor transitions and
			// determine if we need to update the binary epsilon. We keep our
			// uncollapsed label fixed (new value can only appear for the
			// uncollapsed label).
			//
			// FIXME: We should use OpenGM’s smart container for stack
			// allocation.
			IndexType fixedIndex = 0;
			for (IndexType i = 0; i < factor.numberOfVariables(); ++i) {
				if (factor.variableIndex(i) == idx) {
					fixedIndex = i;
					break;
				}
			}
			opengm::FastSequence<IndexType> fixedVariables(1);
			fixedVariables[0] = fixedIndex;
			opengm::FastSequence<LabelType> fixedLabels(1);
			fixedLabels[0] = label;

			typedef ShapeWalker< typename FactorType::ShapeIteratorType> Walker;
			Walker walker(factor.shapeBegin(), factor.numberOfVariables());
			AccumulationType::neutral(epsilons_[f]);
			for (IndexType i = 0; i < factor.size(); ++i, ++walker) {
				bool next = true;
				for (IndexType j = 0; j < factor.numberOfVariables(); ++j) {
					IndexType varIdx = factor.variableIndex(j);
					if (!mappings_[varIdx][walker.coordinateTuple()[j]]) {
						next = false;
						break;
					}
				}

				if (next)
					continue;

				ValueType v = factor(walker.coordinateTuple().begin());
				if (AccumulationType::bop(v, epsilons_[f])) {
					epsilons_[f] = v;
				}
			}
		}
	}

	// If there is just one collapsed label left, all the auxiliary unaries
	// and binaries are equal to the original potentials. We can just
	// uncollapse it.
	if (collapsed_[idx].size() == 1) {
		uncollapse(idx);
	}

	rebuildNecessary_ = true;
}

template<class GM, class ACC>
void
ModelBuilder<GM, ACC>::reset()
{
	epsilons_.assign(original_.numberOfFactors(), ACC::template ineutral<ValueType>());
	rebuildNecessary_ = true;

	for (IndexType i = 0; i < original_.numberOfVariables(); ++i) {
		// We look up the unary factor.
		bool found = false;
		IndexType f;
		for (IndexType j = 0; j < original_.numberOfFactors(i); ++j) {
			f = original_.factorOfVariable(i, j);
			if (original_[f].numberOfVariables() == 1) {
				found = true;
				break;
			}
		}
		// TODO: Maybe we should throw exception?
		OPENGM_ASSERT(found);

		mappings_[i].assign(original_.numberOfLabels(i), false);
		collapsed_[i].clear();
		uncollapsed_[i].clear();

		std::vector< std::pair<LabelType, ValueType> > pairs(original_.numberOfLabels(i));
		for (LabelType j = 0; j < original_.numberOfLabels(i); ++j) {
			opengm::FastSequence<LabelType> labeling(1);
			labeling[0] = j;
			ValueType v = original_[f](labeling.begin());
			pairs[j] = std::make_pair(j, v);
		}

		std::sort(pairs.begin(), pairs.end(), &compare);

		for (LabelType j = 0; j < original_.numberOfLabels(i); ++j) {
			collapsed_[i].push_back(pairs[j].first);
		}

		uncollapse(i);
	}
}

template<class GM, class ACC>
bool
ModelBuilder<GM, ACC>::isFull
(
	IndexType idx
) const
{
	return collapsed_[idx].size() == 0;
}

template<class GM, class ACC>
typename ModelBuilder<GM, ACC>::LabelType
ModelBuilder<GM, ACC>::numberOfLabels
(
	IndexType idx
) const
{
	return uncollapsed_[idx].size() + (isFull(idx) ? 0 : 1);
}

template<class GM, class ACC>
bool
ModelBuilder<GM, ACC>::compare
(
	const std::pair<LabelType, ValueType> &pair1,
	const std::pair<LabelType, ValueType> &pair2
)
{
	// We reverse the ordering, because we use a std::vector as a stack. The
	// smallest element should be the last one.
	return AccumulationType::bop(pair2.second, pair1.second);
}

////////////////////////////////////////////////////////////////////////////////
//
// class EpsilonFunction
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
class EpsilonFunction
: public FunctionBase<EpsilonFunction<GM>,
                      typename GM::ValueType, typename GM::IndexType, typename GM::LabelType>
{
public:
	typedef typename GM::FactorType FactorType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename GM::ValueType ValueType;

	EpsilonFunction(
		const FactorType &factor,
		const std::vector< std::vector<LabelType> > &uncollapsed,
		const std::vector< std::vector<LabelType> > &collapsed,
		ValueType epsilon
	)
	: factor_(&factor)
	, uncollapsed_(&uncollapsed)
	, collapsed_(&collapsed)
	, epsilon_(epsilon)
	{
	}

	template<class ITERATOR> ValueType operator()(ITERATOR begin) const;
	LabelType shape(const IndexType) const;
	IndexType dimension() const;
	IndexType size() const;

private:
	const FactorType *factor_;
	const std::vector< std::vector<LabelType> > *uncollapsed_, *collapsed_;
	ValueType epsilon_;
};

template<class GM>
template<class ITERATOR>
typename EpsilonFunction<GM>::ValueType
EpsilonFunction<GM>::operator()
(
	ITERATOR it
) const
{
	std::vector<LabelType> modified(factor_->numberOfVariables());
	for (IndexType i = 0; i < factor_->numberOfVariables(); ++i, ++it) {
		IndexType varIdx = factor_->variableIndex(i);
		LabelType auxLabel = *it;
		LabelType origLabel;

		if ((*collapsed_)[varIdx].size() == 0) {
			origLabel = auxLabel;
		} else {
			if (auxLabel == 0) {
				return epsilon_;
			}

			origLabel = (*uncollapsed_)[varIdx][auxLabel - 1];
		}

		modified[i] = origLabel;
	}

	return factor_->operator()(modified.begin());
}

template<class GM>
typename EpsilonFunction<GM>::LabelType
EpsilonFunction<GM>::shape
(
	const IndexType idx
) const
{
	IndexType varIdx = factor_->variableIndex(idx);
	return (*uncollapsed_)[varIdx].size() + ((*collapsed_)[varIdx].size() > 0 ? 1 : 0);
}

template<class GM>
typename EpsilonFunction<GM>::IndexType
EpsilonFunction<GM>::dimension() const
{
	return factor_->numberOfVariables();
}

template<class GM>
typename EpsilonFunction<GM>::IndexType
EpsilonFunction<GM>::size() const
{
	IndexType result = 1;
	for (IndexType i = 0; i < factor_->numberOfVariables(); ++i) {
		result *= shape(i);
	}
	return result;
}

} // namespace labelcollapse
} // namespace opengm

#endif
