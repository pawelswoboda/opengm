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
template<class GM, class ACC, class DERIVED>
class ModelBuilder;

template<class GM, class ACC>
class ModelBuilderUnary;

// A (potentially partial) mapping of orignal labels to auxiliary labels
// (and vice versa) for a given variable.
template<class GM>
class Mapping;

// Reorders labels according to their unary potentials.
template<class GM, class ACC>
class Reordering;

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
	LabelCollapse(const GraphicalModelType&, const Parameter& = Parameter());
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
	labelcollapse::ModelBuilderUnary<GraphicalModelType, AccumulationType> builder_;
	const Parameter parameter_;

	InferenceTermination termination_;
	std::vector<LabelType> labeling_;
	ValueType value_;
	ValueType bound_;
};

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


// Namespace for implementation details.
namespace labelcollapse {

////////////////////////////////////////////////////////////////////////////////
//
// class ModelBuilder
//
////////////////////////////////////////////////////////////////////////////////

// Abstract base class.
template<class GM, class ACC, class DERIVED>
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

	typedef Mapping<OriginalModelType> MappingType;

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
	template<class ITERATOR> void uncollapseLabeling(ITERATOR);
	void originalLabeling(const std::vector<LabelType>&, std::vector<LabelType>&) const;
	template<class ITERATOR> void populate(ITERATOR);
	LabelType numberOfLabels(IndexType i) const { return mappings_[i].size(); }

	// The following functions should be overwritten in descendants.
	void uncollapse(const IndexType);
	void reset();

protected:
	const OriginalModelType &original_;
	AuxiliaryModelType auxiliary_;
	std::vector<ValueType> epsilons_;
	std::vector<MappingType> mappings_;
	bool rebuildNecessary_;
};

template<class GM, class ACC, class DERIVED>
ModelBuilder<GM, ACC, DERIVED>::ModelBuilder
(
	const OriginalModelType &gm
)
: original_(gm)
, epsilons_(gm.numberOfFactors(), ACC::template ineutral<ValueType>())
, mappings_(gm.numberOfFactors(), MappingType(0))
, rebuildNecessary_(true)
{
	for (IndexType i = 0; i < gm.numberOfVariables(); ++i) {
		mappings_[i] = MappingType(gm.numberOfLabels(i));
	}
}

template<class GM, class ACC, class DERIVED>
void
ModelBuilder<GM, ACC, DERIVED>::buildAuxiliaryModel()
{
	if (!rebuildNecessary_)
		return;

	// Build space.
	std::vector<LabelType> shape(original_.numberOfVariables());
	for (IndexType i = 0; i < original_.numberOfVariables(); ++i) {
		shape[i] = mappings_[i].size();
	}
	typename AuxiliaryModelType::SpaceType space(shape.begin(), shape.end());
	auxiliary_ = AuxiliaryModelType(space);

	// Build graphical models with all factors.
	for (IndexType i = 0; i < original_.numberOfFactors(); ++i) {
		typedef EpsilonFunction<OriginalModelType> ViewFunction;

		const typename OriginalModelType::FactorType &factor = original_[i];
		const ViewFunction func(factor, epsilons_[i], mappings_);

		auxiliary_.addFactor(
			auxiliary_.addFunction(func),
			factor.variableIndicesBegin(),
			factor.variableIndicesEnd()
		);
	}

	rebuildNecessary_ = false;
}

template<class GM, class ACC, class DERIVED>
template<class ITERATOR>
bool
ModelBuilder<GM, ACC, DERIVED>::isValidLabeling
(
	ITERATOR it
) const
{
	OPENGM_ASSERT(!rebuildNecessary_);

	for (IndexType i = 0; i < original_.numberOfVariables(); ++i, ++it) {
		if (mappings_[i].isCollapsedAuxiliary(*it))
			return false;
	}

	return true;
}

template<class GM, class ACC, class DERIVED>
template<class ITERATOR>
void
ModelBuilder<GM, ACC, DERIVED>::uncollapseLabeling
(
	ITERATOR it
)
{
	OPENGM_ASSERT(!rebuildNecessary_);

	for (IndexType i = 0; i < original_.numberOfVariables(); ++i, ++it) {
		if (mappings_[i].isCollapsedAuxiliary(*it))
			static_cast<DERIVED*>(this)->uncollapse(i);
	}
}

template<class GM, class ACC, class DERIVED>
template<class ITERATOR>
void
ModelBuilder<GM, ACC, DERIVED>::populate
(
	ITERATOR it
)
{
	for (IndexType i = 0; i < original_.numberOfVariables(); ++i, ++it) {
		while (mappings_[i].numberOfLabels() < std::min(*it, original_.numberOfLabels()))
			static_cast<DERIVED*>(this)->uncollapse(i);
	}
}

template<class GM, class ACC, class DERIVED>
void
ModelBuilder<GM, ACC, DERIVED>::originalLabeling
(
	const std::vector<LabelType> &auxiliary,
	std::vector<LabelType> &original
) const
{
	OPENGM_ASSERT(isValidLabeling(auxiliary.begin()));
	OPENGM_ASSERT(auxiliary.size() == original_.numberOfVariables());

	original.assign(auxiliary.size(), 0);
	for (IndexType i = 0; i < original_.numberOfVariables(); ++i) {
		original[i] = mappings_[i].original(auxiliary[i]);
	}
}


////////////////////////////////////////////////////////////////////////////////
//
// class ModelBuilderUnary
//
////////////////////////////////////////////////////////////////////////////////

template<class GM, class ACC>
class ModelBuilderUnary : public ModelBuilder<GM, ACC, ModelBuilderUnary<GM, ACC> > {
public:
	typedef ModelBuilder<GM, ACC, ModelBuilderUnary<GM, ACC> > Parent;

	using typename Parent::AccumulationType;
	using typename Parent::OperatorType;
	using typename Parent::OriginalModelType;
	using typename Parent::AuxiliaryModelType;
	using typename Parent::IndexType;
	using typename Parent::LabelType;
	using typename Parent::ValueType;

	ModelBuilderUnary(const OriginalModelType&);

	void uncollapse(const IndexType);

private:
	typedef std::vector<LabelType> Stack;

	using Parent::original_;
	using Parent::auxiliary_;
	using Parent::epsilons_;
	using Parent::mappings_;
	using Parent::rebuildNecessary_;

	std::vector<Stack> collapsed_;

	void internalChecks() const;

	friend Parent;
};

template<class GM, class ACC>
ModelBuilderUnary<GM, ACC>::ModelBuilderUnary
(
	const OriginalModelType &gm
)
: Parent(gm)
, collapsed_(gm.numberOfVariables())
{
	Reordering<OriginalModelType, AccumulationType> reordering(original_);
	for (IndexType i = 0; i < original_.numberOfVariables(); ++i) {
		collapsed_[i].resize(original_.numberOfLabels(i));

		// We reverse the ordering and use .rbegin(), because we use the
		// std::vector as a stack. The smallest element should be the last one.
		reordering.reorder(i);
		reordering.getOrdered(collapsed_[i].rbegin());
	}

	// From this point on all the invariances should hold.
	internalChecks();

	for (IndexType i = 0; i < original_.numberOfVariables(); ++i)
		uncollapse(i);

	internalChecks();
}

template<class GM, class ACC>
void
ModelBuilderUnary<GM, ACC>::uncollapse
(
	const IndexType idx
)
{
	internalChecks();

	OPENGM_ASSERT(collapsed_[idx].size() > 0);
	OPENGM_ASSERT(!mappings_[idx].full());

	LabelType label = collapsed_[idx].back();
	collapsed_[idx].pop_back();
	mappings_[idx].insert(label);

	// If there is just one collapsed label left, all the auxiliary unaries
	// and binaries are equal to the original potentials. We can just
	// uncollapse it.
	//
	// We just pop it here, because the Mapping class automatically handles
	// this case. (Checked in debug mode.)
	if (collapsed_[idx].size() == 1)
		collapsed_[idx].pop_back();

	internalChecks();

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
			// determine if we need to update the binary epsilon.
			typedef ShapeWalker< typename FactorType::ShapeIteratorType> Walker;
			Walker walker(factor.shapeBegin(), factor.numberOfVariables());
			AccumulationType::neutral(epsilons_[f]);
			for (IndexType i = 0; i < factor.size(); ++i, ++walker) {
				bool next = true;
				for (IndexType j = 0; j < factor.numberOfVariables(); ++j) {
					IndexType varIdx = factor.variableIndex(j);
					if (mappings_[varIdx].isCollapsedOriginal(walker.coordinateTuple()[j])) {
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

	rebuildNecessary_ = true;
}

template<class GM, class ACC>
void
ModelBuilderUnary<GM, ACC>::internalChecks() const
{
#ifndef NDEBUG
	for (IndexType i = 0; i < original_.numberOfVariables(); ++i) {
		for (LabelType j = 0; j < original_.numberOfLabels(i); ++j) {
			size_t count = std::count(collapsed_[i].begin(), collapsed_[i].end(), j);
			if (count != 0) {
				// is collapsed
				OPENGM_ASSERT(mappings_[i].isCollapsedOriginal(j));
			} else {
				// is not collapsed
				OPENGM_ASSERT(!mappings_[i].isCollapsedOriginal(j));
			}
		}
	}
#endif
}


////////////////////////////////////////////////////////////////////////////////
//
// class Mapping
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
class Mapping {
public:
	typedef typename GM::LabelType LabelType;

	Mapping(LabelType numLabels)
	: numLabels_(numLabels)
	, full_(false)
	, mapOrigToAux_(numLabels)
	, mapAuxToOrig_()
	{
	}

	void insert(LabelType origLabel);
	void makeFull();
	bool full() const { return full_; }
	bool isCollapsedAuxiliary(LabelType auxLabel) const;
	bool isCollapsedOriginal(LabelType origLabel) const;
	LabelType auxiliary(LabelType origLabel) const;
	LabelType original(LabelType auxLabel) const;
	LabelType size() const;

private:
	LabelType numLabels_;
	bool full_;
	std::vector<LabelType> mapOrigToAux_;
	std::vector<LabelType> mapAuxToOrig_;
};

template<class GM>
void
Mapping<GM>::insert(
	LabelType origLabel
)
{
	// If this mapping is a full bijection, then we don’t need to insert
	// anything.
	if (full_)
		return;

	// If the label is already mapped, then we don’t need to insert anything.
	if (mapOrigToAux_[origLabel] != 0)
		return;

	// In all other cases, we update our mappings. The zeroth auxiliary label
	// is reserved.
	mapAuxToOrig_.push_back(origLabel);
	mapOrigToAux_[origLabel] = mapAuxToOrig_.size();

	OPENGM_ASSERT(original(auxiliary(origLabel)) == origLabel);

	// Check whether we can convert this mapping to a full bijection.
	// This is the case if all labels are uncollapsed.
	if (mapAuxToOrig_.size() + 1 >= mapOrigToAux_.size())
		makeFull();
}

template<class GM>
bool
Mapping<GM>::isCollapsedAuxiliary
(
	LabelType auxLabel
) const
{
	return !full_ && auxLabel == 0;
}

template<class GM>
bool
Mapping<GM>::isCollapsedOriginal
(
	LabelType origLabel
) const
{
	return !full_ && auxiliary(origLabel) == 0;
}

template<class GM>
typename Mapping<GM>::LabelType
Mapping<GM>::auxiliary
(
	LabelType origLabel
) const
{
	if (full_)
		return origLabel;

	OPENGM_ASSERT(origLabel < mapOrigToAux_.size());
	return mapOrigToAux_[origLabel];
}

template<class GM>
typename Mapping<GM>::LabelType
Mapping<GM>::original
(
	LabelType auxLabel
) const
{
	if (full_)
		return auxLabel;

	OPENGM_ASSERT(auxLabel > 0);
	OPENGM_ASSERT(auxLabel - 1 < mapAuxToOrig_.size());
	return mapAuxToOrig_[auxLabel - 1];
}

template<class GM>
typename Mapping<GM>::LabelType
Mapping<GM>::size() const
{
	return full_ ? numLabels_ : mapAuxToOrig_.size() + 1;
}

template<class GM>
void
Mapping<GM>::makeFull()
{
	full_ = true;
	mapOrigToAux_.clear();
	mapAuxToOrig_.clear();
}

////////////////////////////////////////////////////////////////////////////////
//
// class Reordering
//
////////////////////////////////////////////////////////////////////////////////

template<class GM, class ACC>
class Reordering {
public:
	typedef GM GraphicalModelType;
	typedef ACC AccumulationType;
	OPENGM_GM_TYPE_TYPEDEFS;

	Reordering(const GraphicalModelType&);
	void reorder(IndexType i);
	template<class OUTPUT_ITERATOR> void getOrdered(OUTPUT_ITERATOR);
	template<class RANDOM_ACCESS_ITERATOR> void getMapping(RANDOM_ACCESS_ITERATOR);

private:
	typedef std::pair<LabelType, ValueType> Pair;
	typedef std::vector<Pair> Pairs;
	typedef typename Pairs::const_iterator Iterator;

	static bool compare(const std::pair<LabelType, ValueType>&, const std::pair<LabelType, ValueType>&);

	const GraphicalModelType &gm_;
	Pairs pairs_;
};

template<class GM, class ACC>
Reordering<GM, ACC>::Reordering
(
	const GraphicalModelType &gm
)
: gm_(gm)
{
}

template<class GM, class ACC>
void
Reordering<GM, ACC>::reorder
(
	IndexType idx
)
{
	// We look up the unary factor.
	bool found_unary = false;
	IndexType f;
	for (IndexType i = 0; i < gm_.numberOfFactors(idx); ++i) {
		f = gm_.factorOfVariable(idx, i);
		if (gm_[f].numberOfVariables() == 1) {
			found_unary = true;
			break;
		}
	}
	// TODO: Maybe we should throw exception?
	OPENGM_ASSERT(found_unary);

	pairs_.resize(gm_.numberOfLabels(idx));
	for (LabelType i = 0; i < gm_.numberOfLabels(idx); ++i) {
		opengm::FastSequence<LabelType> labeling(1);
		labeling[0] = i;
		ValueType v = gm_[f](labeling.begin());
		pairs_[i] = std::make_pair(i, v);
	}

	std::sort(pairs_.begin(), pairs_.end(), &compare);
}

template<class GM, class ACC>
template<class OUTPUT_ITERATOR>
void
Reordering<GM, ACC>::getOrdered
(
	OUTPUT_ITERATOR out
)
{
	for (Iterator in = pairs_.begin(); in != pairs_.end(); ++in, ++out)
		*out = in->first;
}

template<class GM, class ACC>
template<class RANDOM_ACCESS_ITERATOR>
void
Reordering<GM, ACC>::getMapping
(
	RANDOM_ACCESS_ITERATOR out
)
{
	LabelType l = 0;
	for (Iterator in = pairs_.begin(); in != pairs_.end(); ++in, ++l)
		out[ in->first ] = l;
}

template<class GM, class ACC>
bool
Reordering<GM, ACC>::compare
(
	const std::pair<LabelType, ValueType> &pair1,
	const std::pair<LabelType, ValueType> &pair2
)
{
	return AccumulationType::bop(pair1.second, pair2.second);
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

	typedef Mapping<GM> MappingType;

	EpsilonFunction(
		const FactorType &factor,
		ValueType epsilon,
		const std::vector<MappingType> &mappings
	)
	: factor_(&factor)
	, mappings_(&mappings)
	, epsilon_(epsilon)
	{
	}

	template<class ITERATOR> ValueType operator()(ITERATOR begin) const;
	LabelType shape(const IndexType) const;
	IndexType dimension() const;
	IndexType size() const;

private:
	const FactorType *factor_;
	const std::vector<MappingType> *mappings_;
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

		if ((*mappings_)[varIdx].isCollapsedAuxiliary(auxLabel))
			return epsilon_;

		LabelType origLabel = (*mappings_)[varIdx].original(auxLabel);
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
	return (*mappings_)[varIdx].size();
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
