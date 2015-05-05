//
// File: labelcollapse_internal.hxx
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
#ifndef OPENGM_LABELCOLLAPSE_INTERNAL_HXX
#define OPENGM_LABELCOLLAPSE_INTERNAL_HXX

#include <vector>
#include <utility>

#include "opengm/datastructures/fast_sequence.hxx"
#include "opengm/functions/function_properties_base.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/space/discretespace.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/utilities/indexing.hxx"
#include "opengm/utilities/metaprogramming.hxx"

#include "labelcollapse_declarations.hxx"

namespace opengm {
namespace labelcollapse {

////////////////////////////////////////////////////////////////////////////////
//
// forward declarations
//
////////////////////////////////////////////////////////////////////////////////

// Builds the auxiliary model given the original model.
template<class GM, class INF>
class ModelBuilder;

// Reorders labels according to their unary potentials.
template<class GM, class ACC>
class Reordering;

// A view function which returns the values from the original model if the
// nodes are not collapsed. If they are, the view function will return the
// corresponding epsilon value.
template<class GM>
class EpsilonFunction;

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
	LabelType numberOfLabels(IndexType) const;
	template<class ITERATOR> void uncollapseLabeling(ITERATOR);
	void uncollapse(const IndexType);
	template<class ITERATOR> void populate(ITERATOR);

	void reset();

private:
	typedef std::vector<LabelType> Stack;
	typedef std::vector<bool> Mapping;

	bool isFull(IndexType) const;

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
			for (unsigned int j = 0; j < 3; ++j)
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
			// determine if we need to update the binary epsilon.
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
template<class ITERATOR>
void
ModelBuilder<GM, ACC>::populate
(
	ITERATOR it
)
{
	for (IndexType i = 0; i < original_.numberOfVariables(); ++i, ++it) {
		while (numberOfLabels(i) < *it && collapsed_[i].size() > 0) {
			uncollapse(i);
		}
	}
}

template<class GM, class ACC>
void
ModelBuilder<GM, ACC>::reset()
{
	epsilons_.assign(original_.numberOfFactors(), ACC::template ineutral<ValueType>());
	rebuildNecessary_ = true;

	Reordering<OriginalModelType, AccumulationType> reordering(original_);
	for (IndexType i = 0; i < original_.numberOfVariables(); ++i) {
		mappings_[i].assign(original_.numberOfLabels(i), false);
		collapsed_[i].resize(original_.numberOfLabels(i));
		uncollapsed_[i].clear();

		// We reverse the ordering and use .rbegin(), because we use the
		// std::vector as a stack. The smallest element should be the last one.
		reordering.reorder(i);
		reordering.getOrdered(collapsed_[i].rbegin());

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
