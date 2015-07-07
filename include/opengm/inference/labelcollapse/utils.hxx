//
// File: utils.hxx
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
#ifndef OPENGM_LABELCOLLAPSE_UTILS_HXX
#define OPENGM_LABELCOLLAPSE_UTILS_HXX

#include <algorithm>
#include <vector>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/datastructures/fast_sequence.hxx>

namespace opengm {
namespace labelcollapse {

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
template<class GM, class ACC>
class EpsilonFunction;

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

	const GraphicalModelType *gm_;
	Pairs pairs_;
};

template<class GM, class ACC>
Reordering<GM, ACC>::Reordering
(
	const GraphicalModelType &gm
)
: gm_(&gm)
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
	for (IndexType i = 0; i < gm_->numberOfFactors(idx); ++i) {
		f = gm_->factorOfVariable(idx, i);
		if ((*gm_)[f].numberOfVariables() == 1) {
			found_unary = true;
			break;
		}
	}
	// TODO: Maybe we should throw exception?
	OPENGM_ASSERT(found_unary);

	pairs_.resize(gm_->numberOfLabels(idx));
	for (LabelType i = 0; i < gm_->numberOfLabels(idx); ++i) {
		opengm::FastSequence<LabelType> labeling(1);
		labeling[0] = i;
		ValueType v = (*gm_)[f](labeling.begin());
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

template<class GM, class ACC>
class EpsilonFunction
: public FunctionBase<EpsilonFunction<GM, ACC>,
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
		const std::vector<MappingType> &mappings
	)
	: factor_(&factor)
	, mappings_(&mappings)
	{
	}

	template<class ITERATOR> ValueType operator()(ITERATOR begin) const;
	LabelType shape(const IndexType) const;
	IndexType dimension() const;
	IndexType size() const;

private:
	const FactorType *factor_;
	const std::vector<MappingType> *mappings_;
};

template<class GM, class ACC>
template<class ITERATOR>
typename EpsilonFunction<GM, ACC>::ValueType
EpsilonFunction<GM, ACC>::operator()
(
	ITERATOR it
) const
{
	// At first we check which labels of current the factor are collapsed.
	std::vector<bool> isCollapsed(factor_->numberOfVariables());
	std::vector<LabelType> modified(factor_->numberOfVariables());
	for (IndexType i = 0; i < factor_->numberOfVariables(); ++i, ++it) {
		IndexType varIdx = factor_->variableIndex(i);
		LabelType auxLabel = *it;

		isCollapsed[i] = (*mappings_)[varIdx].isCollapsedAuxiliary(auxLabel);

		if (! isCollapsed[i]) {
			LabelType origLabel = (*mappings_)[varIdx].original(auxLabel);
			modified[i] = origLabel;
		}
	}

	// If no label is collapsed, we car just return the normal potential of
	// the wrapped factor.
	if (std::count(isCollapsed.begin(), isCollapsed.end(), true) == 0)
		return (*factor_)(modified.begin());

	// Otherwise we fix all the non-collapsed labels and calculate the best
	// local transition in this factor.
	FastSequence<IndexType> fixedVars;
	FastSequence<ValueType> fixedLbls;
	for (IndexType i = 0; i < factor_->numberOfVariables(); ++i) {
		if (! isCollapsed[i]) {
			fixedVars.push_back(i);
			fixedLbls.push_back(modified[i]);
		}
	}

	// TODO: For higher order factor this is not very efficient. Currently we
	// are iterating over all the factor transistions even if we discard many
	// of them.
	//
	// FIXME:
	//   - The following code is not written with performance in mind.
	//   - The following code is ugly.
	//   - Long story short: The following code sucks.
	typedef SubShapeWalker<typename FactorType::ShapeIteratorType, FastSequence<IndexType>, FastSequence<ValueType> > Walker;
	Walker walker(factor_->shapeBegin(), factor_->numberOfVariables(), fixedVars, fixedLbls);
	ValueType result = ACC::template neutral<ValueType>();
	for (size_t z = 0; z < walker.subSize(); ++z, ++walker) {
		bool next = false;
		for (IndexType i = 0; i < factor_->numberOfVariables(); ++i) {
			if (std::find(fixedVars.begin(), fixedVars.end(), i) != fixedVars.end())
				continue;

			IndexType varIdx = factor_->variableIndex(i);
			if (! (*mappings_)[varIdx].isCollapsedOriginal(walker.coordinateTuple()[i])) {
				next = true;
				break;
			}
		}

		if (next)
			continue;

		ValueType current = (*factor_)(walker.coordinateTuple().begin());
		if (ACC::bop(current, result))
			result = current;
	}
	return result;
}

template<class GM, class ACC>
typename EpsilonFunction<GM, ACC>::LabelType
EpsilonFunction<GM, ACC>::shape
(
	const IndexType idx
) const
{
	IndexType varIdx = factor_->variableIndex(idx);
	return (*mappings_)[varIdx].size();
}

template<class GM, class ACC>
typename EpsilonFunction<GM, ACC>::IndexType
EpsilonFunction<GM, ACC>::dimension() const
{
	return factor_->numberOfVariables();
}

template<class GM, class ACC>
typename EpsilonFunction<GM, ACC>::IndexType
EpsilonFunction<GM, ACC>::size() const
{
	IndexType result = 1;
	for (IndexType i = 0; i < factor_->numberOfVariables(); ++i) {
		result *= shape(i);
	}
	return result;
}

} // namepasce labelcollapse
} // namepasce opengm

#endif
