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
template<class GM>
class EpsilonFunction;

// This functor operates on a factor. It unwraps the underlying factor function
// and calls the FUNCTOR on this function.
//
// We need this intermediate step to get C++ template inference working.
template<class FUNCTOR>
class UnwrapFunctionFunctor;

// This functor operates on factor function values. It will determine the new
// best (smallest for energy minimization) epsilon value which is worse
// (higher) than the old epsilon value.
template<class ACC, class VALUE_TYPE>
class EpsilonFunctor;

// This functor operators on factor function values and is a coordinate functor
// (receives value and coordinate input iterator as arguments).
//
// This functor will write all the labels to the output iterator where the
// value is less than or equal to their variable’s epsilon value.
template<class ACC, class INDEX_TYPE, class VALUE_TYPE, class OUTPUT_ITERATOR>
class NonCollapsedFunctionFunctor;

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

////////////////////////////////////////////////////////////////////////////////
//
// class UnwrapFunctionFunctor
//
////////////////////////////////////////////////////////////////////////////////

template<class FUNCTOR>
class UnwrapFunctionFunctor {
public:
	UnwrapFunctionFunctor(FUNCTOR &functor)
	: functor_(functor)
	{
	}

	// We need this functor class, because the operator() is a template
	// for a specific function type.
	//
	// This way we can access the underlying function object without knowing the
	// concrete type (C++ infers the template arguments for class methods).
	template<class FUNCTION>
	void operator()(const FUNCTION &function) {
		function.forAllValuesInAnyOrderWithCoordinate(functor_);
	}

private:
	FUNCTOR &functor_;
};

////////////////////////////////////////////////////////////////////////////////
//
// class EpsilonFunctor
//
////////////////////////////////////////////////////////////////////////////////

template<class ACC, class VALUE_TYPE>
class EpsilonFunctor {
public:
	EpsilonFunctor(VALUE_TYPE oldValue)
	: oldValue_(oldValue)
	, value_(ACC::template neutral<VALUE_TYPE>())
	{
	}

	VALUE_TYPE value() const {
		return value_;
	}

	void operator()(const VALUE_TYPE v)
	{
		if (ACC::ibop(v, oldValue_) && ACC::bop(v, value_)) {
			value_ = v;
		}
	}

private:
	VALUE_TYPE oldValue_;
	VALUE_TYPE value_;
};

////////////////////////////////////////////////////////////////////////////////
//
// class NonCollapsedFunctionFunctor
//
////////////////////////////////////////////////////////////////////////////////

template<class ACC, class INDEX_TYPE, class VALUE_TYPE, class OUTPUT_ITERATOR>
class NonCollapsedFunctionFunctor {
public:
	NonCollapsedFunctionFunctor(INDEX_TYPE variable, VALUE_TYPE epsilon, OUTPUT_ITERATOR iterator)
	: iterator_(iterator)
	, epsilon_(epsilon)
	, variable_(variable)
	{
	}

	template<class INPUT_ITERATOR>
	void operator()(const VALUE_TYPE v, INPUT_ITERATOR it)
	{
		if (ACC::bop(v, epsilon_))
			*iterator_++ = it[variable_];
	}

private:
	OUTPUT_ITERATOR iterator_;
	VALUE_TYPE epsilon_;
	INDEX_TYPE variable_;
};

} // namepasce labelcollapse
} // namepasce opengm

#endif
