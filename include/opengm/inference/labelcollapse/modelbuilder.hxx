//
// File: modelbuilder.hxx
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
#ifndef OPENGM_LABELCOLLAPSE_MODELBUILDER_HXX
#define OPENGM_LABELCOLLAPSE_MODELBUILDER_HXX

#include<vector>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/discretespace.hxx>
#include <opengm/datastructures/fast_sequence.hxx>
#include <opengm/utilities/indexing.hxx>

#include "utils.hxx"


namespace opengm {

namespace labelcollapse {

// Type level function for calculation of the auxiliary model type.
template<class GM, class ACC>
struct ModelBuilderAuxTypeGen {
	typedef typename GM::OperatorType OperatorType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename GM::ValueType ValueType;

	typedef typename opengm::DiscreteSpace<IndexType, LabelType> SpaceType;
	typedef typename meta::TypeListGenerator< EpsilonFunction<GM, ACC> >::type FunctionTypeList;

	typedef GraphicalModel<ValueType, OperatorType, FunctionTypeList, SpaceType> GraphicalModelType;
};

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
	typedef typename ModelBuilderAuxTypeGen<GM, ACC>::GraphicalModelType AuxiliaryModelType;

	typedef Mapping<OriginalModelType> MappingType;

	//
	// Methods
	//
	ModelBuilder(const OriginalModelType&);

	void buildAuxiliaryModel();
	bool rebuildNecessary() const { return rebuildNecessary_; }

	const OriginalModelType& getOriginalModel() const { return *original_; }
	const AuxiliaryModelType& getAuxiliaryModel() const
	{
		OPENGM_ASSERT(!rebuildNecessary_);
		return auxiliary_;
	}

	template<class IN_ITER> bool isValidLabeling(IN_ITER) const;
	template<class IN_ITER, class OUT_ITER> void originalLabeling(IN_ITER, OUT_ITER) const;
	template<class IN_ITER, class OUT_ITER> void auxiliaryLabeling(IN_ITER, OUT_ITER) const;

	template<class IN_ITER, class OUT_ITER> bool moveToAuxiliary(IN_ITER, OUT_ITER) const;
	template<class OUT_ITER> void initialLabeling(OUT_ITER) const;

	template<class IN_ITER, class OUT_ITER> void calculateDepth(IN_ITER, OUT_ITER) const;
	LabelType numberOfLabels(IndexType i) const { return mappings_[i].size(); }

	bool isUncollapsable(const IndexType);
	bool uncollapse(const IndexType);
	template<class INPUT_ITERATOR> bool uncollapseLabeling(INPUT_ITERATOR);

private:
	typedef std::vector<LabelType> Stack;

	void internalChecks() const;

	const OriginalModelType *original_;
	AuxiliaryModelType auxiliary_;
	std::vector<MappingType> mappings_;
	bool rebuildNecessary_;
	std::vector<Stack> collapsed_;
};

template<class GM, class ACC>
ModelBuilder<GM, ACC>::ModelBuilder
(
	const OriginalModelType &gm
)
: original_(&gm)
, mappings_(gm.numberOfFactors(), MappingType(0))
, rebuildNecessary_(true)
, collapsed_(gm.numberOfVariables())
{
	// First set up the mapping. This is needed for the later uncollapsing
	// steps.
	for (IndexType i = 0; i < gm.numberOfVariables(); ++i)
		mappings_[i] = MappingType(gm.numberOfLabels(i));

	// Order nodes by there unary potential.
	Reordering<OriginalModelType, AccumulationType> reordering(*original_);
	for (IndexType i = 0; i < original_->numberOfVariables(); ++i) {
		collapsed_[i].resize(original_->numberOfLabels(i));

		// We reverse the ordering and use .rbegin(), because we use the
		// std::vector as a stack. The smallest element should be the last one.
		reordering.reorder(i);
		reordering.getOrdered(collapsed_[i].rbegin());
	}

	// From this point on all the invariances should hold.
	internalChecks();

	for (IndexType i = 0; i < original_->numberOfVariables(); ++i)
		uncollapse(i);

	internalChecks();
}

template<class GM, class ACC>
void
ModelBuilder<GM, ACC>::buildAuxiliaryModel()
{
	if (!rebuildNecessary_)
		return;

	// Build space.
	std::vector<LabelType> shape(original_->numberOfVariables());
	for (IndexType i = 0; i < original_->numberOfVariables(); ++i) {
		shape[i] = mappings_[i].size();
	}
	typename AuxiliaryModelType::SpaceType space(shape.begin(), shape.end());
	auxiliary_ = AuxiliaryModelType(space);

	// Build graphical models with all factors.
	for (IndexType i = 0; i < original_->numberOfFactors(); ++i) {
		typedef EpsilonFunction<OriginalModelType, AccumulationType> ViewFunction;

		const typename OriginalModelType::FactorType &factor = (*original_)[i];
		const ViewFunction func(factor, mappings_);

		auxiliary_.addFactor(
			auxiliary_.addFunction(func),
			factor.variableIndicesBegin(),
			factor.variableIndicesEnd()
		);
	}

	rebuildNecessary_ = false;
}

template<class GM, class ACC>
template<class INPUT_ITERATOR>
bool
ModelBuilder<GM, ACC>::isValidLabeling
(
	INPUT_ITERATOR it
) const
{
	OPENGM_ASSERT(!rebuildNecessary_);

	for (IndexType i = 0; i < original_->numberOfVariables(); ++i, ++it) {
		if (mappings_[i].isCollapsedAuxiliary(*it))
			return false;
	}

	return true;
}

template<class GM, class ACC>
template<class INPUT_ITERATOR>
bool
ModelBuilder<GM, ACC>::uncollapseLabeling
(
	INPUT_ITERATOR it
)
{
	OPENGM_ASSERT(!rebuildNecessary_);
	bool result = false;

	for (IndexType i = 0; i < original_->numberOfVariables(); ++i, ++it) {
		if (mappings_[i].isCollapsedAuxiliary(*it))
			result = result || uncollapse(i);
	}

	return result;
}

template<class GM, class ACC>
template<class IN_ITER, class OUT_ITER>
void
ModelBuilder<GM, ACC>::originalLabeling
(
	IN_ITER auxiliary,
	OUT_ITER original
) const
{
	OPENGM_ASSERT(! rebuildNecessary_);
	OPENGM_ASSERT(isValidLabeling(auxiliary));

	for (IndexType i = 0; i < original_->numberOfVariables();
	     ++i, ++auxiliary, ++original)
	{
		*original = mappings_[i].original(*auxiliary);
	}
}

template<class GM, class ACC>
template<class IN_ITER, class OUT_ITER>
void
ModelBuilder<GM, ACC>::auxiliaryLabeling
(
	IN_ITER original,
	OUT_ITER auxiliary
) const
{
	OPENGM_ASSERT(! rebuildNecessary_);

	for (IndexType i = 0; i < original_->numberOfVariables();
	     ++i, ++original, ++auxiliary)
	{
		*auxiliary = mappings_[i].auxiliary(*original);
	}
}

template<class GM, class ACC>
template<class IN_ITER, class OUT_ITER>
bool
ModelBuilder<GM, ACC>::moveToAuxiliary
(
	IN_ITER in,
	OUT_ITER out
) const
{
	bool result = false;

	for (IndexType i = 0; i < original_->numberOfVariables(); ++i, ++in, ++out) {
		if (mappings_[i].full()) {
			*out = *in;
		} else {
			*out = mappings_[i].full() ? *in : 0;
		}
	}

	return result;
}

template<class GM, class ACC>
template<class OUT_ITER>
void
ModelBuilder<GM, ACC>::initialLabeling
(
	OUT_ITER out
) const
{
	for (IndexType i = 0; i < original_->numberOfVariables(); ++i, ++out)
		*out = mappings_[i].full() ? 0 : 1;
}

template<class GM, class ACC>
template<class IN_ITER, class OUT_ITER>
void
ModelBuilder<GM, ACC>::calculateDepth
(
	IN_ITER origLabeling,
	OUT_ITER depth
) const
{
	OPENGM_ASSERT(! rebuildNecessary_);

	for (IndexType i = 0; i < original_->numberOfVariables();
	     ++i, ++origLabeling, ++depth)
	{
		LabelType aux = mappings_[i].auxiliary(*origLabeling);
		*depth = mappings_[i].full() ? aux : aux - 1;
	}
}

template<class GM, class ACC>
bool
ModelBuilder<GM, ACC>::isUncollapsable
(
	const IndexType idx
)
{
	return !mappings_[idx].full();
}

template<class GM, class ACC>
bool
ModelBuilder<GM, ACC>::uncollapse
(
	const IndexType idx
)
{
	internalChecks();

	if (!mappings_[idx].full()) {
		OPENGM_ASSERT(collapsed_[idx].size() > 0);

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

		rebuildNecessary_ = true;
		return true;
	}

	return false;
}

template<class GM, class ACC>
void
ModelBuilder<GM, ACC>::internalChecks() const
{
#ifndef NDEBUG
	for (IndexType i = 0; i < original_->numberOfVariables(); ++i) {
		for (LabelType j = 0; j < original_->numberOfLabels(i); ++j) {
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

} // namespace labelcollapse
} // namespace opengm

#endif
