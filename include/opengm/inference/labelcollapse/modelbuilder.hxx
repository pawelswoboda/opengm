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

// Forward declaration for type level function. (TODO: Move this to a header.)
template<class GM, class ACC> struct LabelCollapseAuxTypeGen;

namespace labelcollapse {

// Builds the auxiliary model given the original model. There are different
// implementations available.
template<class GM, class ACC, class DERIVED> class ModelBuilder;
template<class GM, class ACC> class ModelBuilderUnary;
template<class GM, class ACC> class ModelBuilderGeneric;

// There are two different implementations of the ModelBuilder class.
// We use this enum as type level value to decide which implementation to use.
enum UncollapsingBehavior { Unary, Generic };

// Type level function to choose between two different implementation of the
// ModelBuilder class.
template <class GM, class ACC, UncollapsingBehavior T> struct ModelBuilderTypeGen;
template <class GM, class ACC> struct ModelBuilderTypeGen<GM, ACC, Unary>   { typedef ModelBuilderUnary<GM, ACC> Type; };
template <class GM, class ACC> struct ModelBuilderTypeGen<GM, ACC, Generic> { typedef ModelBuilderGeneric<GM, ACC> Type; };

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

	template<class INOUT_ITERATOR> void calculateDepth(INOUT_ITERATOR) const;
	template<class ITERATOR> bool isValidLabeling(ITERATOR) const;
	template<class ITERATOR> void uncollapseLabeling(ITERATOR);
	void originalLabeling(const std::vector<LabelType>&, std::vector<LabelType>&) const;
	template<class ITERATOR> void populate(ITERATOR);
	LabelType numberOfLabels(IndexType i) const { return mappings_[i].size(); }

	// The following functions should be overwritten in descendants.
	void uncollapse(const IndexType);

protected:
	const OriginalModelType *original_;
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
: original_(&gm)
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

	for (IndexType i = 0; i < original_->numberOfVariables(); ++i, ++it) {
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

	for (IndexType i = 0; i < original_->numberOfVariables(); ++i, ++it) {
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
	for (IndexType i = 0; i < original_->numberOfVariables(); ++i, ++it) {
		while (numberOfLabels(i) < std::min(*it, original_->numberOfLabels(i)))
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
	OPENGM_ASSERT(auxiliary.size() == original_->numberOfVariables());

	original.assign(auxiliary.size(), 0);
	for (IndexType i = 0; i < original_->numberOfVariables(); ++i) {
		original[i] = mappings_[i].original(auxiliary[i]);
	}
}

template<class GM, class ACC, class DERIVED>
template<class INOUT_ITERATOR>
void
ModelBuilder<GM, ACC, DERIVED>::calculateDepth
(
	INOUT_ITERATOR it
) const
{
	for (IndexType i = 0; i < original_->numberOfVariables(); ++i, ++it) {
		LabelType aux = mappings_[i].auxiliary(*it);
		*it = mappings_[i].full() ? aux : aux - 1;
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
	using typename Parent::MappingType;

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

	for (IndexType f2 = 0; f2 < original_->numberOfFactors(idx); ++f2) {
		typedef typename OriginalModelType::FactorType FactorType;
		const IndexType f = original_->factorOfVariable(idx, f2);
		const FactorType factor = (*original_)[f];

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

////////////////////////////////////////////////////////////////////////////////
//
// class ModelBuilderGeneric
//
////////////////////////////////////////////////////////////////////////////////

template<class GM, class ACC>
class ModelBuilderGeneric : public ModelBuilder<GM, ACC, ModelBuilderGeneric<GM, ACC> > {
public:
	typedef ModelBuilder<GM, ACC, ModelBuilderGeneric<GM, ACC> > Parent;

	using typename Parent::AccumulationType;
	using typename Parent::OperatorType;
	using typename Parent::OriginalModelType;
	using typename Parent::AuxiliaryModelType;
	using typename Parent::IndexType;
	using typename Parent::LabelType;
	using typename Parent::ValueType;
	using typename Parent::MappingType;

	ModelBuilderGeneric(const OriginalModelType&);

	void buildAuxiliaryModel();
	void uncollapse(const IndexType);

private:
	using Parent::original_;
	using Parent::auxiliary_;
	using Parent::epsilons_;
	using Parent::mappings_;
	using Parent::rebuildNecessary_;

	void updateMappings();
	ValueType calculateNewEpsilon(const IndexType);

	friend Parent;
};

template<class GM, class ACC>
ModelBuilderGeneric<GM, ACC>::ModelBuilderGeneric
(
	const OriginalModelType &gm
)
: Parent(gm)
{
	for (IndexType f = 0; f < original_->numberOfFactors(); ++f)
		epsilons_[f] = calculateNewEpsilon(f);

	for (IndexType i = 0; i < original_->numberOfVariables(); ++i)
		uncollapse(i);
}

template<class GM, class ACC>
void
ModelBuilderGeneric<GM, ACC>::buildAuxiliaryModel()
{
	OPENGM_ASSERT(rebuildNecessary_)
	if (!rebuildNecessary_)
		return;

	// TODO: We probably should update the mapping incrementally. Everything
	// gets faster :-)
	updateMappings();

	Parent::buildAuxiliaryModel();
}

template<class GM, class ACC>
void
ModelBuilderGeneric<GM, ACC>::uncollapse
(
	const IndexType idx
)
{
	bool foundAnyFactor = false;
	IndexType bestFactor = 0;
	ValueType bestEpsilon = ACC::template neutral<ValueType>();
	ValueType bestEpsilonDiff = ACC::template neutral<ValueType>();

	for (IndexType f = 0; f < original_->numberOfFactors(idx); ++f) {
		IndexType factor = original_->factorOfVariable(idx, f);

		ValueType epsilon = calculateNewEpsilon(factor);
		ValueType epsilonDiff = epsilon - epsilons_[factor];

		if ((! foundAnyFactor) || (epsilonDiff < bestEpsilonDiff)) {
			foundAnyFactor = true;
			bestFactor = factor;
			bestEpsilon = epsilon;
			bestEpsilonDiff = epsilonDiff;
		}
	}

	// If the variable is not the first variable of any factor, then we
	// should not update any factor. :-)
	if (foundAnyFactor) {
		epsilons_[bestFactor] = bestEpsilon;
		rebuildNecessary_ = true;
	}
}

template<class GM, class ACC>
typename ModelBuilderGeneric<GM, ACC>::ValueType
ModelBuilderGeneric<GM, ACC>::calculateNewEpsilon(
	const IndexType idx
)
{
	const ValueType &epsilon = epsilons_[idx];
	const typename OriginalModelType::FactorType &factor = (*original_)[idx];

	EpsilonFunctor<ACC, ValueType> functor(epsilon);
	factor.forAllValuesInAnyOrder(functor);
	return functor.value();
}

template<class GM, class ACC>
void
ModelBuilderGeneric<GM, ACC>::updateMappings()
{
	typedef std::vector<LabelType> LabelVec;
	typedef typename LabelVec::iterator Iterator;
	typedef std::back_insert_iterator<LabelVec> Inserter;
	typedef NonCollapsedFunctionFunctor<ACC, IndexType, ValueType, Inserter> FunctionFunctor;
	typedef UnwrapFunctionFunctor<FunctionFunctor> FactorFunctor;

	for (IndexType i = 0; i < original_->numberOfVariables(); ++i) {
		bool foundAnyFactor = false;
		LabelVec nonCollapsed;
		Inserter inserter(nonCollapsed);

		for (IndexType j = 0; j < original_->numberOfFactors(i); ++j) {
			const IndexType f = original_->factorOfVariable(i, j);
			const typename OriginalModelType::FactorType &factor = (*original_)[f];

			IndexType varIdx = 0;
			for (IndexType k = 0; k < factor.numberOfVariables(); ++k) {
				if (factor.variableIndex(k) == i) {
					varIdx = k;
					break;
				}
			}

			FunctionFunctor functionFunctor(varIdx, epsilons_[f], inserter);
			FactorFunctor factorFunctor(functionFunctor);
			factor.callFunctor(factorFunctor);
			foundAnyFactor = true;
		}

		MappingType &m = mappings_[i];
		m = MappingType(original_->numberOfLabels(i));

		// If there are no factors where our variable is the first variable,
		// there are no corresponding epsilon values for this variable. We have
		// no information and cannot distinguish the labels of this variable.
		//
		// The only reasonable action is to include all labels for this
		// variable (make mapping full bijection).
		//
		// In all other cases we introduce the labels for which found some
		// epsilon which is larger. The mapping class will automatically
		// convert itself into a full bijection if it detects that it is
		// necessary.
		if (foundAnyFactor) {
			for (Iterator it = nonCollapsed.begin(); it != nonCollapsed.end(); ++it) {
				m.insert(*it);
			}
		} else {
			m.makeFull();
		}
	}

	rebuildNecessary_ = true;
}

} // namespace labelcollapse
} // namespace opengm

#endif
