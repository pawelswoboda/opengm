//
// File: reparameterization.hxx
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
#ifndef OPENGM_LABELCOLLAPSE_REPARAMETERIZATION_HXX
#define OPENGM_LABELCOLLAPSE_REPARAMETERIZATION_HXX

#include <boost/scoped_ptr.hpp>

#include <opengm/inference/auxiliary/lp_reparametrization.hxx>
#include <opengm/inference/trws/trws_base.hxx>
#include <opengm/inference/trws/trws_decomposition.hxx>
#include <opengm/inference/trws/trws_reparametrization.hxx>
#include <opengm/inference/trws/trws_trws.hxx>

////////////////////////////////////////////////////////////////////////////////
//
// FIXME: THE WHOLE REPARAMETERIZATION MACHINERY IN **THIS** HEADER FILE
// ONLY WORKS FOR UNARY/PAIRWISE MARKOV RANDOM FIELDS. SORRY.
//
////////////////////////////////////////////////////////////////////////////////

namespace opengm {
namespace labelcollapse {

////////////////////////////////////////////////////////////////////////////////
//
// Utility functions.
//
////////////////////////////////////////////////////////////////////////////////

template<class T>
bool
is_almost_equal
(
	const T v1,
	const T v2
)
{
	return std::abs(v1 - v2) < 1e-8;
}

template<class T>
void
permutePairwise
(
	marray::Marray<T> &x
)
{
	FastSequence<size_t> permutation(2);
	permutation[0] = 1;
	permutation[1] = 0;
	marray::Marray<T> result(x.permutedView(permutation.begin()));

	#ifndef NDEBUG
	for (size_t i = 0; i < x.shape(0); ++i)
		for (size_t j = 0; j < x.shape(1); ++j)
			OPENGM_ASSERT(is_almost_equal(x(i, j), result(j, i)));
	#endif

	x = result;
}

template<class GM>
typename GM::IndexType
unaryFactorIndex
(
	const GM &gm,
	const typename GM::IndexType variable
)
{
	marray::Vector<typename GM::IndexType> factorIds;
	size_t count = gm.numberOfNthOrderFactorsOfVariable(variable, 1, factorIds);
	OPENGM_ASSERT_OP(count, ==, 1);
	OPENGM_ASSERT_OP(gm[factorIds[0]].variableIndex(0), ==, variable);
	return factorIds[0];
}

template<class GM>
marray::Marray<typename GM::ValueType>
copyUnary
(
	const GM &gm,
	const typename GM::IndexType variable
)
{
	typename GM::IndexType index = unaryFactorIndex(gm, variable);

	marray::Marray<typename GM::ValueType> result(
		gm[index].shapeBegin(), gm[index].shapeEnd());
	gm[index].copyValues(result.begin());

	return result;
}

template<class GM>
marray::Marray<typename GM::ValueType>
copyUnaryWeighted
(
	const GM &gm,
	const typename GM::IndexType variable,
	const typename GM::ValueType weight
)
{
	marray::Marray<typename GM::ValueType> result = copyUnary(gm, variable);
	std::transform(
		result.begin(), result.end(),
		result.begin(),
		std::bind2nd(std::multiplies<typename GM::ValueType>(), weight)
	);
	return result;
}

template<class GM>
marray::Marray<typename GM::ValueType>
copyUnaryFromRepa
(
	const GM &gm,
	const typename GM::IndexType variable,
	const LPReparametrisationStorage<GM> &repa
)
{
	typename GM::IndexType index = unaryFactorIndex(gm, variable);

	marray::Marray<typename GM::ValueType> result(
		gm[index].shapeBegin(), gm[index].shapeEnd());
	repa.copyFactorValues(index, result.begin());

	#ifndef NDEBUG
	OPENGM_ASSERT_OP(gm[index].numberOfVariables(), ==, 1);
	OPENGM_ASSERT_OP(gm[index].shape(0), ==, result.shape(0));
	for (typename GM::LabelType i = 0; i < gm[index].shape(0); ++i)
		OPENGM_ASSERT(is_almost_equal(result(i), repa.getVariableValue(index, i)));
	#endif

	return result;
}

template<class GM>
marray::Marray<typename GM::ValueType>
copyUnaryFromRepaWeighted
(
	const GM &gm,
	const typename GM::IndexType variable,
	const LPReparametrisationStorage<GM> &repa,
	const typename GM::ValueType weights
)
{
	marray::Marray<typename GM::ValueType> fromRepa = copyUnaryFromRepa(gm, variable, repa);
	marray::Marray<typename GM::ValueType> weigthed = copyUnaryWeighted(gm, variable, 1.0 - weights);

	std::transform(
		fromRepa.begin(), fromRepa.end(),
		weigthed.begin(),
		fromRepa.begin(),
		std::minus<typename GM::ValueType>()
	);
	return fromRepa;
}

template<class GM>
typename GM::IndexType
pairwiseFactorIndex
(
	const GM &gm,
	const typename GM::IndexType left,
	const typename GM::IndexType right
)
{
	OPENGM_ASSERT(gm.variableVariableConnection(left, right));
	marray::Vector<typename GM::IndexType> leftFactorIds, rightFactorIds;

	gm.numberOfNthOrderFactorsOfVariable(left, 2, leftFactorIds);
	gm.numberOfNthOrderFactorsOfVariable(right, 2, rightFactorIds);
	std::sort(leftFactorIds.begin(), leftFactorIds.end());
	std::sort(rightFactorIds.begin(), rightFactorIds.end());

	std::vector<typename GM::IndexType> intersection;
	std::set_intersection(leftFactorIds.begin(), leftFactorIds.end(), rightFactorIds.begin(), rightFactorIds.end(), std::back_inserter(intersection));
	OPENGM_ASSERT_OP(intersection.size(), ==, 1);

	#ifndef NDEBUG
	const typename GM::FactorType &f = gm[intersection[0]];
	OPENGM_ASSERT(
		(f.variableIndex(0) == left && f.variableIndex(1) == right) ||
		(f.variableIndex(1) == left && f.variableIndex(0) == right)
	);
	#endif

	return intersection[0];
}

template<class GM>
marray::Marray<typename GM::ValueType>
copyPairwise
(
	const GM &gm,
	const typename GM::IndexType left,
	const typename GM::IndexType right
)
{
	typename GM::IndexType index = pairwiseFactorIndex(gm, left, right);

	marray::Marray<typename GM::ValueType> result(
		gm[index].shapeBegin(), gm[index].shabeEnd());
	gm[index].copyValues(result.begin());

	if (gm[index].variableIndex(0) == right)
		permutePairwise(result);

	return result;
}

template<class GM>
marray::Marray<typename GM::ValueType>
copyPairwiseFromRepa
(
	const GM &gm,
	const typename GM::IndexType left,
	const typename GM::IndexType right,
	const LPReparametrisationStorage<GM> &repa
)
{
	typename GM::IndexType index = pairwiseFactorIndex(gm, left, right);
	marray::Marray<typename GM::ValueType> result(
		gm[index].shapeBegin(), gm[index].shapeEnd());
	repa.copyFactorValues(index, result.begin());

	#ifndef NDEBUG
	OPENGM_ASSERT_OP(gm[index].numberOfVariables(), ==, 2);
	OPENGM_ASSERT_OP(gm[index].shape(0), ==, result.shape(0));
	OPENGM_ASSERT_OP(gm[index].shape(1), ==, result.shape(1));
	for (typename GM::LabelType i = 0; i < gm[index].shape(0); ++i) {
		for (typename GM::LabelType j = 0; j < gm[index].shape(1); ++j) {
			FastSequence<typename GM::LabelType> labeling(2);
			labeling[0] = i;
			labeling[1] = j;
			OPENGM_ASSERT(is_almost_equal(result(labeling.begin()), repa.getFactorValue(index, labeling.begin())));
		}
	}
	#endif

	if (gm[index].variableIndex(0) == right)
		permutePairwise(result);

	return result;
}

template<class GM>
std::pair<typename GM::ValueType*, typename GM::ValueType*>
getPencil
(
	const GM &gm,
	const typename GM::IndexType from,
	const typename GM::IndexType to,
	LPReparametrisationStorage<GM> &repa
)
{
	typename GM::IndexType index = pairwiseFactorIndex(gm, from, to);

	if (gm[index].variableIndex(0) == from) {
		return repa.getIterators(index, 0);
	} else {
		return repa.getIterators(index, 1);
	}
}

template<class GM1, class GM2>
void
trivialMerge
(
	const LPReparametrisationStorage<GM1> &input,
	LPReparametrisationStorage<GM2> &output
)
{
	const GM2 &gm = output.graphicalModel();
	for (typename GM2::IndexType fidx = 0; fidx < gm.numberOfFactors(); ++fidx) {
		if (gm[fidx].numberOfVariables() < 2)
			continue;

		for (typename GM2::IndexType rel = 0; rel < gm[fidx].numberOfVariables(); ++rel) {
			std::pair<typename GM2::ValueType*, typename GM2::ValueType*> its1, its2;
			its1 = output.getIterators(fidx, rel);
			its2 = const_cast<LPReparametrisationStorage<GM1>&>(input).getIterators(fidx, rel);
			OPENGM_ASSERT_OP(its1.second - its1.first, ==, its2.second - its2.first);

			std::transform(
				its1.first, its1.second,
				its2.first, its1.first,
				std::plus<typename GM2::ValueType>()
			);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//
// class SequenceGeneratorIterator
//
////////////////////////////////////////////////////////////////////////////////

// FIXME: Having two template arguments is also just a hack.
// I'm not even sure what was the reason to introduce a second GM. Maybe it is
// now obselete.
template<class GM>
class SequenceGeneratorIterator {
public:
	typedef GM GraphicalModelType;
	typedef typename GraphicalModelType::IndexType IndexType;
	typedef typename GraphicalModelType::ValueType ValueType;

	typedef trws_base::DecompositionStorage<GM> DecompositionStorageType;
	typedef trws_base::SequenceStorage<GM> SequenceStorageType;
	typedef std::pair<SequenceGeneratorIterator<GM>, SequenceGeneratorIterator<GM> > Iterators;

	typedef std::pair<IndexType, ValueType> Element;
	typedef std::vector<Element> Elements;

	bool operator!=(const SequenceGeneratorIterator&);
	Elements& operator*();
	SequenceGeneratorIterator<GM>& operator++();

	static Iterators makeIterators(const DecompositionStorageType&);

private:
	SequenceGeneratorIterator(size_t index, const DecompositionStorageType&);

	size_t index_; 
	const DecompositionStorageType *storage_;
	Elements current_;
};

template<class GM>
SequenceGeneratorIterator<GM>::SequenceGeneratorIterator
(
	size_t index,
	const DecompositionStorageType &storage
)
: index_(index)
, storage_(&storage)
{
}

template<class GM>
bool
SequenceGeneratorIterator<GM>::operator!=(const SequenceGeneratorIterator &rhs)
{
	return (index_ != rhs.index_) || (storage_ != rhs.storage_);
}

template<class GM>
typename SequenceGeneratorIterator<GM>::Elements&
SequenceGeneratorIterator<GM>::operator*()
{
	OPENGM_ASSERT_OP(index_, <, storage_->numberOfModels());
	const SequenceStorageType &seq = storage_->subModel(index_);
	current_.resize(seq.size());
	for (IndexType i = 0; i < seq.size(); ++i) {
		current_[i].first = seq.varIndex(i);
		current_[i].second = static_cast<ValueType>(1.0)
		                   / storage_->getSubVariableList(current_[i].first).size();
	}
	return current_;
}

template<class GM>
SequenceGeneratorIterator<GM>&
SequenceGeneratorIterator<GM>::operator++()
{
	++index_;
	return *this;
}

template<class GM>
typename SequenceGeneratorIterator<GM>::Iterators
SequenceGeneratorIterator<GM>::makeIterators
(
	const DecompositionStorageType &storage
)
{
	return std::make_pair(
		SequenceGeneratorIterator(0, storage),
		SequenceGeneratorIterator(storage.numberOfModels(), storage)
	);
}

////////////////////////////////////////////////////////////////////////////////
//
// class SequenceReparameterizer
//
////////////////////////////////////////////////////////////////////////////////

// Uses CRTP pattern for static polymorphism.
template<class GM, class DERIVED>
class SequenceReparameterizer {
public:
	typedef GM GraphicalModelType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename GM::ValueType ValueType;

	typedef LPReparametrisationStorage<GM> StorageType;
	typedef std::pair<IndexType, ValueType> ElementType;
	typedef std::vector<ElementType> SequenceType;
	typedef marray::Marray<ValueType> Potentials;
	typedef std::pair<ValueType*, ValueType*> Iterators;

	SequenceReparameterizer(const GraphicalModelType&, const SequenceType&);
	StorageType run();

	template<class INPUT_ITERATOR>
	static StorageType reparameterizeAll(const GM &gm, INPUT_ITERATOR begin, INPUT_ITERATOR end);

protected:
	void forwardPass();
	void backwardPass();

	Potentials copyUnary(IndexType idx) const;
	Potentials copyPairwise(IndexType i, IndexType j) const;
	Iterators pencil(IndexType from, IndexType to);

	const GraphicalModelType &gm_;
	const SequenceType &sequence_;
	StorageType repa_;
};

template<class GM, class DERIVED>
SequenceReparameterizer<GM, DERIVED>::SequenceReparameterizer
(
	const GraphicalModelType &gm,
	const SequenceType &sequence
)
: gm_(gm)
, sequence_(sequence)
, repa_(gm_)
{
}

template<class GM, class DERIVED>
typename SequenceReparameterizer<GM, DERIVED>::StorageType
SequenceReparameterizer<GM, DERIVED>::run()
{
	#ifndef NDEBUG
	std::cout << "SequenceReparameterizer::run()" << std::endl;
	#endif
	repa_ = StorageType(gm_);

	if (sequence_.size() >= 2) {
		static_cast<DERIVED*>(this)->forwardPass();
		static_cast<DERIVED*>(this)->backwardPass();
	}

	return repa_;
}

template<class GM, class DERIVED>
typename SequenceReparameterizer<GM, DERIVED>::Potentials
SequenceReparameterizer<GM, DERIVED>::copyUnary
(
	IndexType idx
) const
{
	OPENGM_ASSERT_OP(idx, >=, 0);
	OPENGM_ASSERT_OP(idx, <, sequence_.size());
	return copyUnaryFromRepaWeighted(gm_, sequence_[idx].first, repa_, sequence_[idx].second);
}

template<class GM, class DERIVED>
typename SequenceReparameterizer<GM, DERIVED>::Potentials
SequenceReparameterizer<GM, DERIVED>::copyPairwise
(
	IndexType i,
	IndexType j
) const
{
	OPENGM_ASSERT(i == j+1 || i == j-1);
	OPENGM_ASSERT_OP(i, >=, 0);
	OPENGM_ASSERT_OP(j, >=, 0);
	OPENGM_ASSERT_OP(i, <, sequence_.size());
	OPENGM_ASSERT_OP(i, <, sequence_.size());
	return copyPairwiseFromRepa(gm_, sequence_[i].first, sequence_[j].first, repa_);
}

template<class GM, class DERIVED>
typename SequenceReparameterizer<GM, DERIVED>::Iterators
SequenceReparameterizer<GM, DERIVED>::pencil
(
	IndexType i,
	IndexType j
)
{
	OPENGM_ASSERT(i == j+1 || i == j-1);
	OPENGM_ASSERT_OP(i, >=, 0);
	OPENGM_ASSERT_OP(j, >=, 0);
	OPENGM_ASSERT_OP(i, <, sequence_.size());
	OPENGM_ASSERT_OP(i, <, sequence_.size());
	return getPencil(gm_, sequence_[i].first, sequence_[j].first, repa_);
}

template<class GM, class DERIVED>
void
SequenceReparameterizer<GM, DERIVED>::forwardPass()
{
	#ifndef NDEBUG
	std::cout << "SequenceReparameterizer::forwardPass()" << std::endl;
	#endif
	Potentials pots;
	Iterators its;

	for (IndexType i = 1; i < sequence_.size(); ++i) {

		// Push potential from unary into pencil.
		pots = copyUnary(i-1);
		its = pencil(i-1, i);
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i-1].first); ++j) {
			OPENGM_ASSERT_OP(j, <, its.second - its.first);
			*(its.first + j) += pots(j);
		}

		// Check whether all unary are empty.
		#ifndef NDEBUG
		pots = copyUnary(i-1); 
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i-1].first); ++j)
			OPENGM_ASSERT(is_almost_equal(static_cast<ValueType>(0), pots(j)));
		#endif

		// Push potential from pencil to next unary.
		pots = copyPairwise(i, i-1);
		its = pencil(i, i-1);
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i].first); ++j) {
			OPENGM_ASSERT_OP(j, <, its.second - its.first);

			ValueType min = pots(j, 0);
			for (LabelType k = 1; k < gm_.numberOfLabels(sequence_[i-1].first); ++k)
				min = std::min(min, pots(j, k));

			*(its.first + j) -= min;
		}

		// Check whether minimum of each pencil is zero.
		#ifndef NDEBUG
		pots = copyPairwise(i, i-1); 
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i].first); ++j) {
			ValueType min = pots(j, 0);
			for (LabelType k = 1; k < gm_.numberOfLabels(sequence_[i-1].first); ++k)
				min = std::min(min, pots(j, k));

			OPENGM_ASSERT(is_almost_equal(static_cast<ValueType>(0), min));
		}
		#endif
	}
}

template<class GM, class DERIVED>
void
SequenceReparameterizer<GM, DERIVED>::backwardPass()
{
	#ifndef NDEBUG
	std::cout << "SequenceReparameterizer::backwardPass()" << std::endl;
	#endif
}

template<class GM, class DERIVED>
template<class INPUT_ITERATOR>
typename SequenceReparameterizer<GM, DERIVED>::StorageType
SequenceReparameterizer<GM, DERIVED>::reparameterizeAll
(
	const GM &gm,
	INPUT_ITERATOR begin,
	INPUT_ITERATOR end
)
{
	#ifndef NDEBUG
	std::cout << "SequenceReparameterizer::reparameterizeAll()" << std::endl;
	#endif
	StorageType repa_(gm);
	for (INPUT_ITERATOR it = begin; it != end; ++it) {
		DERIVED reparameterizer(gm, *it);
		trivialMerge(reparameterizer.run(), repa_);
	}

	#ifndef NDEBUG
	typedef typename GM::IndexType IndexType;
	typedef typename GM::FactorType FactorType;
	typedef typename FactorType::ShapeIteratorType ShapeIteratorType;
	std::cout << "-- BEGIN ORIGINAL GM --" << std::endl;
	for (IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		std::cout << "FACTOR " << i << ":" << std::endl;

		ShapeWalkerSwitchedOrder<ShapeIteratorType> walker(gm[i].shapeBegin(), gm[i].numberOfVariables());
		for (IndexType j = 0; j < gm[i].size(); ++j, ++walker) {
			const FastSequence<LabelType> &labeling = walker.coordinateTuple();
			for (typename FastSequence<LabelType>::ConstIteratorType it = labeling.begin();
			     it != labeling.end(); ++it)
			{
				std::cout << " " << *it;
			}
			std::cout << "  ->  " << gm[i](labeling.begin()) << std::endl;
		}
	}
	std::cout << "-- END ORIGINAL GM --" << std::endl;

	std::cout << "-- BEGIN REPARAMETRIZED GM --" << std::endl;
	for (IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		std::cout << "FACTOR " << i << ":" << std::endl;

		ShapeWalkerSwitchedOrder<ShapeIteratorType> walker(gm[i].shapeBegin(), gm[i].numberOfVariables());
		for (IndexType j = 0; j < gm[i].size(); ++j, ++walker) {
			const FastSequence<LabelType> &labeling = walker.coordinateTuple();
			for (typename FastSequence<LabelType>::ConstIteratorType it = labeling.begin();
			     it != labeling.end(); ++it)
			{
				std::cout << " " << *it;
			}
			std::cout << "  ->  " << repa_.getFactorValue(i, labeling.begin()) << std::endl;
		}
	}
	std::cout << "-- END REPARAMETRIZED GM --" << std::endl;

	// PrintTestData is only declared if this macro is defined.
	#ifdef TRWS_DEBUG_OUTPUT
	std::cout << "-- BEGIN REPARAMETRIZATION --" << std::endl;
	repa_.PrintTestData(std::cout);
	std::cout << "-- END REPARAMETRIZATION --" << std::endl;
	#endif
	#endif

	return repa_;
}

////////////////////////////////////////////////////////////////////////////////
//
// class CanonicalReparameterizer
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
class CanonicalReparameterizer : public SequenceReparameterizer<GM, CanonicalReparameterizer<GM> > {
public:
	typedef SequenceReparameterizer<GM, CanonicalReparameterizer<GM> > Parent;

	using typename Parent::GraphicalModelType;
	using typename Parent::IndexType;
	using typename Parent::LabelType;
	using typename Parent::ValueType;

	using typename Parent::StorageType;
	using typename Parent::ElementType;
	using typename Parent::SequenceType;
	using typename Parent::Potentials;
	using typename Parent::Iterators;

	CanonicalReparameterizer(const GraphicalModelType&, const SequenceType&);

protected:
	using Parent::copyUnary;
	using Parent::copyPairwise;
	using Parent::pencil;

	using Parent::gm_;
	using Parent::sequence_;
	using Parent::repa_;

	void backwardPass();

	friend Parent;
};

template<class GM>
CanonicalReparameterizer<GM>::CanonicalReparameterizer
(
	const GraphicalModelType &gm,
	const SequenceType &sequence
)
: Parent(gm, sequence)
{
}

template<class GM>
void
CanonicalReparameterizer<GM>::backwardPass()
{
	#ifndef NDEBUG
	std::cout << "CanonicalReparameterizer::backwardPass()" << std::endl;
	#endif
	Potentials pots;
	Iterators its;

	// Calculate energy (minimal marginal of last node). The energy is only
	// used for verifying the reparameterization in debug mode.
	#ifndef NDEBUG
	pots = copyUnary(sequence_.size()-1);
	ValueType energy = pots(0);
	for (LabelType i = 1; i < gm_.numberOfLabels(sequence_[sequence_.size()-1].first); ++i)
		energy = std::min(energy, pots(i));
	#endif

	// Backward move
	for (IndexType i = sequence_.size()-1; i > 0; --i) {
		ValueType min;

		// Push potential from unary to pencil.
		pots = copyUnary(i);
		its = pencil(i, i-1);
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i].first); ++j) {
			OPENGM_ASSERT_OP(j, <, its.second - its.first);
			*(its.first + j) += pots(j) * i / (i + 1);
		}

		// Check whether best unary has uniform part of energy.
		#ifndef NDEBUG
		pots = copyUnary(i);
		min = pots(0);
		for (LabelType j = 1; j < gm_.numberOfLabels(sequence_[i].first); ++j)
			min = std::min(min, pots(j));
		OPENGM_ASSERT(is_almost_equal(energy / sequence_.size(), min));
		#endif

		// Push potential from pencil to unary.
		pots = copyPairwise(i-1, i);
		its = pencil(i-1, i);
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i-1].first); ++j) {
			OPENGM_ASSERT_OP(j, <, its.second - its.first);
			min = pots(j, 0);
			for (LabelType k = 1; k < gm_.numberOfLabels(sequence_[i].first); ++k)
				min = std::min(min, pots(j, k));
			*(its.first + j) -= min;
		}

		// Check whether best pairwise has uniform part of energy.
		#ifndef NDEBUG
		pots = copyPairwise(i-1, i);
		min = pots(0, 0);
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i-1].first); ++j)
			for (LabelType k = 0; k < gm_.numberOfLabels(sequence_[i].first); ++k)
				min = std::min(min, pots(j, k));
		OPENGM_ASSERT(is_almost_equal(static_cast<ValueType>(0), min));
		#endif
	}
}

////////////////////////////////////////////////////////////////////////////////
//
// class UniformReparameterizer
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
class UniformReparameterizer : public SequenceReparameterizer<GM, UniformReparameterizer<GM> > {
public:
	typedef SequenceReparameterizer<GM, UniformReparameterizer<GM> > Parent;

	using typename Parent::GraphicalModelType;
	using typename Parent::IndexType;
	using typename Parent::LabelType;
	using typename Parent::ValueType;

	using typename Parent::StorageType;
	using typename Parent::ElementType;
	using typename Parent::SequenceType;
	using typename Parent::Potentials;
	using typename Parent::Iterators;

	UniformReparameterizer(const GraphicalModelType&, const SequenceType&);

protected:
	using Parent::copyUnary;
	using Parent::copyPairwise;
	using Parent::pencil;

	using Parent::gm_;
	using Parent::sequence_;
	using Parent::repa_;

	void backwardPass();

	friend Parent;
};

template<class GM>
UniformReparameterizer<GM>::UniformReparameterizer
(
	const GraphicalModelType &gm,
	const SequenceType &sequence
)
: Parent(gm, sequence)
{
}

template<class GM>
void
UniformReparameterizer<GM>::backwardPass()
{
	#ifndef NDEBUG
	std::cout << "UniformReparameterizer::backwardPass()" << std::endl;
	#endif
	Potentials pots;
	Iterators its;

	// Calculate energy (minimal marginal of last node). The energy is only
	// used for verifying the reparameterization in debug mode.
	#ifndef NDEBUG
	pots = copyUnary(sequence_.size()-1); 
	ValueType energy = pots(0);
	for (LabelType i = 1; i < gm_.numberOfLabels(sequence_[sequence_.size()-1].first); ++i)
		energy = std::min(energy, pots(i));
	#endif

	// Backward move
	for (IndexType i = sequence_.size()-1; i > 0; --i) {
		ValueType min;

		// Push potential from unary to pencil.
		pots = copyUnary(i);
		its = pencil(i, i-1);
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i].first); ++j) {
			OPENGM_ASSERT_OP(j, <, its.second - its.first);
			*(its.first + j) += pots(j) * (2*i) / (2*i + 1);
		}

		// Check whether best unary has uniform part of energy.
		#ifndef NDEBUG
		pots = copyUnary(i);
		min = pots(0);
		for (LabelType j = 1; j < gm_.numberOfLabels(sequence_[i].first); ++j)
			min = std::min(min, pots(j));
		OPENGM_ASSERT(is_almost_equal(energy / (2*sequence_.size()-1), min));
		#endif

		// Push potential from pencil to unary.
		pots = copyPairwise(i-1, i);
		its = pencil(i-1, i);
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i-1].first); ++j) {
			OPENGM_ASSERT_OP(j, <, its.second - its.first);
			min = pots(j, 0);
			for (LabelType k = 1; k < gm_.numberOfLabels(sequence_[i].first); ++k)
				min = std::min(min, pots(j, k));
			*(its.first + j) -= min * (2*i - 1) / (2*i);
		}

		// Check whether best pairwise has uniform part of energy.
		#ifndef NDEBUG
		pots = copyPairwise(i-1, i);
		min = pots(0, 0);
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i-1].first); ++j)
			for (LabelType k = 0; k < gm_.numberOfLabels(sequence_[i].first); ++k)
				min = std::min(min, pots(j, k));
		OPENGM_ASSERT(is_almost_equal(energy / (2*sequence_.size()-1), min));
		#endif
	}

	// Check whether LabelCollapse property holds.
	#ifndef NDEBUG
	for (IndexType i = 1; i < sequence_.size(); ++i) {
		marray::Marray<ValueType> left = copyUnary(i-1);
		marray::Marray<ValueType> right = copyUnary(i);
		marray::Marray<ValueType> pairwise = copyPairwise(i-1, i);

		for (LabelType j = 0; j < left.size(); ++j) {
			for (LabelType k = 0; k < right.size(); ++k) {
				OPENGM_ASSERT_OP(left(j), <=, pairwise(j, k) + 1e-8);
				OPENGM_ASSERT_OP(right(k), <=, pairwise(j, k) + 1e-8);
			}
		}
	}
	#endif
}

////////////////////////////////////////////////////////////////////////////////
//
// class CustomReparameterizer
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
class CustomReparameterizer : public SequenceReparameterizer<GM, CustomReparameterizer<GM> > {
public:
	typedef SequenceReparameterizer<GM, CustomReparameterizer<GM> > Parent;

	using typename Parent::GraphicalModelType;
	using typename Parent::IndexType;
	using typename Parent::LabelType;
	using typename Parent::ValueType;

	using typename Parent::StorageType;
	using typename Parent::ElementType;
	using typename Parent::SequenceType;
	using typename Parent::Potentials;
	using typename Parent::Iterators;

	CustomReparameterizer(const GraphicalModelType&, const SequenceType&);

protected:
	using Parent::copyUnary;
	using Parent::copyPairwise;
	using Parent::pencil;

	using Parent::gm_;
	using Parent::sequence_;
	using Parent::repa_;

	void backwardPass();

	friend Parent;

private:
	ValueType attraction_;
};

template<class GM>
CustomReparameterizer<GM>::CustomReparameterizer
(
	const GraphicalModelType &gm,
	const SequenceType &sequence
)
: Parent(gm, sequence)
{
	attraction_ = 0;
	for (size_t i = 0; i < sequence_.size(); ++i) {
		attraction_ += 1.0 / sequence_[i].second;
	}
	attraction_ *= 10.0;
}

template<class GM>
void
CustomReparameterizer<GM>::backwardPass()
{
	#ifndef NDEBUG
	std::cout << "CustomReparameterizer::backwardPass()" << std::endl;
	#endif
	Potentials pots;
	Iterators its;

	// Calculate energy (minimal marginal of last node). The energy is only
	// used for verifying the reparameterization in debug mode.
	#ifndef NDEBUG
	pots = copyUnary(sequence_.size()-1); 
	ValueType energy = pots(0);
	for (LabelType i = 1; i < gm_.numberOfLabels(sequence_[sequence_.size()-1].first); ++i)
		energy = std::min(energy, pots(i));

	ValueType fractionUnary = energy / ((2*sequence_.size()-1) * attraction_);
	ValueType fractionPairwise = (energy - fractionUnary * sequence_.size()) / (sequence_.size() - 1);
	#endif

	// Backward move
	for (IndexType i = sequence_.size()-1; i > 0; --i) {
		ValueType min;

		// Push potential from unary to pencil.
		pots = copyUnary(i);
		its = pencil(i, i-1);
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i].first); ++j) {
			OPENGM_ASSERT_OP(j, <, its.second - its.first);
			ValueType mul = 1.0 - 1.0 / ((2*i + 1) * attraction_);
			*(its.first + j) += pots(j) * mul;
		}

		// Check whether best unary has uniform part of energy.
		#ifndef NDEBUG
		pots = copyUnary(i);
		min = pots(0);
		for (LabelType j = 1; j < gm_.numberOfLabels(sequence_[i].first); ++j)
			min = std::min(min, pots(j));
		OPENGM_ASSERT(is_almost_equal(fractionUnary, min));
		#endif

		// Push potential from pencil to unary.
		pots = copyPairwise(i-1, i);
		its = pencil(i-1, i);
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i-1].first); ++j) {
			OPENGM_ASSERT_OP(j, <, its.second - its.first);
			min = pots(j, 0);
			for (LabelType k = 1; k < gm_.numberOfLabels(sequence_[i].first); ++k)
				min = std::min(min, pots(j, k));
			ValueType mul = ((2*i - 1) * attraction_) / ((2*i + 1) * attraction_ - 1);
			*(its.first + j) -= min * mul;
		}

		// Check whether best pairwise has uniform part of energy.
		#ifndef NDEBUG
		pots = copyPairwise(i-1, i);
		min = pots(0, 0);
		for (LabelType j = 0; j < gm_.numberOfLabels(sequence_[i-1].first); ++j)
			for (LabelType k = 0; k < gm_.numberOfLabels(sequence_[i].first); ++k)
				min = std::min(min, pots(j, k));
		// OPENGM_ASSERT(is_almost_equal(fractionPairwise, min));
		// FIXME: The above is failing constantly. But all the unaries have the
		// same value and also all the pairwise have the same value.
		//
		// The minimum pairwise value is just not as much as we are expecting.
		// Why not?
		#endif
	}

	// Check whether LabelCollapse property holds.
	#ifndef NDEBUG
	for (IndexType i = 1; i < sequence_.size(); ++i) {
		marray::Marray<ValueType> left = copyUnary(i-1);
		marray::Marray<ValueType> right = copyUnary(i);
		marray::Marray<ValueType> pairwise = copyPairwise(i-1, i);

		for (LabelType j = 0; j < left.size(); ++j) {
			for (LabelType k = 0; k < right.size(); ++k) {
				OPENGM_ASSERT_OP(left(j), <=, pairwise(j, k) + 1e-8);
				OPENGM_ASSERT_OP(right(k), <=, pairwise(j, k) + 1e-8);
			}
		}
	}
	#endif
}

////////////////////////////////////////////////////////////////////////////////
//
// class MinSumDiffusion
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
class MinSumDiffusion {
public:
	typedef GM GraphicalModelType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename GM::ValueType ValueType;

	typedef LPReparametrisationStorage<GM> StorageType;
	typedef marray::Marray<ValueType> Potentials;
	typedef std::pair<ValueType*, ValueType*> Iterators;

	MinSumDiffusion(const GraphicalModelType&);

	void processNode(IndexType);
	void singlePass();
	void run();

	const GraphicalModelType& graphicalModel() const { return gm_; }
	const StorageType& reparameterization() const { return repa_; }

private:
	const GraphicalModelType &gm_;
	StorageType repa_;
};

template<class GM>
MinSumDiffusion<GM>::MinSumDiffusion(
	const GraphicalModelType &gm
)
: gm_(gm)
, repa_(gm)
{
}

template<class GM>
void
MinSumDiffusion<GM>::processNode(
	IndexType var
)
{
	typedef marray::Vector<IndexType> Vec;
	Vec factorIds;
	gm_.numberOfNthOrderFactorsOfVariable(var, 2, factorIds);

	for (IndexType lab = 0; lab < gm_.numberOfLabels(var); ++lab) {
		// Accumulation
		for (typename Vec::const_iterator it = factorIds.begin(); it != factorIds.end(); ++it) {
			IndexType var2 = gm_.secondVariableOfSecondOrderFactor(var, *it);
			Potentials pot = copyPairwiseFromRepa(gm_, var, var2, repa_);
			Iterators its = getPencil(gm_, var, var2, repa_);

			ValueType min = pot(lab, 0);
			for (IndexType lab2 = 0; lab2 < gm_.numberOfLabels(var2); ++lab2) {
				min = std::min(min, pot(lab, lab2));
			}

			*(its.first + lab) -= min;
		}

		// Delta
		Potentials deltas = copyUnaryFromRepa(gm_, var, repa_);
		std::transform(deltas.begin(), deltas.end(), deltas.begin(),
			std::bind2nd(std::divides<ValueType>(), static_cast<ValueType>(factorIds.size() + 1)));

		// Diffusion
		for (typename Vec::const_iterator it = factorIds.begin(); it != factorIds.end(); ++it) {
			IndexType var2 = gm_.secondVariableOfSecondOrderFactor(var, *it);
			Iterators its = getPencil(gm_, var, var2, repa_);

			*(its.first + lab) += deltas(lab);
		}
	}
}

template<class GM>
void
MinSumDiffusion<GM>::singlePass()
{
	for (IndexType var = 0; var < gm_.numberOfVariables(); ++var) {
		processNode(var);
	}
}

template<class GM>
void
MinSumDiffusion<GM>::run()
{
	for (unsigned int i = 0; i < 10000; ++i) {
		singlePass();
	}
}

////////////////////////////////////////////////////////////////////////////////
//
// class Reparameterizer and friends
//
////////////////////////////////////////////////////////////////////////////////

//
// FIXME: THE FOLLOWING IMPLEMENTATION IS JUST A HACK TO GET EVERYTHING WORKING.
//
// Instead of a common interface we just resort to C++ template metaprogramming.
// This is not a very good and clean solution. Its also not very extensible.
//

enum ReparameterizationKind {
	ReparameterizationNone,
	ReparameterizationChainCanonical,
	ReparameterizationChainUniform,
	ReparameterizationChainCustom,
	ReparameterizationDiffusion
};

// Sorry, we need a helper for our hacky wrapper helper...
template<class T, ReparameterizationKind KIND> struct ReparameterizerHelper;
template<template<class> class REPA, class T> typename T::ReparameterizationStorageType chainHelper(T &that);

template<class GM, class ACC, ReparameterizationKind KIND =  ReparameterizationNone>
class Reparameterizer {
public:
	typedef Reparameterizer<GM, ACC, KIND> MyType;
	typedef GM OriginalModelType;
	typedef ACC AccumulationType;
	typedef LPReparametrizer<GM, ACC> ReparameterizerType;
	typedef LPReparametrisationStorage<GM> ReparameterizationStorageType;
	typedef typename ReparameterizerType::ReparametrizedGMType ReparameterizedModelType;
	typedef ReparameterizerHelper<MyType, KIND> HelperType;

	Reparameterizer(const OriginalModelType &gm)
	: gm_(gm)
	, repa_(gm)
	{
	}

	const ReparameterizedModelType& reparameterizedModel() const {
		return rm_;
	}
	
	void reparameterize()
	{
		repa_.Reparametrization() = HelperType::doMagic(*this);
		repa_.getReparametrizedModel(rm_);
	}

private:
	const OriginalModelType &gm_;
	ReparameterizedModelType rm_;
	ReparameterizerType repa_;

	friend HelperType;
	template<template<class> class REPA, class T> friend typename T::ReparameterizationStorageType chainHelper(T &that);
};

template<class T>
struct ReparameterizerHelper<T, ReparameterizationNone> {
	static typename T::ReparameterizationStorageType doMagic(T &that)
	{
		#ifndef NDEBUG
		std::cout << "ReparameterizationNone activated!" << std::endl;
		#endif
		return typename T::ReparameterizationStorageType(that.gm_);
	}
};

template<template<class> class REPA, class T>
typename T::ReparameterizationStorageType chainHelper(T &that)
{
	// FIXME: The whole function is really crappy. (I didn't do it!)
	typedef TRWSi<typename T::OriginalModelType, typename T::AccumulationType> TRWSiType;
	typename TRWSiType::Parameter trwsiParam;
	trwsiParam.maxNumberOfIterations_ = 500;
	trwsiParam.setTreeAgreeMaxStableIter(50);
	trwsiParam.verbose_ = true;
	TRWSiType trwsi(that.gm_, trwsiParam);
	trwsi.infer();

	typedef typename TRWSiType::ReparametrizerType ReparametrizerType;
	typedef typename ReparametrizerType::ReparametrizedGMType TRWSiGMType;
	TRWSiGMType gm;
	boost::scoped_ptr<ReparametrizerType> trwsiReparametrizer(trwsi.getReparametrizer());
	trwsiReparametrizer->reparametrize();
	trwsiReparametrizer->getReparametrizedModel(gm);
	typename T::ReparameterizationStorageType result = trwsiReparametrizer->Reparametrization();

	typedef SequenceGeneratorIterator<typename T::OriginalModelType> SequenceGeneratorIteratorType;
	typename SequenceGeneratorIteratorType::Iterators its = SequenceGeneratorIteratorType::makeIterators(trwsi.getDecompositionStorage());
	trivialMerge(REPA<TRWSiGMType>::reparameterizeAll(gm, its.first, its.second), result);
	return result;
}

template<class T>
struct ReparameterizerHelper<T, ReparameterizationChainCanonical> {
	static typename T::ReparameterizationStorageType doMagic(T &that)
	{
		#ifndef NDEBUG
		std::cout << "ReparameterizationCanonical activated!" << std::endl;
		#endif

		return chainHelper<CanonicalReparameterizer>(that);
	}
};

template<class T>
struct ReparameterizerHelper<T, ReparameterizationChainUniform> {
	static typename T::ReparameterizationStorageType doMagic(T &that)
	{
		#ifndef NDEBUG
		std::cout << "ReparameterizationUniform activated!" << std::endl;
		#endif

		return chainHelper<UniformReparameterizer>(that);
	}
};

template<class T>
struct ReparameterizerHelper<T, ReparameterizationChainCustom> {
	static typename T::ReparameterizationStorageType doMagic(T &that)
	{
		#ifndef NDEBUG
		std::cout << "ReparameterizationCustom activated!" << std::endl;
		#endif

		return chainHelper<CustomReparameterizer>(that);
	}
};

template<class T>
struct ReparameterizerHelper<T, ReparameterizationDiffusion> {
	static typename T::ReparameterizationStorageType doMagic(T &that)
	{
		#ifndef NDEBUG
		std::cout << "ReparameterizationDiffusion activated!" << std::endl;
		#endif

		MinSumDiffusion<typename T::OriginalModelType> diffusion(that.gm_);
		diffusion.run();
		return diffusion.reparameterization();
	}
};

} // namespace labelcollapse
} // namespace opengm

#endif
