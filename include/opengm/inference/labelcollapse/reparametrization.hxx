//
// File: reparametrization.hxx
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
#ifndef OPENGM_LABELCOLLAPSE_REPARAMETRIZATION_HXX
#define OPENGM_LABELCOLLAPSE_REPARAMETRIZATION_HXX

#include <boost/scoped_ptr.hpp>

#include <opengm/inference/auxiliary/lp_reparametrization.hxx>
#include <opengm/inference/trws/trws_base.hxx>
#include <opengm/inference/trws/trws_decomposition.hxx>
#include <opengm/inference/trws/trws_reparametrization.hxx>
#include <opengm/inference/trws/trws_trws.hxx>

////////////////////////////////////////////////////////////////////////////////
//
// FIXME: THE WHOLE REPARAMETRIZATION MACHINERY IN **THIS** HEADER FILE
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
// class LabelCollapsePropertyChecker
//
////////////////////////////////////////////////////////////////////////////////

class LabelCollapsePropertyChecker {
public:
	LabelCollapsePropertyChecker()
	: result_(false)
	, countAll_(0)
	, countLC_(0)
	{
	}

	template<class GM>
	bool operator()(const GM &gm, const LPReparametrisationStorage<GM> &repa)
	{
		result_ = false;
		countAll_ = 0;
		countLC_ = 0;

		for (typename GM::IndexType i = 0; i < gm.numberOfFactors(); ++i) {
			OPENGM_ASSERT_OP(gm[i].numberOfVariables(), <=, 2);
			if (gm[i].numberOfVariables() == 2) {
				const typename GM::FactorType &factor = gm[i];
				typename GM::IndexType left = factor.variableIndex(0);
				typename GM::IndexType right = factor.variableIndex(1);

				marray::Marray<typename GM::ValueType> lpot = copyUnaryFromRepa(gm, left, repa);
				marray::Marray<typename GM::ValueType> rpot = copyUnaryFromRepa(gm, right, repa);
				marray::Marray<typename GM::ValueType> ppot = copyPairwiseFromRepa(gm, left, right, repa);

				ShapeWalker<typename GM::FactorType::ShapeIteratorType> walker(factor.shapeBegin(), factor.numberOfVariables());
				for (typename GM::IndexType j = 0; j < factor.size(); ++j, ++walker) {
					++countAll_;

					if (lpot(walker.coordinateTuple()[0]) <= (ppot(walker.coordinateTuple().begin()) + 1e-8) &&
					    rpot(walker.coordinateTuple()[1]) <= (ppot(walker.coordinateTuple().begin()) + 1e-8))
					{
						++countLC_;
					}
				}
			}
		}

		result_ = countAll_ == countLC_;
		return result_;
	}

	template<class GM>
	bool operator()(const GM &gm)
	{
		LPReparametrisationStorage<GM> repa(gm);
		(*this)(gm, repa);
	}

	bool result() { return result_; }
	size_t countAll() { return countAll_; }
	size_t countLC() { return countLC_; }

	std::string str() const
	{
		std::stringstream ss;
		ss << "LC property: " << countLC_ << " / " << countAll_
		   << " (" << (countLC_ * 100.0 / countAll_) << "%)";
		return ss.str();
	}

private:
	bool result_;
	size_t countAll_;
	size_t countLC_;
};

////////////////////////////////////////////////////////////////////////////////
//
// class SequenceReparametrizerWrapper
//
////////////////////////////////////////////////////////////////////////////////

template<class GM, class ACC, template<class> class SEQ_REPA>
class SequenceReparametrizerWrapper : public LPReparametrizer<GM, ACC> {
public:
	typedef LPReparametrizer<GM, ACC> Parent;

	using typename Parent::GraphicalModelType;
	using typename Parent::IndexType;
	using typename Parent::LabelType;
	using typename Parent::ValueType;
	using typename Parent::MaskType;
	using typename Parent::RepaStorageType;

	typedef TRWSi<GM, ACC> TRWSiType;
	typedef typename TRWSiType::ReparametrizerType ReparametrizerType;

	typedef trws_base::SequenceStorage<GM> SequenceStorageType;
	typedef std::pair<IndexType, ValueType> Element;
	typedef std::vector<Element> Elements;

	SequenceReparametrizerWrapper(const GraphicalModelType &gm);
	virtual void reparametrize(const MaskType* = NULL);
	void getApproximateLabeling(std::vector<LabelType> &labeling);

private:
	static typename TRWSiType::Parameter trwsiParameter();

	using Parent::_gm;
	using Parent::_repastorage;

	TRWSiType trwsi_;
	boost::scoped_ptr<ReparametrizerType> trwsiRepa_;
};

template<class GM, class ACC, template<class> class SEQ_REPA>
typename SequenceReparametrizerWrapper<GM, ACC, SEQ_REPA>::TRWSiType::Parameter
SequenceReparametrizerWrapper<GM, ACC, SEQ_REPA>::trwsiParameter()
{
	typename TRWSiType::Parameter result;
	result.maxNumberOfIterations_ = 1000;
	result.setTreeAgreeMaxStableIter(100);
	result.precision_ = 0;
	result.verbose_ = true;
	return result;
}

template<class GM, class ACC, template<class> class SEQ_REPA>
SequenceReparametrizerWrapper<GM, ACC, SEQ_REPA>::SequenceReparametrizerWrapper
(
	const GraphicalModelType &gm
)
: Parent(gm)
, trwsi_(gm, trwsiParameter())
{
}

template<class GM, class ACC, template<class> class SEQ_REPA>
void
SequenceReparametrizerWrapper<GM, ACC, SEQ_REPA>::getApproximateLabeling
(
	std::vector<LabelType> &labeling
)
{
	trwsi_.arg(labeling);
}


template<class GM, class ACC, template<class> class SEQ_REPA>
void
SequenceReparametrizerWrapper<GM, ACC, SEQ_REPA>::reparametrize
(
	const MaskType *mask
)
{
	if (mask != NULL) {
		throw std::runtime_error("SequenceReparametrizerWrapper has no support for mask.");
	}

	typedef typename ReparametrizerType::ReparametrizedGMType SubModelType;
	SubModelType rgm;

	_repastorage = RepaStorageType(_gm); // clear old repa

	trwsi_.infer();
	trwsiRepa_.reset(trwsi_.getReparametrizer());
	trwsiRepa_->getReparametrizedModel(rgm);

	typename TRWSiType::Storage &sto = trwsi_.getDecompositionStorage();
	for (size_t i = 0; i < sto.numberOfModels(); ++i) {
		const SequenceStorageType &seq = sto.subModel(i);
		Elements current(seq.size());
		for (IndexType i = 0; i < seq.size(); ++i) {
			current[i].first = seq.varIndex(i);
			current[i].second = static_cast<ValueType>(1.0)
			                  / sto.getSubVariableList(current[i].first).size();
		}

		SEQ_REPA<SubModelType> reparametrizer(rgm, current);
		trivialMerge(reparametrizer.run(), _repastorage);
	}
}

////////////////////////////////////////////////////////////////////////////////
//
// class SequenceReparametrizer
//
////////////////////////////////////////////////////////////////////////////////

// Uses CRTP pattern for static polymorphism.
template<class GM, class DERIVED>
class SequenceReparametrizer {
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

	SequenceReparametrizer(const GraphicalModelType&, const SequenceType&);
	StorageType run();

	template<class INPUT_ITERATOR>
	static StorageType reparametrizeAll(const GM &gm, INPUT_ITERATOR begin, INPUT_ITERATOR end);

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
SequenceReparametrizer<GM, DERIVED>::SequenceReparametrizer
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
typename SequenceReparametrizer<GM, DERIVED>::StorageType
SequenceReparametrizer<GM, DERIVED>::run()
{
	#ifndef NDEBUG
	std::cout << "SequenceReparametrizer::run()" << std::endl;
	#endif
	repa_ = StorageType(gm_);

	if (sequence_.size() >= 2) {
		static_cast<DERIVED*>(this)->forwardPass();
		static_cast<DERIVED*>(this)->backwardPass();
	}

	return repa_;
}

template<class GM, class DERIVED>
typename SequenceReparametrizer<GM, DERIVED>::Potentials
SequenceReparametrizer<GM, DERIVED>::copyUnary
(
	IndexType idx
) const
{
	OPENGM_ASSERT_OP(idx, >=, 0);
	OPENGM_ASSERT_OP(idx, <, sequence_.size());
	return copyUnaryFromRepaWeighted(gm_, sequence_[idx].first, repa_, sequence_[idx].second);
}

template<class GM, class DERIVED>
typename SequenceReparametrizer<GM, DERIVED>::Potentials
SequenceReparametrizer<GM, DERIVED>::copyPairwise
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
typename SequenceReparametrizer<GM, DERIVED>::Iterators
SequenceReparametrizer<GM, DERIVED>::pencil
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
SequenceReparametrizer<GM, DERIVED>::forwardPass()
{
	#ifndef NDEBUG
	std::cout << "SequenceReparametrizer::forwardPass()" << std::endl;
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
SequenceReparametrizer<GM, DERIVED>::backwardPass()
{
	#ifndef NDEBUG
	std::cout << "SequenceReparametrizer::backwardPass()" << std::endl;
	#endif
}

////////////////////////////////////////////////////////////////////////////////
//
// class CanonicalReparametrizer
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
class CanonicalReparametrizer : public SequenceReparametrizer<GM, CanonicalReparametrizer<GM> > {
public:
	typedef SequenceReparametrizer<GM, CanonicalReparametrizer<GM> > Parent;

	using typename Parent::GraphicalModelType;
	using typename Parent::IndexType;
	using typename Parent::LabelType;
	using typename Parent::ValueType;

	using typename Parent::StorageType;
	using typename Parent::ElementType;
	using typename Parent::SequenceType;
	using typename Parent::Potentials;
	using typename Parent::Iterators;

	CanonicalReparametrizer(const GraphicalModelType&, const SequenceType&);

protected:
	using Parent::copyUnary;
	using Parent::copyPairwise;
	using Parent::pencil;

	using Parent::gm_;
	using Parent::sequence_;

	void backwardPass();

	friend Parent;
};

template<class GM>
CanonicalReparametrizer<GM>::CanonicalReparametrizer
(
	const GraphicalModelType &gm,
	const SequenceType &sequence
)
: Parent(gm, sequence)
{
}

template<class GM>
void
CanonicalReparametrizer<GM>::backwardPass()
{
	#ifndef NDEBUG
	std::cout << "CanonicalReparametrizer::backwardPass()" << std::endl;
	#endif
	Potentials pots;
	Iterators its;

	// Calculate energy (minimal marginal of last node). The energy is only
	// used for verifying the reparametrization in debug mode.
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
// class UniformReparametrizer
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
class UniformReparametrizer : public SequenceReparametrizer<GM, UniformReparametrizer<GM> > {
public:
	typedef SequenceReparametrizer<GM, UniformReparametrizer<GM> > Parent;

	using typename Parent::GraphicalModelType;
	using typename Parent::IndexType;
	using typename Parent::LabelType;
	using typename Parent::ValueType;

	using typename Parent::StorageType;
	using typename Parent::ElementType;
	using typename Parent::SequenceType;
	using typename Parent::Potentials;
	using typename Parent::Iterators;

	UniformReparametrizer(const GraphicalModelType&, const SequenceType&);

protected:
	using Parent::copyUnary;
	using Parent::copyPairwise;
	using Parent::pencil;

	using Parent::gm_;
	using Parent::sequence_;

	void backwardPass();

	friend Parent;
};

template<class GM>
UniformReparametrizer<GM>::UniformReparametrizer
(
	const GraphicalModelType &gm,
	const SequenceType &sequence
)
: Parent(gm, sequence)
{
}

template<class GM>
void
UniformReparametrizer<GM>::backwardPass()
{
	#ifndef NDEBUG
	std::cout << "UniformReparametrizer::backwardPass()" << std::endl;
	#endif
	Potentials pots;
	Iterators its;

	// Calculate energy (minimal marginal of last node). The energy is only
	// used for verifying the reparametrization in debug mode.
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
// class CustomReparametrizer
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
class CustomReparametrizer : public SequenceReparametrizer<GM, CustomReparametrizer<GM> > {
public:
	typedef SequenceReparametrizer<GM, CustomReparametrizer<GM> > Parent;

	using typename Parent::GraphicalModelType;
	using typename Parent::IndexType;
	using typename Parent::LabelType;
	using typename Parent::ValueType;

	using typename Parent::StorageType;
	using typename Parent::ElementType;
	using typename Parent::SequenceType;
	using typename Parent::Potentials;
	using typename Parent::Iterators;

	CustomReparametrizer(const GraphicalModelType&, const SequenceType&);

protected:
	using Parent::copyUnary;
	using Parent::copyPairwise;
	using Parent::pencil;

	using Parent::gm_;
	using Parent::sequence_;

	void backwardPass();

	friend Parent;

private:
	ValueType attraction_;
};

template<class GM>
CustomReparametrizer<GM>::CustomReparametrizer
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
CustomReparametrizer<GM>::backwardPass()
{
	#ifndef NDEBUG
	std::cout << "CustomReparametrizer::backwardPass()" << std::endl;
	#endif
	Potentials pots;
	Iterators its;

	// Calculate energy (minimal marginal of last node). The energy is only
	// used for verifying the reparametrization in debug mode.
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

template<class GM, class ACC>
class MinSumDiffusion : public LPReparametrizer<GM, ACC> {
public:
	typedef LPReparametrizer<GM, ACC> Parent;

	using typename Parent::GraphicalModelType;
	using typename Parent::IndexType;
	using typename Parent::LabelType;
	using typename Parent::ValueType;
	using typename Parent::RepaStorageType;
	using typename Parent::MaskType;

	typedef marray::Marray<ValueType> Potentials;
	typedef std::pair<ValueType*, ValueType*> Iterators;

	MinSumDiffusion(const GraphicalModelType&);

	void processNode(IndexType);
	void singlePass();
	virtual void reparametrize(const MaskType* = NULL);
	ValueType dualObjective() const;

private:
	using Parent::_gm;
	using Parent::_repastorage;
};

template<class GM, class ACC>
MinSumDiffusion<GM, ACC>::MinSumDiffusion
(
	const GraphicalModelType &gm
)
: Parent(gm)
{
}

template<class GM, class ACC>
void
MinSumDiffusion<GM, ACC>::processNode
(
	IndexType var
)
{
	typedef marray::Vector<IndexType> Vec;
	Vec factorIds;
	size_t count = _gm.numberOfNthOrderFactorsOfVariable(var, 2, factorIds);

	// BUG? If the vector does not contain any elements, constructing iterators
	// will violate some assertions. This seems strange...
	if (count == 0)
		return;

	for (IndexType lab = 0; lab < _gm.numberOfLabels(var); ++lab) {
		// Accumulation
		for (typename Vec::const_iterator it = factorIds.begin(); it != factorIds.end(); ++it) {
			IndexType var2 = _gm.secondVariableOfSecondOrderFactor(var, *it);
			Potentials pot = copyPairwiseFromRepa(_gm, var, var2, _repastorage);
			Iterators its = getPencil(_gm, var, var2, _repastorage);

			ValueType min = pot(lab, 0);
			for (IndexType lab2 = 0; lab2 < _gm.numberOfLabels(var2); ++lab2) {
				min = std::min(min, pot(lab, lab2));
			}

			*(its.first + lab) -= min;
		}

		// Delta
		Potentials deltas = copyUnaryFromRepa(_gm, var, _repastorage);
		std::transform(deltas.begin(), deltas.end(), deltas.begin(),
			std::bind2nd(std::divides<ValueType>(), static_cast<ValueType>(factorIds.size() + 1)));

		// Diffusion
		for (typename Vec::const_iterator it = factorIds.begin(); it != factorIds.end(); ++it) {
			IndexType var2 = _gm.secondVariableOfSecondOrderFactor(var, *it);
			Iterators its = getPencil(_gm, var, var2, _repastorage);

			*(its.first + lab) += deltas(lab);
		}
	}
}

template<class GM, class ACC>
void
MinSumDiffusion<GM, ACC>::singlePass()
{
	for (IndexType var = 0; var < _gm.numberOfVariables(); ++var) {
		processNode(var);
	}
}

template<class GM, class ACC>
void
MinSumDiffusion<GM, ACC>::reparametrize
(
	const MaskType *mask
)
{
	if (mask != NULL) {
		throw std::runtime_error("MinSumDiffusion has no support for mask.");
	}

	_repastorage = RepaStorageType(_gm);

	unsigned int i = 0;
	unsigned int noProgress = 0;
	ValueType dual = dualObjective();

	while (noProgress < 100) {
		++i;
		std::cout << "MinSum-Diffusion: iteration " << i << " | ";

		singlePass();
		ValueType newObjective = dualObjective();

		if (newObjective > dual + 1e-10)
			noProgress = 0;
		else
			++noProgress;

		dual = newObjective;
		std::cout << "dual: " << dual << " | ";

		LabelCollapsePropertyChecker checker;
		checker(_gm, _repastorage);
		std::cout << checker.str() << std::endl;
	}
}

template<class GM, class ACC>
typename MinSumDiffusion<GM, ACC>::ValueType
MinSumDiffusion<GM, ACC>::dualObjective() const
{
	ValueType dual = 0;
	for (IndexType i = 0; i < _gm.numberOfFactors(); ++i) {
		// TODO: This is kind of slow-ish.
		marray::Marray<ValueType> result(_gm[i].shapeBegin(), _gm[i].shapeEnd());
		_repastorage.copyFactorValues(i, result.begin());

		ValueType min = result(0);
		for (size_t j = 0; j < result.size(); ++j)
			min = std::min(min, result(j));

		dual += min;
	}
	return dual;
}

////////////////////////////////////////////////////////////////////////////////
//
// class Reparametrizer and friends
//
////////////////////////////////////////////////////////////////////////////////

enum ReparametrizationKind {
	ReparametrizationNone,
	ReparametrizationTRWS,
	ReparametrizationChainCanonical,
	ReparametrizationChainUniform,
	ReparametrizationChainCustom,
	ReparametrizationDiffusion
};

template<class GM, class ACC, ReparametrizationKind KIND>
struct ReparametrizerTypeGen;

template<class GM, class ACC>
struct ReparametrizerTypeGen<GM, ACC, ReparametrizationNone> {
	typedef LPReparametrizer<GM, ACC> Type;
};

template<class GM, class ACC>
struct ReparametrizerTypeGen<GM, ACC, ReparametrizationChainCanonical> {
	typedef SequenceReparametrizerWrapper<GM, ACC, CanonicalReparametrizer> Type;
};

template<class GM, class ACC>
struct ReparametrizerTypeGen<GM, ACC, ReparametrizationChainUniform> {
	typedef SequenceReparametrizerWrapper<GM, ACC, UniformReparametrizer> Type;
};

template<class GM, class ACC>
struct ReparametrizerTypeGen<GM, ACC, ReparametrizationChainCustom> {
	typedef SequenceReparametrizerWrapper<GM, ACC, CustomReparametrizer> Type;
};

template<class GM, class ACC>
struct ReparametrizerTypeGen<GM, ACC, ReparametrizationDiffusion> {
	typedef MinSumDiffusion<GM, ACC> Type;
};

} // namespace labelcollapse
} // namespace opengm

#endif
