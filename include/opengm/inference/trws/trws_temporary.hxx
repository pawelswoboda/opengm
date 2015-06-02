#pragma once
#ifndef OPENGM_TRWS_TEMPORARY_HXX
#define OPENGM_TRWS_TEMPORARY_HXX

#include "trws_reparametrization.hxx"
#include "trws_subproblemsolver.hxx"

namespace opengm {
namespace hack{

using namespace ::opengm::trws_base;

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
	for (typename GM::LabelType i = 0; i < gm[index].shape(0); ++i) {
		FastSequence<typename GM::LabelType> labeling(1);
		labeling[0] = i;
		OPENGM_ASSERT(is_almost_equal(result(labeling.begin()), repa.getVariableValue(index, i)));
	}
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
	#endif
	OPENGM_ASSERT(
		(f.variableIndex(0) == left && f.variableIndex(1) == right) ||
		(f.variableIndex(1) == left && f.variableIndex(0) == right)
	);

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

template<class GM>
void
trivialMerge
(
	const GM &gm,
	const LPReparametrisationStorage<GM> &input,
	LPReparametrisationStorage<GM> &output
)
{
	LPReparametrisationStorage<GM> result(gm);
	for (typename GM::IndexType fidx = 0; fidx < gm.numberOfFactors(); ++fidx) {
		if (gm[fidx].numberOfVariables() < 2)
			continue;

		for (typename GM::IndexType rel = 0; rel < gm[fidx].numberOfVariables(); ++rel) {
			std::pair<typename GM::ValueType*, typename GM::ValueType*> its1, its2;
			its1 = output.getIterators(fidx, rel);
			its2 = const_cast<LPReparametrisationStorage<GM>&>(input).getIterators(fidx, rel);
			OPENGM_ASSERT_OP(its1.second - its1.first, ==, its2.second - its2.first);

			std::transform(
				its1.first, its1.second,
				its2.first, its1.first,
				std::plus<typename GM::ValueType>()
			);
		}
	}
}


////////////////////////////////////////////////////////////////////////////////
//
// class SequenceGeneratorIterator
//
////////////////////////////////////////////////////////////////////////////////

template<class GM, class GM2>
class SequenceGeneratorIterator {
public:
	typedef GM GraphicalModelType;
	typedef typename GraphicalModelType::IndexType IndexType;
	typedef typename GraphicalModelType::ValueType ValueType;

	typedef DecompositionStorage<GM2> DecompositionStorageType;
	typedef SequenceStorage<GM2> SequenceStorageType;
	typedef std::pair<SequenceGeneratorIterator<GM, GM2>, SequenceGeneratorIterator<GM, GM2> > Iterators;

	typedef std::pair<IndexType, ValueType> Element;
	typedef std::vector<Element> Elements;

	bool operator!=(const SequenceGeneratorIterator&);
	Elements& operator*();
	SequenceGeneratorIterator<GM, GM2>& operator++();

	static Iterators makeIterators(const DecompositionStorageType&);

private:
	SequenceGeneratorIterator(size_t index, const DecompositionStorageType&);

	size_t index_; 
	const DecompositionStorageType *storage_;
	Elements current_;
};

template<class GM, class GM2>
SequenceGeneratorIterator<GM, GM2>::SequenceGeneratorIterator
(
	size_t index,
	const DecompositionStorageType &storage
)
: index_(index)
, storage_(&storage)
{
}

template<class GM, class GM2>
bool
SequenceGeneratorIterator<GM, GM2>::operator!=(const SequenceGeneratorIterator &rhs)
{
	return (index_ != rhs.index_) || (storage_ != rhs.storage_);
}

template<class GM, class GM2>
typename SequenceGeneratorIterator<GM, GM2>::Elements&
SequenceGeneratorIterator<GM, GM2>::operator*()
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

template<class GM, class GM2>
SequenceGeneratorIterator<GM, GM2>&
SequenceGeneratorIterator<GM, GM2>::operator++()
{
	++index_;
	return *this;
}

template<class GM, class GM2>
typename SequenceGeneratorIterator<GM, GM2>::Iterators
SequenceGeneratorIterator<GM, GM2>::makeIterators
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
	std::cout << "SequenceReparameterizer::run()" << std::endl;
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
	std::cout << "SequenceReparameterizer::forwardPass()" << std::endl;
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
	std::cout << "SequenceReparameterizer::backwardPass()" << std::endl;
}

template<class GM, class DERIVED>
template<class INPUT_ITERATOR>
typename SequenceReparameterizer<GM, DERIVED>::StorageType
SequenceReparameterizer<GM, DERIVED>::reparametrizeAll
(
	const GM &gm,
	INPUT_ITERATOR begin,
	INPUT_ITERATOR end
)
{
	std::cout << "SequenceReparameterizer::reparametrizeAll()" << std::endl;
	StorageType repa_(gm);
	for (INPUT_ITERATOR it = begin; it != end; ++it) {
		DERIVED reparametrizer(gm, *it);
		trivialMerge(gm, reparametrizer.run(), repa_);
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

	std::cout << "-- BEGIN REPARAMETRIZATION --" << std::endl;
	repa_.PrintTestData(std::cout);
	std::cout << "-- END REPARAMETRIZATION --" << std::endl;
	#endif

	return repa_;
}


////////////////////////////////////////////////////////////////////////////////
//
// class CanonicalReparametrizer
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
class CanonicalReparametrizer : public SequenceReparameterizer<GM, CanonicalReparametrizer<GM> > {
public:
	typedef SequenceReparameterizer<GM, CanonicalReparametrizer<GM> > Parent;

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
	using Parent::repa_;

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
	std::cout << "CanonicalReparametrizer::backwardPass()" << std::endl;
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
// class UniformReparametrizer
//
////////////////////////////////////////////////////////////////////////////////

template<class GM>
class UniformReparametrizer : public SequenceReparameterizer<GM, UniformReparametrizer<GM> > {
public:
	typedef SequenceReparameterizer<GM, UniformReparametrizer<GM> > Parent;

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
	using Parent::repa_;

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
	std::cout << "UniformReparametrizer::backwardPass()" << std::endl;
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

} // namespace hack
} // namepace opengm
#endif
