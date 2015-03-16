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

#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/graphicalmodel/space/discretespace.hxx"
#include "opengm/functions/function_properties_base.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
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


// A (potentially partial) mapping of orignal labels to auxiliary labels
// (and vice versa) for a given variable.
template<class GM>
class Mapping;

// A view function which returns the values from the original model if the
// nodes are not collapsed. If they are, the view function will return the
// corresponding epsilon value.
template<class GM>
class EpsilonFunction;

// This functor operates on factor function values. It will determine the new
// best (smallest for energy minimization) epsilon value which is worse
// (higher) than the old epsilon value.
template<class ACC, class VALUE_TYPE>
class EpsilonFunctor;

// This functor operates on a factor. It unwraps the underlying factor function
// and calls the NonCollapsedFunctionFunctor on this function.
//
// We need this intermediate step to get C++ template inference working.
template<class FUNCTOR>
class NonCollapsedFactorFunctor;

// This functor operators on factor function values and is a coordinate functor
// (receives value and coordinate input iterator as arguments).
//
// This functor will write all the labels to the output iterator where the
// value is less than or equal to their variable’s epsilon value.
template<class ACC, class INDEX_TYPE, class VALUE_TYPE, class OUTPUT_ITERATOR>
class NonCollapsedFunctionFunctor;

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
		const AuxiliaryModelType gm = builder_.getAuxiliaryModel();

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

		// Update the model. This will try to make more labels available where
		// the current labeling is invalid.
		builder_.uncollapseLabeling(labeling.begin());

		if (visitor(*this) != visitors::VisitorReturnFlag::ContinueInf)
			exitInf = true;
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
	typedef Mapping<GM> MappingType;

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
	void updateMappings();
	ValueType calculateNewEpsilon(const IndexType);

	const OriginalModelType &original_;

	std::vector<ValueType> epsilons_;
	std::vector<MappingType> mappings_;

	bool rebuildNecessary_;
	AuxiliaryModelType auxiliary_;
};

template<class GM, class ACC>
ModelBuilder<GM, ACC>::ModelBuilder
(
	const OriginalModelType &gm
)
: original_(gm)
, mappings_(gm.numberOfVariables(), MappingType(0))
, rebuildNecessary_(true)
{
	reset();
}

template<class GM, class ACC>
void
ModelBuilder<GM, ACC>::buildAuxiliaryModel()
{
	OPENGM_ASSERT(rebuildNecessary_)
	if (!rebuildNecessary_)
		return;

	// TODO: We probably should update the mapping incrementally. Everything
	// gets faster :-)
	updateMappings();

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
		const ViewFunction func(factor, mappings_, epsilons_[i]);

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
		original[i] = mappings_[i].original(auxiliary[i]);
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
		if ((*it == 0) && (!mappings_[i].full()))
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
		if ((*it == 0) && (!mappings_[i].full())) {
			// Zero is the label for “collapsed” labels. If it is selected during
			// inference we need to uncollapse one label now.
			uncollapse(i);
		}
	}
}

template<class GM, class ACC>
void
ModelBuilder<GM, ACC>::uncollapse
(
	const IndexType idx
)
{
	bool foundAnyFactor = false;
	IndexType bestFactor = 0;
	ValueType bestEpsilon = ACC::template neutral<ValueType>();
	ValueType bestEpsilonDiff = ACC::template neutral<ValueType>();

	for (IndexType f = 0; f < original_.numberOfFactors(idx); ++f) {
		IndexType factor = original_.factorOfVariable(idx, f);

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
typename ModelBuilder<GM, ACC>::ValueType
ModelBuilder<GM, ACC>::calculateNewEpsilon(
	const IndexType idx
)
{
	const ValueType &epsilon = epsilons_[idx];
	const typename OriginalModelType::FactorType &factor = original_[idx];

	EpsilonFunctor<ACC, ValueType> functor(epsilon);
	factor.forAllValuesInAnyOrder(functor);
	return functor.value();
}

template<class GM, class ACC>
void
ModelBuilder<GM, ACC>::updateMappings()
{
	typedef std::vector<LabelType> LabelVec;
	typedef typename LabelVec::iterator Iterator;
	typedef std::back_insert_iterator<LabelVec> Inserter;
	typedef NonCollapsedFunctionFunctor<ACC, IndexType, ValueType, Inserter> FunctionFunctor;
	typedef NonCollapsedFactorFunctor<FunctionFunctor> FactorFunctor;

	for (IndexType i = 0; i < original_.numberOfVariables(); ++i) {
		bool foundAnyFactor = false;
		LabelVec nonCollapsed;
		Inserter inserter(nonCollapsed);

		for (IndexType j = 0; j < original_.numberOfFactors(i); ++j) {
			const IndexType f = original_.factorOfVariable(i, j);
			const typename OriginalModelType::FactorType &factor = original_[f];

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
		m = MappingType(original_.numberOfLabels(i));

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

template<class GM, class ACC>
void
ModelBuilder<GM, ACC>::reset()
{
	epsilons_.assign(original_.numberOfFactors(), ACC::template ineutral<ValueType>());
	rebuildNecessary_ = true;

	for (IndexType f = 0; f < original_.numberOfFactors(); ++f)
		epsilons_[f] = calculateNewEpsilon(f);

	for (IndexType i = 0; i < original_.numberOfVariables(); ++i)
		uncollapse(i);
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
		if (v > oldValue_ && v < value_) {
			value_ = v;
		}
	}

private:
	VALUE_TYPE oldValue_;
	VALUE_TYPE value_;
};

////////////////////////////////////////////////////////////////////////////////
//
// class NonCollapsedFactorFunctor and NonCollapsedFunctionFunctor
//
////////////////////////////////////////////////////////////////////////////////

template<class FUNCTOR>
class NonCollapsedFactorFunctor {
public:
	NonCollapsedFactorFunctor(FUNCTOR &functor)
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
	typedef Mapping<GM> MappingType;
	typedef typename GM::FactorType FactorType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename GM::ValueType ValueType;

	EpsilonFunction(
		const FactorType &factor,
		const std::vector<MappingType> &mappings,
		ValueType epsilon
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
		LabelType origLabel;

		if ((*mappings_)[varIdx].full()) {
			origLabel = auxLabel;
		} else {
			if (auxLabel == 0) {
				return epsilon_;
			}

			origLabel = (*mappings_)[varIdx].original(auxLabel);
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
