//
// File: toulbar2.hxx
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
#ifndef OPENGM_TOULBAR2_HXX
#define OPENGM_TOULBAR2_HXX

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <boost/scoped_ptr.hpp>
#include <toulbar2lib.hpp>

#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/utilities/indexing.hxx"
#include "opengm/utilities/metaprogramming.hxx"


namespace opengm {

template<class GM, class ACC>
class ToulBar2 : public opengm::Inference<GM, ACC>
{
public:
	//
	// Types
	//
	typedef GM GraphicalModelType;
	typedef ACC AccumulationType;
	OPENGM_GM_TYPE_TYPEDEFS;
	typedef long double HighPrecision;

	typedef visitors::EmptyVisitor< ToulBar2<GM, ACC> > EmptyVisitorType;
	typedef visitors::VerboseVisitor< ToulBar2<GM, ACC> > VerboseVisitorType;
	typedef visitors::TimingVisitor< ToulBar2<GM, ACC> > TimingVisitorType;

	struct Parameter {
		Parameter()
		: normalizationFactor(1e6)
		{
		}

		HighPrecision normalizationFactor;

		double getResolution()
		{
			throw std::runtime_error("Not implemented.");
		}

		void setResolution(double resolution)
		{
			HighPrecision nom, denom, expRes;
			expRes = std::pow(10.0l, -static_cast<HighPrecision>(resolution));
			nom    = -std::log(10.0l);
			denom  = std::log(1.0l - expRes);
			normalizationFactor = nom / denom;

			// Assertion checks that resolution can be ensured by underlying
			// Cost data type.
			OPENGM_ASSERT(
				normalizationFactor >
				std::pow(2.0l, INTEGERBITS-1) / std::log10(std::pow(10.0l, -resolution))
			);
		}
	};

	//
	// Methods
	//
	ToulBar2(const GraphicalModelType&, const Parameter& = Parameter());
	std::string name() const { return "ToulBar2"; }
	const GraphicalModelType& graphicalModel() const { return gm_; }

	InferenceTermination infer();
	template<class VISITOR> InferenceTermination infer(VISITOR&);
	InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;
	ValueType bound() const;
	virtual ValueType value() const;

private:
	Cost potentialToCost(const ValueType&) const;

	const GraphicalModelType &gm_;
	const Parameter parameter_;
	const boost::scoped_ptr<WeightedCSPSolver> solver_;
	ValueType offset;
	InferenceTermination result_;
};

void
ToulBar2_initialize()
{
	tb2init();
}

template<class GM, class ACC>
ToulBar2<GM, ACC>::ToulBar2(
	const GraphicalModelType &gm,
	const Parameter &parameter
)
: gm_(gm)
, parameter_(parameter)
, solver_(WeightedCSPSolver::makeWeightedCSPSolver(STORE_SIZE, MAX_COST))
, result_(UNKNOWN)
{
	// WARNING: We need to call this before every invocation of the Toulbar2
	// solver.
	//
	// THIS IS HIGHLY PROBLEMATIC!
	//
	// The caller of our ToulBar2 wrapper is responsible that only on instance
	// is used at the same time!
	//
	// FIXME: This needs to be sorted out as soon as possible.
	ToulBar2_initialize();

	typedef typename GraphicalModelType::FactorType FactorType;
	typedef ShapeWalkerSwitchedOrder<typename FactorType::ShapeIteratorType> Walker;

	WeightedCSP &prob = *solver_->getWCSP();

	for (IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
		// API wants to have named variables... We pass empty string.
		int ii = prob.makeEnumeratedVariable("", 0, gm_.numberOfLabels(i)-1);
		OPENGM_ASSERT_OP(i, ==, ii);
	}

	// FIXME: The current code silently assumes that ACC == opengm::Minimizer.
	// If it is a Maximizier we should flip the sign of the potentials.
	if (std::numeric_limits<ValueType>::has_infinity) {
		offset = std::numeric_limits<ValueType>::infinity();
	} else {
		offset = std::numeric_limits<ValueType>::max();
	}

	for (IndexType i = 0; i < gm_.numberOfFactors(); ++i) {
		const FactorType &factor = gm_[i];
		offset = std::min(offset, factor.min());
	}
	OPENGM_ASSERT_OP(offset, >=, 0);

	for (IndexType i = 0; i < gm_.numberOfFactors(); ++i) {
		const FactorType &factor = gm_[i];
		std::vector<Cost> costs(factor.size());
		std::vector<Cost>::iterator it;
		Walker walker(factor.shapeBegin(), factor.numberOfVariables());
		for (it = costs.begin(); it != costs.end(); ++it, ++walker) {
			*it = potentialToCost(factor(walker.coordinateTuple().begin()));
		}
		
		switch (factor.numberOfVariables()) {
		case 0:
			// We ignore a constant factor. The value will be later
			// added to the final result.
			break;
		case 1:
			prob.postUnary(factor.variableIndex(0), costs);
			break;
		case 2:
			prob.postBinaryConstraint(
				factor.variableIndex(0), factor.variableIndex(1),
				costs
			);
			break;
		case 3:
			prob.postTernaryConstraint(
				factor.variableIndex(0),
				factor.variableIndex(1),
				factor.variableIndex(2),
				costs
			);
			break;
		default:
			// FIXME: The code for the default path is broken.
			throw std::runtime_error("Not implemented.");

			// ToulBar2 API wants node as int, not unsigned int.
			// It also takes a “int *” and not “const int *”.
			// To be safe, we copy all the indices, just in case.
			// TODO: Check if we can throw static_cast + const_cast at this one.
			std::vector<int> variables(factor.numberOfVariables());
			factor.variableIndices(variables.begin());

			int constraint = prob.postNaryConstraintBegin(
				&variables[0],
				factor.numberOfVariables(),
				0 // default cost -> meaningless, overriden later
			);

			typedef ShapeWalker<typename FactorType::ShapeIteratorType> Walker;
			Walker walker(factor.shapeBegin(), factor.numberOfVariables());
			for (IndexType j = 0; j < factor.size(); ++j) {
				// Same thing as above.
				// TODO: Another case for static_cast + const_cast?
				variables.assign(
					walker.coordinateTuple().begin(),
					walker.coordinateTuple().end()
				);

				prob.postNaryConstraintTuple(
					constraint,
					&variables[0],
					factor.numberOfVariables(),
					factor(walker.coordinateTuple().begin())
				);
				++walker;
			}

			prob.postNaryConstraintEnd(constraint);
		}
	}

	// Needs to be called before search.
	prob.sortConstraints();
	prob.histogram();
}

template<class GM, class ACC>
InferenceTermination
ToulBar2<GM, ACC>::infer()
{
	EmptyVisitorType visitor;
	return infer(visitor);
}

template<class GM, class ACC>
template<class VISITOR>
InferenceTermination
ToulBar2<GM, ACC>::infer(
	VISITOR& visitor
)
{
	visitor.begin(*this);
	result_ = solver_->solve() ? NORMAL : UNKNOWN;
	visitor.end(*this);
	return result_;
}

template<class GM, class ACC>
InferenceTermination
ToulBar2<GM, ACC>::arg(
	std::vector<LabelType>& labeling,
	const size_t idx
) const
{
	if (result_ != NORMAL)
		return result_;
	
	if (idx != 1)
		return UNKNOWN;

	const std::vector<int> &solution = solver_->getWCSP()->getSolution();
	OPENGM_ASSERT_OP(solution.size(), ==, gm_.numberOfVariables());

	labeling.resize(gm_.numberOfVariables());
	for (IndexType i = 0; i < gm_.numberOfVariables(); ++i) {
		OPENGM_ASSERT_OP(solution[i], <, gm_.numberOfLabels(i));
		labeling[i] = solution[i];
	}

	return result_;
}

template<class GM, class ACC>
typename ToulBar2<GM, ACC>::ValueType
ToulBar2<GM, ACC>::bound() const
{
	OPENGM_ASSERT_OP(result_, ==, NORMAL);
	return value();
}

template<class GM, class ACC>
typename ToulBar2<GM, ACC>::ValueType
ToulBar2<GM, ACC>::value() const
{
	OPENGM_ASSERT_OP(result_, ==, NORMAL);

	std::vector<LabelType> labeling;
	arg(labeling, 1);

	return gm_.evaluate(labeling);
}

template<class GM, class ACC>
Cost
ToulBar2<GM, ACC>::potentialToCost(
	const ValueType &value
) const
{
	HighPrecision costFloat = static_cast<HighPrecision>(value) - static_cast<HighPrecision>(offset);
	costFloat *= parameter_.normalizationFactor;

	return static_cast<Cost>(costFloat);
}

} // namespace opengm

#endif
