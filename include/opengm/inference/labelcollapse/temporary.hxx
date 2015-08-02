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
#ifndef OPENGM_LABELCOLLAPSE_TEMPORARY_HXX
#define OPENGM_LABELCOLLAPSE_TEMPORARY_HXX

#include <opengm/inference/trws/trws_trws.hxx>

namespace opengm {
namespace labelcollapse {

template<class INF>
void
temporaryTheorem2
(
	INF &inf,
	std::vector<typename INF::GraphicalModelType::LabelType> *out = NULL
)
{
	std::cout << "[DBG] BEGIN THEOREM2" << std::endl;
	std::vector<typename INF::LabelType> labeling;
	inf.reparametrizer().getApproximateLabeling(labeling);
	if (out)
		*out = labeling;

	std::cout << "Presolving: ";
	for (size_t i = 0; i < labeling.size(); ++i)
		std::cout << " " << labeling[i];
	std::cout << std::endl;

	inf.populateLabeling(labeling.begin());
	std::cout << "[DBG] END THEOREM2" << std::endl;
}

template<class INF>
void
temporaryTheorem3
(
	INF &inf,
	std::vector<typename INF::GraphicalModelType::LabelType> *out = NULL
)
{
	std::cout << "[DBG] BEGIN THEOREM3" << std::endl;
	std::vector<typename INF::LabelType> labeling;
	temporaryTheorem2(inf, &labeling);
	if (out)
		*out = labeling;

	std::vector<typename INF::LabelType> auxLabeling(labeling.size());
	inf.auxiliaryLabeling(labeling.begin(), auxLabeling.begin());

	typename INF::ValueType epsilon = INF::AccumulationType::template neutral<typename INF::ValueType>();
	const typename INF::ReparametrizedModelType &gm = inf.reparametrizedModel();
	for (typename INF::IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		const typename INF::ReparametrizedModelType::FactorType &factor = gm[i];
		FastSequence<typename INF::LabelType> factorLabeling(factor.numberOfVariables());

		for(typename INF::IndexType j = 0; j < factor.numberOfVariables(); ++j)
			factorLabeling[j] = labeling[factor.variableIndex(j)];

		INF::AccumulationType::iop(factor(factorLabeling.begin()), epsilon);
	}
	std::cout << "epsilon = " << epsilon << std::endl;

	inf.increaseEpsilonTo(epsilon);
	// Populating is not necessary, but this triggers epsilon expansion.
	inf.populateLabeling(labeling.begin());
	std::cout << "[DBG] END THEOREM3" << std::endl;
}

} // namespace labelcollapse
} // namespace opengm

#endif
