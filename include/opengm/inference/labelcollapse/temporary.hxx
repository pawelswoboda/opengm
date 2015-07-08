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
	typedef TRWSi<typename INF::ReparameterizedModelType, typename INF::AccumulationType> TRWSiType;
	typename TRWSiType::Parameter param;
	param.maxNumberOfIterations_ = 300;
	param.setTreeAgreeMaxStableIter(50);
	param.verbose_ = true;

	TRWSiType trwsi(inf.reparameterizedModel(), param);
	trwsi.infer();

	std::vector<typename INF::LabelType> labeling;
	trwsi.arg(labeling);
	if (out)
		*out = labeling;

	inf.populateLabeling(labeling.begin());

}

template<class INF>
void
temporaryTheorem3
(
	INF &inf,
	std::vector<typename INF::GraphicalModelType::LabelType> *out = NULL
)
{
	std::vector<typename INF::LabelType> labeling;
	temporaryTheorem2(inf, &labeling);
	if (out)
		*out = labeling;

	std::vector<typename INF::LabelType> auxLabeling(labeling.size());
	inf.auxiliaryLabeling(labeling.begin(), auxLabeling.begin());

	typename INF::ValueType epsilon = INF::AccumulationType::template neutral<typename INF::ValueType>();
	const typename INF::ReparameterizedModelType &gm = inf.reparameterizedModel();
	for (typename INF::IndexType i = 0; i < gm.numberOfFactors(); ++i) {
		const typename INF::ReparameterizedModelType::FactorType &factor = gm[i];
		FastSequence<typename INF::LabelType> factorLabeling(factor.numberOfVariables());

		for(typename INF::IndexType j = 0; j < factor.numberOfVariables(); ++j)
			factorLabeling[j] = labeling[factor.variableIndex(j)];

		INF::AccumulationType::iop(factor(factorLabeling.begin()), epsilon);
	}

	inf.increaseEpsilonTo(epsilon);
	// Populating is not necessary, but this triggers epsilon expansion.
	inf.populateLabeling(labeling.begin());
}

} // namespace labelcollapse
} // namespace opengm

#endif
