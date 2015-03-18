//
// File: labelcollapse_visitor.hxx
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
#ifndef OPENGM_LABELCOLLAPSE_VISITOR_HXX
#define OPENGM_LABELCOLLAPSE_VISITOR_HXX

#include <iostream>
#include <vector>

namespace opengm {
namespace visitors {

template<class INFERENCE>
class LabelCollapseStatisticsVisitor {
public:
	typedef typename INFERENCE::IndexType IndexType;
	typedef typename INFERENCE::LabelType LabelType;
	typedef typename INFERENCE::ValueType ValueType;

	LabelCollapseStatisticsVisitor(bool verbose = true);
	void begin(const INFERENCE&);
	void end(const INFERENCE&);
	size_t operator()(const INFERENCE&);
	void addLog(const std::string&) const;
	void log(const std::string&, double) const;

	LabelType labels() const;
	LabelType labelsReduced() const;
	LabelType space() const;
	LabelType spaceReduced() const;

private:
	void update(const INFERENCE&);
	void verbose(const INFERENCE&, const std::string&) const;

	bool verbose_;
	unsigned int iterations_;
	std::vector<LabelType> origNumberOfLabels_;
	std::vector<LabelType> auxNumberOfLabels_;
};

template<class INFERENCE>
LabelCollapseStatisticsVisitor<INFERENCE>::LabelCollapseStatisticsVisitor
(
	bool verbose
)
: verbose_(verbose)
{
}

template<class INFERENCE>
void
LabelCollapseStatisticsVisitor<INFERENCE>::update
(
	const INFERENCE &inf
)
{
	inf.originalNumberOfLabels(origNumberOfLabels_.begin());
	inf.currentNumberOfLabels(auxNumberOfLabels_.begin());
}

template<class INFERENCE>
void
LabelCollapseStatisticsVisitor<INFERENCE>::verbose
(
	const INFERENCE &inf,
	const std::string &logName
) const
{
	if (! verbose_)
		return;

	std::cout << logName;
	std::cout << "value " << inf.value();
	std::cout << " bound " << inf.bound();
	std::cout << " labels " << labels() << " (- " << labelsReduced();
	std::cout << ") space " << space() << " (- " << spaceReduced();
	std::cout << ")" << std::endl;
}

template<class INFERENCE>
void
LabelCollapseStatisticsVisitor<INFERENCE>::begin
(
	const INFERENCE &inf
)
{

	iterations_ = 0;

	IndexType numberOfVariables = inf.graphicalModel().numberOfVariables();
	origNumberOfLabels_.resize(numberOfVariables);
	auxNumberOfLabels_.resize(numberOfVariables);
	update(inf);
	verbose(inf, "begin: ");
}

template<class INFERENCE>
size_t
LabelCollapseStatisticsVisitor<INFERENCE>::operator()
(
	const INFERENCE &inf
)
{
	++iterations_;
	update(inf);
	verbose(inf, "step: ");
	return VisitorReturnFlag::ContinueInf;
}

template<class INFERENCE>
void
LabelCollapseStatisticsVisitor<INFERENCE>::end
(
	const INFERENCE &inf
)
{
	update(inf);
	verbose(inf, "end: ");
}

template<class INFERENCE>
void
LabelCollapseStatisticsVisitor<INFERENCE>::addLog
(
	const std::string &logName
) const
{
	if (verbose_)
		std::cout << logName << std::endl;
}

template<class INFERENCE>
void
LabelCollapseStatisticsVisitor<INFERENCE>::log(
	const std::string &logName,
	const double logValue
) const
{
	if (verbose_)
		std::cout << logName << " value " << logValue << std::endl;
}


template<class INFERENCE>
typename LabelCollapseStatisticsVisitor<INFERENCE>::LabelType
LabelCollapseStatisticsVisitor<INFERENCE>::labels() const
{
	typedef typename std::vector<LabelType>::const_iterator Iterator;

	LabelType sum = 0;
	for (Iterator it = auxNumberOfLabels_.begin(); it != auxNumberOfLabels_.end(); ++it) {
		sum += *it;
	}
	return sum;
}

template<class INFERENCE>
typename LabelCollapseStatisticsVisitor<INFERENCE>::LabelType
LabelCollapseStatisticsVisitor<INFERENCE>::labelsReduced() const
{
	typedef typename std::vector<LabelType>::const_iterator Iterator;

	LabelType sum = 0;
	for (Iterator it = origNumberOfLabels_.begin(); it != origNumberOfLabels_.end(); ++it) {
		sum += *it;
	}

	return sum - labels();
}

template<class INFERENCE>
typename LabelCollapseStatisticsVisitor<INFERENCE>::LabelType
LabelCollapseStatisticsVisitor<INFERENCE>::space() const
{
	typedef typename std::vector<LabelType>::const_iterator Iterator;

	LabelType product = 1;
	for (Iterator it = auxNumberOfLabels_.begin(); it != auxNumberOfLabels_.end(); ++it) {
		product *= *it;
	}
	return product;
}

template<class INFERENCE>
typename LabelCollapseStatisticsVisitor<INFERENCE>::LabelType
LabelCollapseStatisticsVisitor<INFERENCE>::spaceReduced() const
{
	typedef typename std::vector<LabelType>::const_iterator Iterator;

	LabelType product = 1;
	for (Iterator it = origNumberOfLabels_.begin(); it != origNumberOfLabels_.end(); ++it) {
		product *= *it;
	}

	return product - space();
}

} // namespace visitors
} // namespace opengm

#endif // OPENGM_LABELCOLLAPSE_VISITOR_HXX
