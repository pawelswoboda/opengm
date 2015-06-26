//
// File: labelcollapse_declarations.hxx
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
#ifndef OPENGM_LABELCOLLAPSE_DECLARATIONS_HXX
#define OPENGM_LABELCOLLAPSE_DECLARATIONS_HXX

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
template<class GM, class ACC>
struct LabelCollapseAuxTypeGen;

//
// Namespace for internal implementation details.
//
namespace labelcollapse {

// Builds the auxiliary model given the original model.
template<class GM, class INF>
class ModelBuilder;

// Reorders labels according to their unary potentials.
template<class GM, class ACC>
class Reordering;

// A view function which returns the values from the original model if the
// nodes are not collapsed. If they are, the view function will return the
// corresponding epsilon value.
template<class GM, class ACC>
class EpsilonFunction;

} // namespace labelcollapse
} // namespace opengm

#endif
