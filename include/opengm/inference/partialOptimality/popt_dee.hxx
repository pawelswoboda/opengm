#ifndef OPENGM_DEE_HXX
#define OPENGM_DEE_HXX


#include <opengm/opengm.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/partialOptimality/popt_data.hxx>
#include <opengm/utilities/tribool.hxx>
#include <opengm/datastructures/marray/marray.hxx>

#include <vector>

#include "popt_inference_base.hxx"

/*
    DEE4, DEE3 and DEE1 can be found on the pages 27-29 of Alexander Shekhovtsov's PhD Thesis
    "Exact and Partial Energy Minimization in Computer Vision"
    http://cmp.felk.cvut.cz/ ̃shekhovt/publications/as-phd-thesis-TR.pdf

    DEEPairwise (alias DEE2) can be found on page 8 of
    "Maximum Persistency in Energy Minimization, Alexander Shekhovtsov"
    http://cmp.felk.cvut.cz/~shekhovt/publications/shekhovtsov-2014-improving-mappings-CVPR-final-copy.pdf

*/

namespace opengm
{

////! [class DEE - dead end elimination]
/// based on "The dead-end elimination theorem and its use in protein side-chain positioning",
/// originally by J. Desmet, M. D. Maeyer, B. Hazes and I. Lasters, Nature 1992
/// also see "Maximum Persistency in Energy Minimization", A. Shekhovtsov, CVPR 2014
/// for concise description.
///
/// Corresponding author: Paul Swoboda, email: swoboda@math.uni-heidelberg.de
///
///\ingroup inference

template<class DATA, class ACC>
class DEE : public POpt_Inference<DATA, ACC>
{
public:
    typedef typename DATA::GraphicalModelType GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    typedef ValueType GraphValueType; 
    enum MethodType {DEE1,DEE_Pairwise,DEE3,DEE4};

    struct Parameter {
       Parameter(MethodType m = DEE1) {
          method_ = m;
       }
       MethodType method_;
    };

    DEE(DATA& gmD, const Parameter& param);

    std::string name() const { return "DEE"; };
    const GraphicalModelType& graphicalModel() const { return gm_; };

    InferenceTermination infer();

    InferenceTermination dee4();
    InferenceTermination dee3();
    InferenceTermination dee1();
    InferenceTermination deePairwise();

private:
    DATA& gmD_;
    const GraphicalModelType& gm_;
    Parameter param_;

    const static double eps_;
};

template<class DATA,class ACC>
const double DEE<DATA,ACC>::eps_ = 1.0e-8;

template<class DATA,class ACC>
DEE<DATA,ACC>::DEE(DATA& gmD, const Parameter& param) 
: gmD_(gmD), 
   gm_(gmD.graphicalModel()),
   param_(param)
{
}

template<class DATA,class ACC>
InferenceTermination
DEE<DATA,ACC>::infer()
{
   switch( param_.method_ ) {
      case DEE1: dee1(); break;
      case DEE_Pairwise: deePairwise(); break;
      case DEE3: dee3(); break;
      case DEE4: dee4(); break;
      default: throw;
   }
}


template<class DATA, class ACC>
InferenceTermination 
DEE<DATA,ACC>::dee4()
{
    if (!gm_.maxFactorOrder(2))
    {
        throw RuntimeError("This implementation of DEE4 supports only factors of order <= 2.");
        return INFERENCE_ERROR;
    }

    std::queue<IndexType> variables;
    std::vector<bool> variableInQueue;

    for (IndexType v = 0; v < gm_.numberOfVariables(); v++)
    {
        if(gmD_.getOpt(v)){
            variableInQueue.std::vector<bool>::push_back(false);
        }else{
            variables.std::queue<IndexType>::push(v);
            variableInQueue.std::vector<bool>::push_back(true);
        }
    }

    ///DEE4
    while (!(variables.std::queue<IndexType>::empty()))
    {
        LabelType singleShape [] = {0};
        LabelType doubleShape [] = {0,0};

        IndexType variable = variables.std::queue<IndexType>::front();
        OPENGM_ASSERT(!gmD_.getOpt(variable));
        variables.std::queue<IndexType>::pop();
        variableInQueue[variable] = false;

        typename marray::Vector<IndexType> firstOrderFactor;
        typename marray::Vector<IndexType> secondOrderFactor;

        IndexType numNeighbours = gm_.numberOfNthOrderFactorsOfVariable (variable,1,firstOrderFactor);
        OPENGM_ASSERT(firstOrderFactor.size() == 1);
        numNeighbours = gm_.numberOfNthOrderFactorsOfVariable(variable,2,secondOrderFactor);

        IndexType numLabels = gm_.numberOfLabels(variable);

        for (LabelType alpha = 0; alpha < numLabels; alpha++)
        {
            if(gmD_.getPOpt(variable,alpha) == opengm::Tribool::Maybe)
            {
                singleShape[0] = alpha;
                ValueType functionValueAlpha = gm_[firstOrderFactor[0]](singleShape);

                ValueType DEE4 = 0;
                ValueType minSum = functionValueAlpha;

                for (IndexType i = 0; i < numNeighbours; i++)
                {
                    IndexType factor = secondOrderFactor[i];
                    IndexType neighbour = gm_.secondVariableOfSecondOrderFactor(size_t(variable),factor);

                    size_t varIdx, neighbourIdx;

                    if(variable < neighbour)
                    {
                        varIdx = 0;
                        neighbourIdx = 1;
                    }
                    else
                    {
                        varIdx = 1;
                        neighbourIdx = 0;
                    }

                    doubleShape[varIdx] = alpha;
                    LabelType firstLabel = 0;

                    for(LabelType labelNeighbour = 0; labelNeighbour < gm_.numberOfLabels(neighbour); labelNeighbour++)
                    {

                        if(!(gmD_.getPOpt(neighbour,labelNeighbour) == opengm::Tribool::False))
                        {
                            firstLabel = labelNeighbour;
                            break;
                        }
                    }

                    doubleShape[neighbourIdx] = firstLabel;
                    ValueType minNeighbour = gm_[factor](doubleShape);

                    for(LabelType labelNeighbour = firstLabel+1; labelNeighbour < gm_.numberOfLabels(neighbour); labelNeighbour++)
                    {

                        if(!(gmD_.getPOpt(neighbour,labelNeighbour)== opengm::Tribool::False))
                        {
                            doubleShape[neighbourIdx] = labelNeighbour;

                            if (gm_[factor](doubleShape) < minNeighbour)
                                minNeighbour = gm_[factor](doubleShape);
                        }
                    }

                    minSum += minNeighbour;

                }

                for(LabelType beta = 0; beta < numLabels; beta++)
                {
                    if((gmD_.getPOpt(variable,beta) == opengm::Tribool::Maybe) && beta!=alpha)
                    {
                        DEE4 = minSum;

                        for (IndexType i = 0; i < numNeighbours; i++)
                        {
                            IndexType factor = secondOrderFactor[i];
                            IndexType neighbour = gm_.secondVariableOfSecondOrderFactor(variable,factor);

                            size_t varIdx, neighbourIdx;

                            if(variable < neighbour)
                            {
                                varIdx = 0;
                                neighbourIdx = 1;
                            }
                            else
                            {
                                varIdx = 1;
                                neighbourIdx = 0;
                            }

                            doubleShape[varIdx] = beta;
                            LabelType firstLabel = 0;

                            for(LabelType labelNeighbour = 0; labelNeighbour < gm_.numberOfLabels(neighbour); labelNeighbour++)
                            {
                                if(!(gmD_.getPOpt(neighbour,labelNeighbour) == opengm::Tribool::False))
                                {
                                    firstLabel = labelNeighbour;
                                    break;
                                }
                            }

                            doubleShape[neighbourIdx] = firstLabel;
                            ValueType maxNeighbour = gm_[factor](doubleShape);

                            for (LabelType labelNeighbour = firstLabel+1; labelNeighbour < gm_.numberOfLabels(neighbour); labelNeighbour++)
                            {
                                if(!(gmD_.getPOpt(neighbour,labelNeighbour) == opengm::Tribool::False))
                                {
                                    doubleShape[neighbourIdx] = labelNeighbour;

                                    if (gm_[factor](doubleShape) > maxNeighbour)
                                        maxNeighbour = gm_[factor](doubleShape);
                                }
                            }


                            DEE4 -= maxNeighbour;
                        }

                        singleShape[0] = beta;
                        ValueType functionValueBeta = gm_[firstOrderFactor[0]](singleShape);

                        DEE4 -= functionValueBeta;


                        if (eps_ < DEE4)
                        {
                            gmD_.setFalse(variable,alpha);

                            for(size_t n=0; n<numNeighbours; n++)
                            {
                                IndexType factor = secondOrderFactor[n];
                                IndexType neighbour = gm_.secondVariableOfSecondOrderFactor(variable,factor);

                                if(!(variableInQueue[neighbour] || gmD_.getOpt(neighbour)))
                                {
                                    variables.std::queue<IndexType>::push(neighbour);
                                    variableInQueue[neighbour] = true;
                                }
                            }

                            break;
                        }
                    }
                }
            }
        }
    }
    return CONVERGENCE;
}

template<class DATA, class ACC>
InferenceTermination 
DEE<DATA,ACC>::dee3()
{
    if (!gm_.maxFactorOrder(2))
    {
        throw RuntimeError("This implementation of DEE3 supports only factors of order <= 2.");
        return INFERENCE_ERROR;
    }

    ///DEE3

    std::queue<IndexType> variables;
    std::vector<bool> variableInQueue;

    for (IndexType v = 0; v < gm_.numberOfVariables(); v++)
    {
        if(gmD_.getOpt(v)){
            variableInQueue.std::vector<bool>::push_back(false);
        }else{
            variables.std::queue<IndexType>::push(v);
            variableInQueue.std::vector<bool>::push_back(true);
        }
    }

    while (!(variables.std::queue<IndexType>::empty()))
    {
        IndexType singleShape [] = {0};
        IndexType doubleShapeAlpha [] = {0,0};
        IndexType doubleShapeBeta [] = {0,0};

        IndexType variable = variables.std::queue<IndexType>::front();
        OPENGM_ASSERT(!gmD_.getOpt(variable));
        variables.std::queue<IndexType>::pop();
        variableInQueue[variable] = false;


        typename marray::Vector<size_t> firstOrderFactor;
        typename marray::Vector<size_t> secondOrderFactor;

        size_t numNeighbours = gm_.numberOfNthOrderFactorsOfVariable (variable,1,firstOrderFactor);
        OPENGM_ASSERT(firstOrderFactor.size() == 1);
        numNeighbours = gm_.numberOfNthOrderFactorsOfVariable(variable,2,secondOrderFactor);

        IndexType numLabels = gm_.numberOfLabels(variable);

        for (LabelType alpha = 0; alpha < numLabels; alpha++)
        {

            if(gmD_.getPOpt(variable,alpha) == opengm::Tribool::Maybe)
            {
                singleShape[0] = alpha;
                ValueType functionValueAlpha = gm_[firstOrderFactor[0]](singleShape);

                for (LabelType beta = 0; beta < numLabels; beta++)
                {
                    if((gmD_.getPOpt(variable,beta) == opengm::Tribool::Maybe) && beta!=alpha)
                    {
                        singleShape[0] = beta;
                        ValueType functionValueBeta = gm_[firstOrderFactor[0]](singleShape);

                        ValueType DEE3 = functionValueAlpha - functionValueBeta;

                        for (IndexType i = 0; i < numNeighbours; i++)
                        {
                            IndexType factor = secondOrderFactor[i];
                            IndexType neighbour = gm_.secondVariableOfSecondOrderFactor(variable,factor);

                            IndexType varIdx, neighbourIdx;

                            if(variable < neighbour)
                            {
                                varIdx = 0;
                                neighbourIdx = 1;
                            }
                            else
                            {
                                varIdx = 1;
                                neighbourIdx = 0;
                            }

                            LabelType firstLabel = 0;

                            for(LabelType label = 0; label < gm_.numberOfLabels(neighbour); label++)
                            {
                                if(!(gmD_.getPOpt(neighbour,label) == opengm::Tribool::False))
                                {
                                    firstLabel = label;
                                    break;
                                }
                            }

                            doubleShapeAlpha[varIdx] = alpha;
                            doubleShapeBeta[varIdx] = beta;
                            doubleShapeAlpha[neighbourIdx] = firstLabel;
                            doubleShapeBeta[neighbourIdx] = firstLabel;
                            ValueType minNeighbour = gm_[factor](doubleShapeAlpha) - gm_[factor](doubleShapeBeta);

                            for (LabelType labelNeighbour = firstLabel+1; labelNeighbour < gm_.numberOfLabels(neighbour); labelNeighbour++)
                            {
                                if(!(gmD_.getPOpt(neighbour,labelNeighbour) == opengm::Tribool::False))
                                {
                                    doubleShapeAlpha[neighbourIdx] = labelNeighbour;
                                    doubleShapeBeta[neighbourIdx] = labelNeighbour;
                                    if (minNeighbour > (gm_[factor](doubleShapeAlpha) - gm_[factor](doubleShapeBeta)))
                                        minNeighbour = gm_[factor](doubleShapeAlpha) - gm_[factor](doubleShapeBeta);
                                }
                            }

                            DEE3 += minNeighbour;
                        }

                        if (eps_ < DEE3)
                        {
                            gmD_.setFalse(variable,alpha);

                            for(size_t n=0; n<numNeighbours; n++)
                            {
                                IndexType factor = secondOrderFactor[n];
                                IndexType neighbour = gm_.secondVariableOfSecondOrderFactor(variable,factor);

                                if(!(variableInQueue[neighbour] || gmD_.getOpt(neighbour)))
                                {
                                    variables.std::queue<IndexType>::push(neighbour);
                                    variableInQueue[neighbour] = true;
                                }
                            }

                            break;
                        }
                    }
                }
            }
        }
    }
    return CONVERGENCE;
}

template<class DATA, class ACC>
InferenceTermination 
DEE<DATA,ACC>::dee1()
{

    if (!gm_.maxFactorOrder(2))
    {
        throw RuntimeError("This implementation of DEE1 supports only factors of order <= 2.");
        return INFERENCE_ERROR;
    }

    //DEE1
    std::queue<IndexType> variables;
    std::vector<bool> variableInQueue;

    for (IndexType v = 0; v < gm_.numberOfVariables(); v++)
    {
        if(gmD_.getOpt(v)){
            variableInQueue.std::vector<bool>::push_back(false);
        }else{
            variables.std::queue<IndexType>::push(v);
            variableInQueue.std::vector<bool>::push_back(true);
        }
    }

    while (!(variables.std::queue<IndexType>::empty()))
    {
        IndexType singleShape [] = {0};
        IndexType doubleShape [] = {0,0};

        IndexType variable = variables.std::queue<IndexType>::front();
        OPENGM_ASSERT(!gmD_.getOpt(variable));
        variables.std::queue<IndexType>::pop();
        variableInQueue[variable] = false;

        typename marray::Vector<IndexType> firstOrderFactor;
        typename marray::Vector<IndexType> secondOrderFactor;

        IndexType numNeighbours = gm_.numberOfNthOrderFactorsOfVariable(variable,1,firstOrderFactor);
        OPENGM_ASSERT(firstOrderFactor.size() == 1);
        numNeighbours = gm_.numberOfNthOrderFactorsOfVariable(variable,2,secondOrderFactor);

        IndexType numLabels = gm_.numberOfLabels(variable);

        for(LabelType alpha = 0; alpha < numLabels; alpha++)
        {
            if(gmD_.getPOpt(variable,alpha) == opengm::Tribool::Maybe)
            {
                singleShape[0] = alpha;
                ValueType functionValueAlpha = gm_[firstOrderFactor[0]](singleShape);

                bool DEE1 = true;
                ValueType neighbourSum;

                //Set first labelVector and compute number of iteration
                IndexType neighbours [numNeighbours];
                IndexType numberOfLabelsOfNeighbours [numNeighbours];
                IndexType numberOfLabelVector = 1;
                LabelType labelVector [numNeighbours];
                std::vector<LabelType> neighboursLabelVector [numNeighbours];

                for (IndexType i = 0; i < numNeighbours; i++)
                {
                    neighbours[i] = gm_.secondVariableOfSecondOrderFactor(variable,secondOrderFactor[i]);
                    numberOfLabelsOfNeighbours[i] = 0;

                    for (LabelType j = 0; j < gm_.numberOfLabels(neighbours[i]); j++)
                    {
                        if (!(gmD_.getPOpt(neighbours[i],j) == opengm::Tribool::False))
                        {
                            numberOfLabelsOfNeighbours[i] += 1;
                            neighboursLabelVector[i].std::vector<LabelType>::push_back(j);
                        }
                    }
                    numberOfLabelVector = numberOfLabelVector * numberOfLabelsOfNeighbours[i];

                    labelVector[i] = neighboursLabelVector[i][0];
                }

                for (size_t labeling = 0; labeling < numberOfLabelVector; labeling++)
                {
                    ValueType maxValue = 0;

                    for (LabelType beta = 0; beta < numLabels; beta++)
                    {
                        if(gmD_.getPOpt(variable,beta) == opengm::Tribool::Maybe && beta!=alpha)
                        {
                            singleShape[0] = beta;
                            ValueType functionValueBeta = gm_[firstOrderFactor[0]](singleShape);

                            neighbourSum = functionValueAlpha - functionValueBeta;

                            for (IndexType factor = 0; factor < numNeighbours; factor++)
                            {
                                IndexType varIdx, neighbourIdx;

                                if(variable < neighbours[factor])
                                {
                                    varIdx = 0;
                                    neighbourIdx = 1;
                                }
                                else
                                {
                                    varIdx = 1;
                                    neighbourIdx = 0;
                                }

                                doubleShape[varIdx] = alpha;
                                doubleShape[neighbourIdx] = labelVector[factor];
                                neighbourSum += gm_[secondOrderFactor[factor]](doubleShape);

                                doubleShape[varIdx] = beta;
                                neighbourSum -= gm_[secondOrderFactor[factor]](doubleShape);
                            }

                            if (neighbourSum > maxValue)
                            {
                                maxValue = neighbourSum;
                            }
                        }
                    }

                    if (maxValue < eps_)
                    {
                        DEE1 = false;
                        break;
                    }

                    //Get new labelVector
                    IndexType gNwmL = numNeighbours-1; //gNwmL - greatestNeighbourWithMaximalLabel

                    for(IndexType i = 0; i < numNeighbours; ++i)
                    {
                        if(labelVector[i] != neighboursLabelVector[i][numberOfLabelsOfNeighbours[i]-1])
                        {
                            gNwmL = i;
                            break;
                        }
                    }

                    for(IndexType i = 0; i < numberOfLabelsOfNeighbours[gNwmL]-1; i++)
                    {
                        if(neighboursLabelVector[gNwmL][i] == labelVector[gNwmL])
                        {
                            labelVector[gNwmL] = neighboursLabelVector[gNwmL][i+1];
                            break;
                        }
                    }

                    for (IndexType i = 0; i < gNwmL; i++)
                    {
                        labelVector[i] = neighboursLabelVector[i][0];
                    }
                }


                if (DEE1)
                {
                    gmD_.setFalse(variable,alpha);

                    for(IndexType n=0; n<numNeighbours; n++)
                    {
                        IndexType factor = secondOrderFactor[n];
                        IndexType neighbour = gm_.secondVariableOfSecondOrderFactor(variable,factor);

                        if(!(variableInQueue[neighbour] || gmD_.getOpt(neighbour)))
                        {
                            variables.std::queue<IndexType>::push(neighbour);
                            variableInQueue[neighbour] = true;
                        }
                    }
                }
            }
        }
    }
    return CONVERGENCE;
}

template<class DATA, class ACC>
InferenceTermination 
DEE<DATA,ACC>::deePairwise()
{
    return CONVERGENCE;
}


} // namespace opengm

#endif // OPENGM_DEE_HXX


