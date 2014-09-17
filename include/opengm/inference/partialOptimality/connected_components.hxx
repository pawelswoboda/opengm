#pragma once
#ifndef OPENGM_CC_HXX
#define OPENGM_CC_HXX

#include <vector>
#include <set>
#include <map>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
#include "opengm/utilities/disjoint-set.hxx"
#include "opengm/functions/view.hxx"
#include "opengm/functions/explicit_function.hxx"
#include "opengm/graphicalmodel/space/discretespace.hxx"
#include "opengm/graphicalmodel/graphicalmodel.hxx"

namespace opengm {

template<class GM, class GMV>
void getConnectComp(
		const GM& gm_,
		std::vector< std::vector<typename GMV::IndexType> >& cc2gm, 
		std::vector<GMV>& models
		)
{
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename GM::ValueType ValueType;
	typedef typename GM::OperatorType OperatorType;
	typedef disjoint_set<IndexType> Set;

	models.clear();
	Set CC(gm_.numberOfVariables());
	std::map<IndexType, IndexType> representives;
	std::vector< std::vector<IndexType> > cc2gmINT;
	std::vector< IndexType > gm2ccIDX(gm_.numberOfVariables());

	for(IndexType f=0 ; f < gm_.numberOfFactors() ; ++f){
		OPENGM_ASSERT( gm_[f].numberOfVariables() <= 2)
			if(gm_[f].numberOfVariables() == 2){
				IndexType var1 = gm_[f].variableIndex(0);
				IndexType var2 = gm_[f].variableIndex(1);
				CC.join(var1,var2);
			}
	}

	CC.representativeLabeling(representives);

	std::vector<bool> isSet(CC.numberOfSets(),true);
	cc2gmINT.resize(CC.numberOfSets());
	std::vector<std::set<IndexType> > setFactors(CC.numberOfSets());
	IndexType numCC = CC.numberOfSets();
	std::vector<IndexType> IndexOfCC(CC.numberOfSets(),0);
	for(IndexType var = 0 ; var < gm_.numberOfVariables() ; ++var){
		IndexType n = CC.find(var);
		n = representives[n];
		cc2gmINT[n].push_back(var);
		gm2ccIDX[var]=IndexOfCC[n];
		IndexOfCC[n]++;

		for(IndexType i=0;i<gm_.numberOfFactors(var);++i){
			IndexType fkt = gm_.factorOfVariable(var,i);
			if(gm_[fkt].numberOfVariables() == 1){
				setFactors[n].insert(fkt);
			}
			else if(gm_[fkt].numberOfVariables() == 2){
				IndexType var1 = gm_[fkt].variableIndex(0);
				IndexType var2 = gm_[fkt].variableIndex(1);
				setFactors[n].insert(fkt);
			} else {
				throw "only pairwise or unary factors supported in connected components";
			}
		}  

	}
	models.resize(numCC);
	cc2gm.resize(numCC);
	IndexType countCC = 0;
	typename std::set<IndexType>::iterator it;

   std::cout << "found indices for connected components: " << CC.numberOfSets() << " components" << std::endl;

	for(IndexType i=0;i<CC.numberOfSets();++i){
		if(isSet[i] == true){
			LabelType StateSpace[cc2gmINT[i].size()];
			for(IndexType j=0;j<cc2gmINT[i].size();++j){
				LabelType label=gm_.numberOfLabels(cc2gmINT[i][j]);
				StateSpace[j]=label;
			}
			GMV gmV(typename GM::SpaceType(StateSpace,StateSpace+cc2gmINT[i].size()));
			
         std::cout << "component " << i << " has " << cc2gmINT[i].size() << " nodes" << std::endl;

			for(it=setFactors[i].begin();it!=setFactors[i].end();it++){
				//if(gm_.numberOfVariables(*it) == 2){
					IndexType var[gm_.numberOfVariables(*it)];
					for(IndexType l=0;l<gm_.numberOfVariables(*it);++l){
						IndexType idx=gm_.variableOfFactor(*it,l);
						var[l]=gm2ccIDX[idx];

					}
					ViewFunction<GM> func(gm_[*it]);
					gmV.addFactor(gmV.addFunction(func),var,var + gm_.numberOfVariables(*it));
				//}
				//else{
				//	IndexType idx=gm_.variableOfFactor(*it,0);
				//	IndexType var[]={gm2ccIDX[idx]};          
				//	cout << "idx = " << idx << endl;
				//	cout << "var = " << var[0] << endl;
				//	gmV.addFactor(gmV.addFunction(unaryFunc[idx]),var,var + 1);
				//}
			}

			models[countCC]=gmV;
			cc2gm[countCC]=cc2gmINT[i];
			countCC++;

		}
	}
}

template<class GM, class T>
void reassembleFromCC(
		std::vector<T>& l,
		std::vector<std::vector<T> >& ls, 
		std::vector< std::vector<typename GM::IndexType> >& cc2gm
		)
{
	typedef typename GM::IndexType IndexType;
	IndexType totalLength = 0;
	OPENGM_ASSERT( cc2gm.size() == ls.size() );
	for( IndexType c=0; c<cc2gm.size(); c++ ) {
		totalLength += cc2gm[c].size(); 
		OPENGM_ASSERT( cc2gm[c].size() == ls[c].size() );
	}
	l.resize(totalLength);

	for( IndexType c=0; c<cc2gm.size(); c++ ) {
		for( IndexType i=0; i<cc2gm[c].size(); i++ ) {
			l[ cc2gm[c][i] ] = ls[c][i];
		}
	}
}

} // end namespace opengm

#endif // #ifndef OPENGM_PCC_HXX
