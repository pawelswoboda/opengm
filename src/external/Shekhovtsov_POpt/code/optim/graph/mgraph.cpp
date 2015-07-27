#include "mgraph.h"

#include "exttype/itern.h"
#include "debug/logs.h"
#include "exttype/itern.h"
#include "data/dataset.h"

namespace datastruct{

	void mgraph::init(int nV,const dynamic::num_array<int,2> & _E){
		_nV = nV;
		int nE = _E.size()[1];
		E.resize(nE);
		for(int e=0;e<nE;++e){
			E[e] = mint2(_E[mint2(0,e)],_E[mint2(1,e)]);
		};
		edge_index();
	};
	dynamic::num_array<int,2> mgraph::get_E(){
		dynamic::num_array<int,2> R;
		R.resize(mint2(2,nE()));
		for(int e=0;e<nE();++e){
			R.subdim<1>(e) << E[e];
		};
		return R;
	};

	void mgraph::edge_index(){
		out.clear();
		in.clear();
		intf n_out(nV());
		intf n_in(nV());
		n_out<<0;
		n_in<<0;

		for(int i=0;i<E.size();++i){
			mint2 st = E[i];
			++n_out[st[0]];
			++n_in[st[1]];
		};

		out.resize(nV());
		in.resize(nV());
		for(int i=0;i<nV();++i){
			out[i].reserve(n_out[i]);
			in[i].reserve(n_in[i]);
		};

		for(int i=0;i<E.size();++i){
			mint2 st = E[i];
			out[st[0]].push_back(i);
			in[st[1]].push_back(i);
		};		
	};

	/*
	void mgraph::create_grid(const mint2& sz){
		int n1 = sz[0];
		int n2 = sz[1];
		int nV = n1*n2;
		V.resize(nV);
		for(int i=0;i<nV;++i)V[i]=i;
		int nE = (n1-1)*n2+(n2-1)*n1;
		E.reserve(nE);

		for(iter2 ii(mint2(n1-1,n2));ii.allowed();++ii){
			E.push_back(mint2(ii[0]+ii[1]*n1,(ii[0]+1)+ii[1]*n1));
		};
		for(iter2 ii(mint2(n1,n2-1));ii.allowed();++ii){
			E.push_back(mint2(ii[0]+ii[1]*n1,ii[0]+(ii[1]+1)*n1));
		};
		edge_index();
	};
	*/

	template<int rank,bool lastindexfastest> void mgraph::create_grid(const intn<rank>& sz){
		dynamic::LinIndex<rank,lastindexfastest> geometry(sz); 
		_nV = geometry.linsize();
		//V.resize(nV);
		int nE=0;
		for(int k=0;k<rank;++k){
			intn<rank> sz1 = sz;
			--sz1[k];
			nE+=sz1.prod();
		};
		E.reserve(nE);
		for(int k=0;k<rank;++k){
			if(sz[k]<1)throw debug_exception("invalid size");
			intn<rank> cc = sz;
			--cc[k];
			for(itern<rank> ii(cc);ii.allowed();++ii){
				intn<rank> jj=ii;
				++jj[k];
				E.push_back(mint2(geometry.linindex(ii),geometry.linindex(jj)));
			};
		};
		edge_index();
	};

	int mgraph::edge(int s,int t){
		intf & o = out[s];
		for(int j=0;j<o.size();++j){
			if(E[o[j]][1]==t){
				return o[j];
			};
		};
		throw debug_exception("edge does not exist in the graph.");
	};

	int mgraph::out_edge(int s,int t){
		intf & o = out[s];
		for(int j=0;j<o.size();++j){
			if(E[o[j]][1]==t){
				return o[j];
			};
		};
		throw debug_exception("edge does not exist in the graph.");
	};

	int mgraph::in_edge(int s,int t){
		intf & ee = in[s];
		for(int j=0;j<ee.size();++j){
			if(E[ee[j]][0]==t){
				return ee[j];
			};
		};
		throw debug_exception("edge does not exist in the graph.");
	};

	class my_node_order{
	public:
		const datastruct::mgraph & G;
		my_node_order(const datastruct::mgraph & _G):G(_G){};
	public:
		bool operator()( int u1, int u2)const{
			if(G.nns(u1)>G.nns(u2)) return true;
			return false;
		};
	};

	int mgraph::greedy_coloring(dynamic::fixed_array1<int> & coloring){
		dynamic::fixed_array1<int> nodes(nV());
		dynamic::fixed_array1<int> colors;
		colors.reserve(nV());
		for(int i=0;i<nV();++i){
			nodes[i] = i;
			coloring[i] = -1;
		};
		std::stable_sort(nodes.begin(),nodes.end(),my_node_order(*this));
		for(int i=0;i<nV();++i){
			int v = nodes[i];
			for(int k=0;k<colors.size();++k){//mark all colors available
				colors[k] = 1;
			};
			for(int j=0;j<in[v].size();++j){
				int u = E[in[v][j]][0];
				if(coloring[u]>=0){
					colors[coloring[u]] = 0;//the color can not be taken by v
				};
			};
			for(int j=0;j<out[v].size();++j){
				int u = E[out[v][j]][1];
				if(coloring[u]>=0){
					colors[coloring[u]] = 0;//the color can not be taken by v
				};
			};
			bool foundcolor=false;
			for(int k=0;k<colors.size();++k){
				if(colors[k] == 1){//there is free color to asign to v
					coloring[v] = k;
					foundcolor = true;
				};
			};
			if(!foundcolor){// assign new color
				int k = colors.size();
				colors.push_back(k);
				coloring[v] = k;
			};
		};
		return colors.size();
	};

/*
	template<> const fixed_vect<int> & mgraph::dir_in<true>(const int s)const{
		return in[s];
	};
	template<> const fixed_vect<int> & mgraph::dir_in<false>(const int s)const{
		return out[s];
	};
	template<> const fixed_vect<int> & mgraph::dir_out<true>(const int s)const{
		return out[s];
	};
	template<> const fixed_vect<int> & mgraph::dir_out<false>(const int s)const{
		return in[s];
	};
*/
	template void mgraph::create_grid<2,true>(const intn<2>& sz);
	template void mgraph::create_grid<2,false>(const intn<2>& sz);

	namespace{
		void make(){
			mgraph G;
			G.create_grid<1,true>(0);
			G.create_grid<2,true>(mint2(0,0));
			G.create_grid<3,true>(mint3(0,0,0));
			G.create_grid<1,false>(0);
			G.create_grid<2,false>(mint2(0,0));
			G.create_grid<3,false>(mint3(0,0,0));
		};
	};
};