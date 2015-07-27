#include "dee.h"

#include "optim/graph/mgraph.h"

class node{
public:
	node * next;
	bool active;
	int v;
	//int l;
	node():next(0),active(0){};
};

node * first_active=0;
node * last_active=0;

void queue_push_back(node * p){
	if(!p->active){
		assert(p->next==0);
		p->active = true;
		if(last_active){
			last_active->next = p;
		}else{
			first_active = p;
		};
		last_active = p;
	};
};

node * queue_pop_front(){
	node * p = first_active;
	if(p){
		first_active = p->next;
		p->next = 0;
		p->active = 0;
		if(p==last_active){
			last_active = 0;
		};
	};
	return p;
};


long long dee(const num_array<int,2>  & E, const num_array<int,2> & f1, const num_array<int,3> & f2, num_array<int,2> & X, num_array<int,2> & P, double eps){

	int eliminated = 0;

	int K = f1.size()[0];
	int nV = f1.size()[1];
	int nE = E.size()[1];

	dynamic::num_array<node,1> nodes(nV);

	datastruct::mgraph G;
	G.init(nV,E);

	for(int v=0;v<nV;++v){
		nodes(v).v = v;
		queue_push_back(&nodes(v));
		for(int i=0;i<K;++i){
			P(i,v) = i;
		};
	};

	while(first_active){
		node * p = queue_pop_front();
		int s = p->v;
		//check DEE condition
		for(int a = 0; a<K;++a){// try eliminate label a
			if(X(a,s)==0)continue;
			for(int b = 0; b<K;++b){// by finding a better label b
				if(b==a || X(b,s)==0)continue;
				int delta = f1(a,s)-f1(b,s);
				//
				for(int j=0;j<G.out[s].size();++j){// edges s->t
					int e = G.out[s][j];
					int t = G.E[e][1];
					int r = (1<<30);
					const int * p1 = &f2(a,0,e);
					const int * p2 = &f2(b,0,e);
					const int * pX = &X(0,t);
					int stride = &f2(0,1,e)-&f2(0,0,e);//f2.stride(0,1,0);
					for(int l=0;l<K;++l){
						if(*pX!=0){
							int d = *p1-*p2;
							//if(X(l,t)==0)continue;
							//int d = f2(a,l,e)-f2(b,l,e);
							if(d<r){
								r = d;
							};
						};
						++pX;
						p1+=stride;
						p2+=stride;
					};
					delta += r;
				};
				for(int j=0;j<G.in[s].size();++j){ // edges t->s
					int e = G.in[s][j];
					int t = G.E[e][0];
					int r = (1<<30);
					//for(int l=0;l<K;++l){
					//	if(X(l,t)==0)continue;
					//	int d = f2(l,a,e)-f2(l,b,e);
					//	if(d<r){
					//		r = d;
					//	};
					//};
					const int * p1 = &f2(0,a,e);
					const int * p2 = &f2(0,b,e);
					const int * pX = &X(0,t);
					for(int l=0;l<K;++l){
						if(*pX!=0){
							int d = *p1-*p2;
							if(d<r){
								r = d;
							};
						};
						++pX;
						++p1;
						++p2;
					};
					delta += r;
				};
				if(delta>=eps){// elliminate a and make neighbours active
					// check if b not allready mapped
                    //int c = P(b,s); //must be b or where b is mapped to
                    //if(c==a) continue;
					//P(a,s) = c;
					X(a,s) = 0;
					P(a,s) = b;
					for(int k=0;k<K;++k){
					    if(P(k,s)==a){
					        P(k,s) = b;
					    };
					};
					++eliminated;
					for(int j=0;j<G.out[s].size();++j){// edges s->t
						int e = G.out[s][j];
						int t = G.E[e][1];
						queue_push_back(&nodes[t]);
					};
					for(int j=0;j<G.in[s].size();++j){// edges t->s
						int e = G.in[s][j];
						int t = G.E[e][0];
						queue_push_back(&nodes[t]);
					};
				};
			};
		};
	};
	return eliminated;
};
