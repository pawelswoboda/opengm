#include "dee.h"

#include "optim/graph/mgraph.h"
#include <queue>

//#include <yvals.h>
//#define CHAR16_T
//#include "mex/mex_io.h"

using namespace exttype;

class node {
public:
    bool active;
    int v;
    node():active(0) {};
};

class edge {
public:
    bool active;
    int e;
    edge():active(0) {};
};

class dee2_compute {
public:
    const num_array<int,2>  & E;
    const num_array<int,2> & f1;
    const num_array<int,3> & f2;
    num_array<int,2> & X;
    num_array<int,2> & P;
    double eps;
public:
    int eliminated;
    int eliminated2;
    int K;
    int nV;
    int nE;
    datastruct::mgraph G;
    dynamic::num_array<int,4> M;
    dynamic::num_array<int,2> M_valid;
    dynamic::num_array<int,3> eX; // size K x K x nE eliminated edges
    dynamic::num_array<node,1> nodes;
    dynamic::num_array<edge,1> edges;
    std::queue<node*> nq;
    std::queue<edge*> eq;
    dynamic::num_array<int,2> delta_s;
    dynamic::num_array<int,2> delta_t;
public:

    void push_node(int s) {
        node * p = &nodes[s];
        if(!p->active) {
            nq.push(p);
            p->active = true;
        };
    };

    int pop_node() {
        node * p = nq.front();
        p->active = false;
        nq.pop();
        return p->v;
    };

    void push_edge(int e) {
        edge * p = &edges[e];
        if(!p->active) {
            eq.push(p);
            p->active = true;
        };
    };

    int pop_edge() {
        edge * p = eq.front();
        p->active = false;
        eq.pop();
        return p->e;
    };

    dee2_compute(const num_array<int,2>  & _E, const num_array<int,2> & _f1, const num_array<int,3> & _f2, num_array<int,2> & _X, num_array<int,2> & _P, double _eps):E(_E),f1(_f1),f2(_f2),X(_X),P(_P),eps(_eps) {
        eliminated = 0;
        eliminated2 = 0;
        K = f1.size()[0];
        nV = f1.size()[1];
        nE = E.size()[1];
        G.init(nV,E);
        M.resize(mint4(K,K,nE,2));
        eX.resize(mint3(K,K,nE));
        eX << 1;
        M_valid.resize(mint2(nE,2));
        nodes.resize(nV);
        edges.resize(nE);
        delta_s.resize(mint2(K,K));
        delta_t.resize(mint2(K,K));
        //nq.reserve(nV);
        //eq.reserve(nE);
        M_valid << 0;
        for(int v=0; v<nV; ++v) {
            nodes(v).v = v;
            push_node(v);
            for(int i=0; i<K; ++i) {
                P(i,v) = i;
            };
        };
        for(int e=0; e<nE; ++e) {
            edges[e].e = e;
            push_edge(e);
        };
    };
    //
    void update_M0(int e) {
        if(M_valid(e,0))return;
        int s = G.E[e][0];
        int t = G.E[e][1];
        for(int a = 0; a<K; ++a) { // try eliminate label a
            if(X(a,s)==0)continue;
            for(int b = 0; b<K; ++b) { // by finding a better label b
                if(b==a || X(b,s)==0)continue;
                const int * p1 = &f2(a,0,e);
                const int * p2 = &f2(b,0,e);
                const int * pX = &X(0,t);
                const int * peX = &eX(a,0,e);
                int stride = f2.stride(0,1,0);
                int r = (1<<30);
                for(int l=0; l<K; ++l) {
                    //if(*pX!=0 && eX(b,l,e)!=0 && eX(a,l,e)!=0){//  *peX!=0){
                    if(*pX!=0 && *peX!=0) {
                        int d = *p1-*p2;
                        //if(X(l,t)==0)continue;
                        //int d = f2(a,l,e)-f2(b,l,e);
                        if(d<r) {
                            r = d;
                        };
                    };
                    ++pX;
                    p1+=stride;
                    p2+=stride;
                    peX+=stride;
                };
                M(a,b,e,0) = r;
            };
        };
        M_valid(e,0) = true;
    };

    void update_M1(int e) {
        if(M_valid(e,1))return;
        int s = G.E[e][1];
        int t = G.E[e][0];
        for(int a = 0; a<K; ++a) { // try eliminate label a
            if(X(a,s)==0)continue;
            for(int b = 0; b<K; ++b) { // by finding a better label b
                if(b==a || X(b,s)==0)continue;
                int r = (1<<30);
                const int * p1 = &f2(0,a,e);
                const int * p2 = &f2(0,b,e);
                const int * pX = &X(0,t);
                const int * peX = &eX(0,a,e);
                for(int l=0; l<K; ++l) {
                    //if(*pX!=0 && eX(l,b,e)!=0 && eX(l,a,e)!=0){ //*peX!=0){
                    if(*pX!=0 && *peX!=0) {
                        int d = *p1-*p2;
                        if(d<r) {
                            r = d;
                        };
                    };
                    ++pX;
                    ++p1;
                    ++p2;
                    ++peX;
                };
                M(a,b,e,1) = r;
            };
        };
        M_valid(e,1) = true;
    };

    void edges_delta(int s, int skip_edge, dynamic::num_array<int, 2> & delta) {
        delta << 0;
        for(int j=0; j<G.out[s].size(); ++j) { // edges s->t
            int e = G.out[s][j];
            if(e==skip_edge)continue;
            int t = G.E[e][1];
            update_M0(e);
            for(int a=0; a<K; ++a) {
                for(int b=0; b<K; ++b) {
                    delta(a,b) += M(a,b,e,0);
                };
            };
            /*
            int * pM = &M(0,0,e,0);
            for(int * pd = delta.begin(); pd != delta.end();++pd, ++pM){
            	*pd += *pM;
            };
            */
        };
        for(int j=0; j<G.in[s].size(); ++j) { // edges t->s
            int e = G.in[s][j];
            if(e==skip_edge)continue;
            int t = G.E[e][0];
            update_M1(e);
            for(int a=0; a<K; ++a) {
                for(int b=0; b<K; ++b) {
                    delta(a,b) += M(a,b,e,1);
                };
            };
            /*
            int * pM = &M(0,0,e,1);
            for(int * pd = delta.begin(); pd != delta.end();++pd, ++pM){
            	*pd += *pM;
            };
            */
        };
    };

    void activate(int s) {
        for(int j=0; j<G.out[s].size(); ++j) { // edges s->t
            int e = G.out[s][j];
            int t = G.E[e][1];
            push_node(t);
            M_valid(e,1) = false;
            push_edge(e);
        };
        for(int j=0; j<G.in[s].size(); ++j) { // edges t->s
            int e = G.in[s][j];
            int t = G.E[e][0];
            push_node(t);
            M_valid(e,0) = false;
            push_edge(e);
        };
    };

    void compute() {
        while(!eq.empty() || !nq.empty()) {
            while(!nq.empty()) {
                int s = pop_node();
                edges_delta(s,-1,delta_s);
                //check DEE1 condition
                for(int a = 0; a<K; ++a) { // try eliminate label a
                    if(X(a,s)==0)continue;
                    for(int b = 0; b<K; ++b) { // by finding a better label b
                        if(b==a || X(b,s)==0)continue;
                        int delta = f1(a,s)-f1(b,s);
                        delta += delta_s(a,b);
                        if(delta>=eps) { // elliminate a and make neighbours active
                            //P(a,s) = b;
                            // check if b not allready mapped
                            int c = P(b,s); //must be b or where b is mapped to
                            //P(a,s) = c;
                            for(int k=0; k<K; ++k) {
                                if(P(k,s)==a) {
                                    P(k,s)=c;
                                    X(k,s) = 0;
                                };
                            };
                            ++eliminated;
                            activate(s);
                            for(int j=0; j<G.out[s].size(); ++j) { // edges s->t
                                int e = G.out[s][j];
                                for(int l=0; l<K; ++l) {
                                    if(eX(a,l,e) == 0)continue; // already eliminated
                                    eX(a,l,e) = 0;
                                    ++eliminated2;
                                };
                            };
                            for(int j=0; j<G.in[s].size(); ++j) { // edges t->s
                                int e = G.in[s][j];
                                for(int l=0; l<K; ++l) {
                                    if(eX(l,a,e) == 0)continue; // already eliminated
                                    eX(l,a,e) = 0;
                                    ++eliminated2;
                                };
                            };
                        };
                    };
                };
            };
            // pick an edge
            if(!eq.empty()) {
                int e = pop_edge();
                int s = G.E[e][0];
                int t = G.E[e][1];
                edges_delta(s,e,delta_s);
                edges_delta(t,e,delta_t);
                //check DEE2 condition
                for(int a1 = 0; a1<K; ++a1) { // try eliminate label (a1,a2)
                    if(X(a1,s)==0)continue;
                    for(int a2 = 0; a2<K; ++a2) {
                        if(X(a2,t)==0)continue;
                        if(eX(a1,a2,e) == 0)continue; // eliminated
                        for(int b1 = 0; b1<K; ++b1) { // by finding a better label (b1,b2)
                            if(b1==a1 || X(b1,s)==0)continue;
                            for(int b2 = 0; b2<K; ++b2) {
                                if(b2==a2 || X(b2,t)==0)continue;
                                int delta = f1(a1,s)-f1(b1,s) + f1(a2,t)-f1(b2,t) + f2(a1,a2,e) - f2(b1,b2,e);
                                delta += delta_s(a1,b1);
                                delta += delta_t(a2,b2);
                                if(delta>=eps) { // elliminate (a1,a2) and make neighbours active
                                    //std::cout<<"map: ("<<a1<<","<<a2<<")->("<<b1<<","<<b2<<") on edge "<<e<<"\n";
                                    //X(a1,s) = 0;
                                    //X(a2,t) = 0;
                                    //P(a1,s) = b1;
                                    //P(a2,t) = b2;
                                    //++eliminated;
                                    //++eliminated;
                                    if(eX(a1,a2,e)) {
                                        eX(a1,a2,e) = 0;
                                        ++eliminated2;
                                        activate(s);
                                        activate(t);
                                    };
                                    //return;
                                };
                            };
                        };
                    };
                };
            };
        };
       // mexPrintf("elim2: %i (%3.2f%%)\n",eliminated2,float(eliminated2)/((K*K-1)*nE));
    };
};


long long dee2(const num_array<int,2>  & E, const num_array<int,2> & f1, const num_array<int,3> & f2, num_array<int,2> & X, num_array<int,2> & P, double eps) {
    dee2_compute alg(E,f1,f2,X,P,eps);
    alg.compute();
    return alg.eliminated;
};
