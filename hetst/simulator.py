"""Causal, single-commodity, periodic-review tree inventory simulator."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Network:
    parent: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    @property
    def n(self): return len(self.parent)
    @property
    def leaves(self): return np.array([v for v in range(self.n) if not np.any(self.parent==v)])
    @property
    def edges(self):
        child=np.arange(1,self.n); par=self.parent[child]
        return np.r_[par,child],np.r_[child,par],np.r_[np.zeros(len(child),int),np.ones(len(child),int)]


def tree(nleaf, demand, calibration_end, seed=0):
    """Three echelons; assignment fixed before experiments, not inferred from data."""
    nmid=max(2,int(np.sqrt(nleaf))); parent=np.r_[-1,np.zeros(nmid,int),1+np.arange(nleaf)%nmid]
    n=len(parent); mean=np.zeros(n); std=np.zeros(n)
    calibration=np.zeros((calibration_end,n)); calibration[:,-nleaf:]=demand[:calibration_end]
    for v in range(n-1,0,-1): calibration[:,parent[v]]+=calibration[:,v]
    mean[:]=np.maximum(calibration.mean(0),.1); std[:]=calibration.std(0)
    return Network(parent,mean,std)


def synthetic(kind,seed,T=480,nleaf=12):
    rng=np.random.default_rng(seed); t=np.arange(T); rates=rng.uniform(5,12,nleaf)
    seasonal=1+.35*np.sin(2*np.pi*t[:,None]/28+rng.uniform(0,1,nleaf))
    if kind=='seasonal': demand=rng.poisson(rates*seasonal)
    elif kind=='intermittent': demand=rng.poisson(rates*seasonal/.4)*(rng.random((T,nleaf))<.4)
    elif kind=='correlated':
        shock=np.zeros(T)
        for i in range(1,T): shock[i]=.85*shock[i-1]+rng.normal(0,.22)
        demand=rng.poisson(rates*seasonal*np.exp(shock[:,None]-.1))
    else: raise ValueError(kind)
    return demand.astype(float)


class Inventory:
    def __init__(self,net,T,seed,z=1.0,lead=2,shift_at=None,shift='none'):
        self.net=net; self.T=T; self.lead=lead; self.shift_at=shift_at; self.shift=shift
        self.rng=np.random.default_rng(seed)
        self.S=net.mean*(lead+1)+z*net.std*np.sqrt(lead+1)
        self.inv=self.S.copy(); self.pending=np.zeros(net.n)
        self.customer=np.zeros(net.n); self.transit=np.zeros((T+20,net.n))
        self.disruption=self.rng.random((T,net.n))<.04
        self.t=0; self.external_received=0.; self.customer_served=0.; self.initial_stock=self.inv.sum()
    def step(self,leaf_demand,expedite=None):
        n=self.net.n;t=self.t; mu=self.net.mean;p=self.net.parent
        changed=self.shift_at is not None and t>=self.shift_at
        lead=self.lead+(1 if changed and self.shift=='lead' else 0)
        zfactor=.65 if changed and self.shift=='policy' else 1.
        arrivals=self.transit[t].copy();self.inv+=arrivals;self.transit[t]=0
        self.external_received+=arrivals[0]
        exp=np.zeros(n) if expedite is None else np.maximum(np.asarray(expedite),0)
        self.inv+=exp
        # Expedites are external emergency purchases, separately costed.
        self.external_received+=exp.sum()
        demand=np.zeros(n);demand[self.net.leaves]=leaf_demand
        backlog=np.zeros(n);orders=np.zeros(n);flows=np.zeros(n);served=np.zeros(n)
        current_service=np.zeros(n)
        for v in range(n-1,-1,-1):
            children=np.where(p==v)[0]
            if len(children):
                need=self.pending[children].sum();old=max(0,need-demand[v])
            else: need=self.customer[v]+demand[v];old=self.customer[v]
            cap=mu[v]*1.6*(.25 if self.disruption[t,v] else 1.)
            delivered=min(self.inv[v],need,cap);self.inv[v]-=delivered;served[v]=delivered
            current_service[v]=max(0,delivered-old)
            if len(children):
                send=self.pending[children]*(delivered/need if need>0 else 0)
                self.pending[children]-=send;self.transit[t+lead,children]+=send;flows[children]=send
                backlog[v]=self.pending[children].sum()
            else:
                self.customer[v]=need-delivered;backlog[v]=self.customer[v];self.customer_served+=delivered
            inbound=self.transit[t+1:,v].sum()+self.pending[v]
            ip=self.inv[v]+inbound-backlog[v]
            orders[v]=max(0,zfactor*self.S[v]-ip)
            if v>0:
                self.pending[v]+=orders[v];demand[p[v]]+=orders[v]
            else:self.transit[t+lead,v]+=orders[v]
        fill=np.divide(current_service,demand,out=np.ones(n),where=demand>0)
        # Nominal contracted lead is known, future realized arrivals are not features.
        x=np.stack([self.inv,backlog,demand,np.full(n,self.lead),orders,fill],axis=-1)
        src,dst,rel=self.net.edges; child=np.r_[np.arange(1,n),np.arange(1,n)]
        cap=mu[child]*1.6
        realized=np.r_[flows[1:],orders[1:]]/cap
        edge=np.stack([rel,cap/mu[child],realized,realized-1],-1)
        self.t+=1
        # Physical conservation excludes unreceived external supply in root pipeline.
        residual=self.initial_stock+self.external_received-self.customer_served-self.inv.sum()-self.transit[:,1:].sum()
        if abs(residual)>1e-6:raise AssertionError(('mass balance',residual))
        return x,edge,backlog,{'holding':self.inv.sum(),'backlog':self.customer.sum(),'demand':leaf_demand.sum(),'served_current':current_service[self.net.leaves].sum(),'expedite':exp.sum(),'mass_residual':residual}


def simulate(demand,seed=0,shift='none'):
    T=len(demand); c1=int(.6*T);c2=int(.8*T);net=tree(demand.shape[1],demand,c1)
    env=Inventory(net,T,seed,shift_at=c2,shift=shift)
    xs=[];es=[];bs=[];ledger=[]
    for d in demand:
        x,e,b,l=env.step(d);xs.append(x);es.append(e);bs.append(b);ledger.append(l)
    return dict(net=net,x=np.array(xs),edge=np.array(es),b=np.array(bs),ledger=ledger,demand=demand,c1=c1,c2=c2,seed=seed,shift=shift)
