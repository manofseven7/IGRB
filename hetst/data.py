from pathlib import Path
import json,hashlib
import numpy as np
import torch

def retail_demands(zip_path,out,nleaf=12):
    """Demand drivers only: unrelated SKUs are scaled into a synthetic single commodity.
    SKU selection and scales use the first 60% of dates exclusively.
    """
    import pandas as pd,zipfile
    with zipfile.ZipFile(zip_path) as z:
        name=next(n for n in z.namelist() if n.endswith('.xlsx'))
        with z.open(name) as f:df=pd.read_excel(f)
    raw_rows=len(df)
    df=df[(df.Quantity>0)&(df.UnitPrice>0)&~df.InvoiceNo.astype(str).str.upper().str.startswith('C')].copy()
    df['day']=pd.to_datetime(df.InvoiceDate).dt.floor('D')
    dates=pd.date_range(df.day.min(),df.day.max(),freq='D');cut=int(.6*len(dates))
    train=df[df.day<dates[cut]]
    chosen=train.groupby('StockCode').Quantity.sum().sort_values(ascending=False).head(nleaf).index.tolist()
    mat=df[df.StockCode.isin(chosen)].pivot_table(index='day',columns='StockCode',values='Quantity',aggfunc='sum').reindex(index=dates,columns=chosen).fillna(0).values
    scale=mat[:cut].mean(0).clip(.1);mat=mat/scale*8
    out=Path(out);out.mkdir(exist_ok=True,parents=True);np.save(out/'retail_demand.npy',mat)
    meta=dict(source='UCI Online Retail',doi='10.24432/C5BW33',license='CC BY 4.0',sha256=hashlib.sha256(Path(zip_path).read_bytes()).hexdigest(),raw_rows=raw_rows,filtered_rows=len(df),dates=[str(dates[0].date()),str(dates[-1].date())],days=len(dates),sku=[str(s) for s in chosen],train_scale=scale.tolist(),interpretation='Real sales demand drivers; synthetic interchangeable commodity, network, inventory and labels. Not observed back-orders.')
    (out/'retail_provenance.json').write_text(json.dumps(meta,indent=2));return mat

class Prepared:
    def __init__(self,sim,reference=None,window=6,horizon=2):
        self.sim=sim;self.net=sim['net'];self.window=window;self.horizon=horizon
        raw=sim['x'].copy();raw[...,[0,1,2,4]]/=self.net.mean[None,:,None]
        end=sim['c1'];start=20
        if reference is None:
            self.mean=raw[start:end].mean((0,1));self.std=raw[start:end].std((0,1)).clip(.01)
            self.em=sim['edge'][start:end].mean((0,1));self.es=sim['edge'][start:end].std((0,1)).clip(.01)
        else:self.mean,self.std,self.em,self.es=reference.mean,reference.std,reference.em,reference.es
        T,N,_=raw.shape;t=np.arange(T);pos=np.zeros((T,16));freq=10000**(np.arange(0,16,2)/16)
        pos[:,0::2]=np.sin(t[:,None]/freq);pos[:,1::2]=np.cos(t[:,None]/freq)
        cyc=np.stack([np.sin(2*np.pi*t/28),np.cos(2*np.pi*t/28)],-1)
        self.context=np.c_[pos,cyc];self.x=np.concatenate([(raw-self.mean)/self.std,np.broadcast_to(self.context[:,None,:],(T,N,18))],-1).astype('float32')
        self.edge=((sim['edge']-self.em)/self.es).astype('float32')
        self.b=(sim['b']/self.net.mean).astype('float32');self.y=(self.b>1e-6).astype('float32')
        self.times=np.arange(max(start,window-1),T-horizon)
        self.parts={'train':self.times[self.times+horizon<sim['c1']],'val':self.times[(self.times+horizon>=sim['c1'])&(self.times+horizon<sim['c2'])],'test':self.times[self.times>=sim['c2']]}
        src,dst,rel=self.net.edges;self.graph=[torch.tensor(a,dtype=torch.long) for a in [src,dst,rel]]
    def batch(self,times):
        ix=np.asarray(times)[:,None]+np.arange(-self.window+1,1)[None,:]
        return tuple(torch.from_numpy(a) for a in [self.x[ix],self.edge[ix],self.y[np.asarray(times)+self.horizon],self.b[np.asarray(times)+self.horizon]])
    def tabular(self,times):
        # All methods have causal states; tree baseline receives the complete six-cycle local history.
        x=self.batch(times)[0].numpy();return x.transpose(0,2,1,3).reshape(-1,x.shape[1]*x.shape[-1])
    def graph_tabular(self,times):
        """Causal topology-aware histories for the graph residual expert."""
        ix=np.asarray(times)[:,None]+np.arange(-self.window+1,1)[None,:]
        return self.graph_tabular_windows(self.x[ix])
    def graph_tabular_windows(self,x):
        x=np.asarray(x);B,W,N,D=x.shape;p=self.net.parent;features=[]
        for v in range(N):
            local=x[:,:,v]
            parent=local if p[v]<0 else x[:,:,p[v]]
            children=np.where(p==v)[0]
            child=local if len(children)==0 else x[:,:,children].mean(2)
            siblings=np.where(p==p[v])[0] if p[v]>=0 else np.array([v])
            siblings=siblings[siblings!=v]
            sibling=local if len(siblings)==0 else x[:,:,siblings].mean(2)
            global_leaf=x[:,:,self.net.leaves].mean(2)
            z=np.concatenate([local,parent,child,sibling,global_leaf,local-parent,local-sibling],-1)
            features.append(z.reshape(B,-1))
        return np.stack(features,1).reshape(B*N,-1)
    def transform_online(self,x,edge,t):
        raw=x.copy();raw[:,[0,1,2,4]]/=self.net.mean[:,None]
        return np.c_[(raw-self.mean)/self.std,np.tile(self.context[t],(len(raw),1))].astype('float32'),((edge-self.em)/self.es).astype('float32')
