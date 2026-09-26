import numpy as np
from sklearn.metrics import f1_score,roc_auc_score,average_precision_score,brier_score_loss

def events(y,p,threshold=.5,horizon=2,previous=None):
    """Bounded one-to-one matching; warnings only while currently non-backordered.
    p[t] forecasts y[t+horizon]. Alert episodes use rising edges to avoid duplicates.
    Only fully observed alert windows and event onsets with a complete lookback count.
    """
    y=np.asarray(y,bool);p=np.asarray(p);T,N=y.shape
    prev=np.zeros(N,bool) if previous is None else np.asarray(previous,bool)
    onset=y & ~np.vstack([prev,y[:-1]])
    active=(p>=threshold)&~y
    rising=active & ~np.vstack([np.zeros(N,bool),active[:-1]])
    matched=0;false=0;eligible=0;leads=[];alerts=0
    for v in range(N):
        es=list(np.where(onset[horizon:,v])[0]+horizon);eligible+=len(es);used=set()
        for t in np.where(rising[:,v])[0]:
            if t+horizon>=T:continue
            alerts+=1
            candidates=[e for e in es if t<e<=t+horizon and e not in used]
            if candidates:
                e=candidates[0];used.add(e);matched+=1;leads.append(e-t)
            else:false+=1
    return dict(event_recall=matched/eligible if eligible else np.nan,alert_precision=matched/alerts if alerts else np.nan,false_alerts_per_100=100*false/(max(1,T-horizon)*N),ewlt=np.mean(leads) if leads else np.nan,event_count=eligible,alert_count=alerts)

def metrics(y,b,p,q,threshold=.5):
    yy=y.ravel().astype(int);pp=p.ravel();positive=y>0
    return dict(f1=f1_score(yy,pp>=threshold,average='macro',zero_division=0),auroc=roc_auc_score(yy,pp) if len(np.unique(yy))==2 else np.nan,auprc=average_precision_score(yy,pp) if yy.any() else np.nan,brier=brier_score_loss(yy,pp),rmse_pos=float(np.sqrt(np.mean((q[positive]-b[positive])**2))) if positive.any() else np.nan,rmse_all=float(np.sqrt(np.mean((p*q-b)**2))),prevalence=float(yy.mean()))

def paired_block_bootstrap(y,p1,p2,block=7,reps=1000,seed=812):
    rng=np.random.default_rng(seed);T=len(y);values=[]
    for _ in range(reps):
        starts=rng.integers(0,T,size=int(np.ceil(T/block)))
        ix=np.concatenate([(s+np.arange(block))%T for s in starts])[:T]
        a=f1_score(y[ix].ravel(),(p1[ix]>=.5).ravel(),average='macro',zero_division=0)
        b=f1_score(y[ix].ravel(),(p2[ix]>=.5).ravel(),average='macro',zero_division=0);values.append(a-b)
    return np.quantile(values,[.025,.975]).tolist()
