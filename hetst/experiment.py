import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time,copy,random,platform,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import f1_score
from scipy.special import expit
from scipy.stats import norm
from xgboost import XGBClassifier,XGBRegressor
from .simulator import synthetic,simulate,Inventory
from .data import Prepared,retail_demands
from .models import Model
from .evaluation import metrics,events,paired_block_bootstrap

torch.set_num_threads(1)

def loss(logit,q,y,b,kind):
    pos=y.sum();neg=y.numel()-pos
    # Weights are fixed from training in fit_neural, not recomputed per batch.
    weights=torch.where(y>0,loss.wp,loss.wn)
    cls=(torch.nn.functional.binary_cross_entropy_with_logits(logit,y,reduction='none')*weights).mean()
    reg=((q[y>0]-b[y>0])**2).mean() if (y>0).any() else q.sum()*0
    if kind=='ClsOnly':return cls
    if kind=='RegOnly':return reg
    return cls+reg

def fit_neural(kind,d,seed,out,epochs=50,lrs=(.002,.0006),width=16,heads=2):
    tr=d.parts['train'];va=d.parts['val'];tx,te,ty,tb=d.batch(tr);vx,ve,vy,vb=d.batch(va)
    pos=float(ty.sum());total=ty.numel();loss.wp=total/(2*max(pos,1));loss.wn=total/(2*max(total-pos,1))
    best=None;bestval=float('inf');logs=[];tic=time.perf_counter()
    for lr in lrs:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=Model(kind,width=width,heads=heads);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        local=float('inf');pat=0;history=[]
        for ep in range(epochs):
            model.train();perm=torch.randperm(len(tx));tl=[]
            for ids in perm.split(64):
                opt.zero_grad();a,q=model(tx[ids],te[ids],*d.graph);l=loss(a,q,ty[ids],tb[ids],kind);l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();tl.append(float(l.detach()))
            model.eval()
            with torch.no_grad():a,q=model(vx,ve,*d.graph);vl=float(loss(a,q,vy,vb,kind));vf=f1_score(vy.numpy().ravel(),(a.numpy().ravel()>0),average='macro',zero_division=0)
            history.append(dict(epoch=ep+1,train_loss=float(np.mean(tl)),val_loss=vl,val_f1=vf,elapsed=time.perf_counter()-tic))
            if vl<local-1e-5:
                local=vl;pat=0;state=copy.deepcopy(model.state_dict());chosen_ep=ep+1
            else:pat+=1
            if pat>=8:break
        logs.append(dict(lr=lr,epochs=history,best_epoch=chosen_ep,best_val=local))
        if local<bestval:bestval=local;best=state;bestlr=lr;bestep=chosen_ep
    model.load_state_dict(best);model.eval()
    seconds=time.perf_counter()-tic
    torch.save(dict(state=best,kind=kind,width=width,heads=heads,lr=bestlr,seed=seed),out/'model.pt')
    (out/'training.json').write_text(json.dumps(logs,indent=2))
    return model,dict(train_seconds=seconds,selected_lr=bestlr,best_epoch=bestep,parameters=sum(p.numel() for p in model.parameters()))

def fit_xgb(d,seed,out):
    tr=d.parts['train'];va=d.parts['val'];X=d.tabular(tr);V=d.tabular(va);y=d.y[tr+2].ravel();b=d.b[tr+2].ravel();yv=d.y[va+2].ravel();bv=d.b[va+2].ravel()
    best=None;bestval=np.inf;tic=time.perf_counter();logs=[]
    for depth in [3,6]:
        clf=XGBClassifier(n_estimators=120,max_depth=depth,learning_rate=.05,subsample=.9,colsample_bytree=.9,n_jobs=1,random_state=seed,tree_method='hist')
        reg=XGBRegressor(n_estimators=120,max_depth=depth,learning_rate=.05,subsample=.9,colsample_bytree=.9,n_jobs=1,random_state=seed,tree_method='hist')
        clf.fit(X,y);reg.fit(X[y>0],b[y>0]);p=clf.predict_proba(V)[:,1];q=np.maximum(0,reg.predict(V))
        vp=yv.mean();w=np.where(yv>0,1/(2*y.mean()),1/(2*(1-y.mean())))
        val=-(w*(yv*np.log(p.clip(1e-7,1))+(1-yv)*np.log((1-p).clip(1e-7,1)))).mean()+((q[yv>0]-bv[yv>0])**2).mean()
        logs.append(dict(depth=depth,val_loss=float(val)))
        if val<bestval:bestval=val;best=(clf,reg);chosen=depth
    best[0].save_model(out/'classifier.ubj');best[1].save_model(out/'regressor.ubj');(out/'training.json').write_text(json.dumps(logs,indent=2))
    return best,dict(train_seconds=time.perf_counter()-tic,selected_depth=chosen)

def predict(kind,model,d,times):
    if kind=='Persistence':return d.y[times].copy(),d.b[times].copy()
    if kind=='LocalBalance':
        raw=d.sim['x'][times];z=raw[:,:,1]+2*d.net.mean-raw[:,:,0];sd=np.sqrt(2)*np.maximum(d.net.std,.5)
        return norm.cdf(z/sd),np.maximum(0,z)/d.net.mean
    if kind=='XGBoost':
        X=d.tabular(times);shape=(len(times),d.net.n)
        return model[0].predict_proba(X)[:,1].reshape(shape),np.maximum(0,model[1].predict(X)).reshape(shape)
    ps=[];qs=[]
    with torch.no_grad():
        for chunk in np.array_split(times,max(1,int(np.ceil(len(times)/64)))):
            x,e,_,_=d.batch(chunk);a,q=model(x,e,*d.graph);ps.append(torch.sigmoid(a).numpy());qs.append(q.numpy())
    return np.concatenate(ps),np.concatenate(qs)

def online_pred(kind,model,d,xs,es):
    w=d.window;x=np.array(xs[-w:]);e=np.array(es[-w:]);N=d.net.n
    if kind=='XGBoost':
        X=x.transpose(1,0,2).reshape(N,-1);return model[0].predict_proba(X)[:,1],np.maximum(0,model[1].predict(X))
    with torch.no_grad():a,q=model(torch.tensor(x[None]),torch.tensor(e[None]),*d.graph)
    return torch.sigmoid(a)[0].numpy(),q[0].numpy()

def closed_loop(kind,model,d,start=None):
    """Fixed policy: reserve <=15% of calibrated leaf demand/cycle, arrive at t+2.
    Policy is fixed before test and never tuned against test costs.
    """
    sim=d.sim;start=sim['c2'] if start is None else start;T=len(sim['x']);env=Inventory(d.net,T,sim['seed'],shift_at=sim['c2'],shift=sim['shift'])
    xs=[];es=[];actions=np.zeros((T+2,d.net.n));records=[];leaves=d.net.leaves
    budget=.15*d.net.mean[leaves].sum()
    for t,dem in enumerate(sim['demand']):
        x,e,b,l=env.step(dem,actions[t]);xx,ee=d.transform_online(x,e,t);xs.append(xx);es.append(ee)
        if start<=t<T-2 and kind!='NoAction':
            if kind=='Persistence':p=(b>1e-6).astype(float);q=b/d.net.mean
            elif kind=='Uniform':p=np.ones(d.net.n);q=np.zeros(d.net.n);q[leaves]=budget/len(leaves)/d.net.mean[leaves]
            elif kind=='LocalBalance':
                z=b+2*d.net.mean-x[:,0];p=norm.cdf(z/(np.sqrt(2)*np.maximum(d.net.std,.5)));q=np.maximum(0,z)/d.net.mean
            else:p,q=online_pred(kind,model,d,xs,es)
            remaining=budget
            # rank by predicted occurrence; cap purchase by conditional magnitude
            for v in leaves[np.argsort(-p[leaves],kind='stable')]:
                if p[v]<.5:continue
                a=min(remaining,max(0,q[v]*d.net.mean[v]));actions[t+2,v]=a;remaining-=a
        if t>=start:
            l.update(t=t,cost=.1*l['holding']+5*l['backlog']+2*l['expedite']);records.append(l)
    frame=pd.DataFrame(records)
    return dict(cost=float(frame.cost.sum()),fill_rate=float(frame.served_current.sum()/max(frame.demand.sum(),1e-8)),backlog_unit_cycles=float(frame.backlog.sum()),emergency_units=float(frame.expedite.sum()),holding_unit_cycles=float(frame.holding.sum())),frame

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='results/main');ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--seeds',default='11,23,37,53,71');ap.add_argument('--cases',default='seasonal,intermittent,correlated,retail');ap.add_argument('--methods',default='Persistence,LocalBalance,XGBoost,LSTM,GCN,RGCN,TGRU,HomGAT,RelGRU,HetST');ap.add_argument('--shifts',default='none,lead,policy');ap.add_argument('--retail-zip',default='../data_download/online_retail.zip');ap.add_argument('--width',type=int,default=16);ap.add_argument('--heads',type=int,default=2);ap.add_argument('--lr-grid',default='.002,.0006');ap.add_argument('--skip-policy',action='store_true');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);seeds=list(map(int,args.seeds.split(',')));methods=args.methods.split(',');cases=args.cases.split(',');shifts=args.shifts.split(',')
    (out/'config.json').write_text(json.dumps(vars(args),indent=2));(out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,cpu_threads=torch.get_num_threads()),indent=2))
    rows=[];policies=[]
    for ci,case in enumerate(cases):
        if case=='retail':
            cached=Path('data/retail_demand.npy')
            if cached.exists():demand=np.load(cached)
            else:demand=retail_demands(args.retail_zip,'data')
        else:demand=synthetic(case,101)
        sim=simulate(demand,seed=1001);d=Prepared(sim)
        variants={s:Prepared(simulate(demand,seed=1001,shift=s),reference=d) for s in shifts}
        dataout=out/case;dataout.mkdir(exist_ok=True)
        np.savez_compressed(dataout/'data.npz',demand=demand,x=sim['x'],edge=sim['edge'],backlog=sim['b'],parent=d.net.parent,mean=d.net.mean,std=d.net.std)
        (dataout/'split.json').write_text(json.dumps({k:v.tolist() for k,v in d.parts.items()}))
        (dataout/'normalization.json').write_text(json.dumps({k:getattr(d,k).tolist() for k in ['mean','std','em','es']}))
        for seed in seeds:
            for method in methods:
                run=dataout/f'{method}_{seed}';run.mkdir(exist_ok=True)
                if (run/'complete.json').exists():
                    saved=json.loads((run/'complete.json').read_text());rows+=saved['rows'];policies+=saved['policies'];continue
                print(f'START {case} {method} seed={seed}',flush=True)
                if method=='XGBoost':model,info=fit_xgb(d,seed,run)
                elif method in ['Persistence','LocalBalance']:model=None;info=dict(train_seconds=0)
                else:model,info=fit_neural(method,d,seed,run,args.epochs,tuple(map(float,args.lr_grid.split(','))),args.width,args.heads)
                current=[];cp=[]
                for shift,ds in variants.items():
                    ts=ds.parts['test'];p,q=predict(method,model,ds,ts);yy=ds.y[ts+2];bb=ds.b[ts+2];m=metrics(yy,bb,p,q)
                    if method=='ClsOnly':m['rmse_pos']=m['rmse_all']=np.nan
                    if method=='RegOnly':
                        for key in ['f1','auroc','auprc','brier']:m[key]=np.nan
                    pp=np.r_[p,np.zeros((2,ds.net.n))];ev=events(ds.y[ds.sim['c2']:],pp,previous=ds.y[ds.sim['c2']-1])
                    if method=='RegOnly':ev={k:np.nan for k in ev}
                    r=dict(case=case,method=method,seed=seed,shift=shift,**m,**ev,**info);current.append(r)
                    np.savez_compressed(run/f'predictions_{shift}.npz',time=ts,y=yy,b=bb,p=p,q=q)
                    if not args.skip_policy and method not in ['ClsOnly','RegOnly']:
                        pm,ledger=closed_loop(method,model,ds);cp.append(dict(case=case,method=method,seed=seed,shift=shift,**pm));ledger.to_csv(run/f'policy_{shift}.csv',index=False)
                    print(f'RESULT {case} {method} {shift} F1={m["f1"]:.3f} RMSE+={m["rmse_pos"]:.3f}',flush=True)
                rows+=current;policies+=cp;(run/'complete.json').write_text(json.dumps(dict(rows=current,policies=cp),indent=2))
                pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(policies).to_csv(out/'policy.csv',index=False)
        if not args.skip_policy:
            for shift,ds in variants.items():
                for method in ['NoAction','Uniform']:
                    pm,ledger=closed_loop(method,None,ds);policies.append(dict(case=case,method=method,seed=0,shift=shift,**pm));ledger.to_csv(dataout/f'policy_{method}_{shift}.csv',index=False)
        pd.DataFrame(policies).to_csv(out/'policy.csv',index=False)
    print('COMPLETED',out,flush=True)

if __name__=='__main__':main()
