"""Intermittency-aware graph residual boosting (IGRB)."""
from dataclasses import dataclass
import json
import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import average_precision_score,brier_score_loss,f1_score
from xgboost import XGBClassifier,XGBRegressor
from .simulator import Inventory

def _classifier(seed,depth,trees=180):
 return XGBClassifier(n_estimators=trees,max_depth=depth,learning_rate=.05,subsample=.9,colsample_bytree=.85,min_child_weight=2,reg_lambda=2,n_jobs=1,random_state=seed,tree_method='hist')
def _regressor(seed,depth,trees=180):
 return XGBRegressor(n_estimators=trees,max_depth=depth,learning_rate=.05,subsample=.9,colsample_bytree=.85,min_child_weight=2,reg_lambda=2,n_jobs=1,random_state=seed,tree_method='hist',objective='reg:squarederror')
def _threshold(y,p):
 grid=np.linspace(.15,.85,141);scores=np.array([f1_score(y.ravel(),(p>=t).ravel(),average='macro',zero_division=0) for t in grid]);best=scores.max();ids=np.where(scores>=best-1e-12)[0];i=ids[np.argmin(abs(grid[ids]-.5))];return float(grid[i]),float(scores[i])
def _rolling_oof(data,seed,depth):
 times=data.parts['train'];n=len(times);bounds=[int(n*.4),int(n*.6),int(n*.8),n];p=np.full((n,data.net.n),np.nan);q=np.full_like(p,np.nan)
 for fold in range(3):
  cut,end=bounds[fold],bounds[fold+1];fit=times[:cut];pred=times[cut:end];X=data.tabular(fit);y=data.y[fit+2].ravel();b=data.b[fit+2].ravel();c=_classifier(seed+fold,depth,120).fit(X,y);r=_regressor(seed+fold,depth,120).fit(X[y>0],b[y>0]);V=data.tabular(pred);p[cut:end]=c.predict_proba(V)[:,1].reshape(len(pred),data.net.n);q[cut:end]=np.maximum(0,r.predict(V)).reshape(len(pred),data.net.n)
 return p,q
def _allocation_proxy(data,times,p,q,threshold):
 leaves=data.net.leaves;mu=data.net.mean;budget=.15*mu[leaves].sum();truth=data.b[np.asarray(times)+2];cost=0.
 for i in range(len(times)):
  rem=budget;action=np.zeros(data.net.n)
  for v in leaves[np.argsort(-p[i,leaves],kind='stable')]:
   if p[i,v]<threshold:continue
   a=min(rem,max(0,q[i,v]*mu[v]));action[v]=a;rem-=a
   if rem<=1e-12:break
  backlog=truth[i,leaves]*mu[leaves];cost+=5*np.maximum(0,backlog-action[leaves]).sum()+2*action[leaves].sum()
 return cost/max(len(times),1)

@dataclass
class IGRBModel:
 local_classifier:object;local_regressor:object;graph_classifier:object;graph_regressor:object
 alpha:float;threshold:float;local_threshold:float;local_depth:int;graph_depth:int;policy_threshold:float=.5;policy_scale:float=1.
 def predict_arrays(self,L,G,shape):
  p0=self.local_classifier.predict_proba(L)[:,1];q0=np.maximum(0,self.local_regressor.predict(L));pg=self.graph_classifier.predict_proba(G)[:,1];qg=np.maximum(0,self.graph_regressor.predict(G));p=expit(logit(np.clip(p0,1e-5,1-1e-5))+self.alpha*(logit(np.clip(pg,1e-5,1-1e-5))-logit(np.clip(p0,1e-5,1-1e-5))));q=np.maximum(0,q0+self.alpha*(qg-q0));return p.reshape(shape),q.reshape(shape)
 def predict(self,data,times):return self.predict_arrays(data.tabular(times),data.graph_tabular(times),(len(times),data.net.n))
 def predict_window(self,data,window):
  x=np.asarray(window,dtype=np.float32)[None];L=x.transpose(0,2,1,3).reshape(data.net.n,-1);return self.predict_arrays(L,data.graph_tabular_windows(x),(data.net.n,))
 def local_only(self):return IGRBModel(self.local_classifier,self.local_regressor,self.graph_classifier,self.graph_regressor,0.,self.local_threshold,self.local_threshold,self.local_depth,self.graph_depth)
 def metadata(self):return dict(alpha=self.alpha,threshold=self.threshold,local_threshold=self.local_threshold,local_depth=self.local_depth,graph_depth=self.graph_depth,policy_threshold=self.policy_threshold,policy_scale=self.policy_scale)

def validation_policy_cost(model,data,threshold,scale,start=None,end=None):
 sim=data.sim;T=len(sim['x']);start=sim['c1'] if start is None else int(start);end=sim['c2'] if end is None else int(end);env=Inventory(data.net,T,sim['seed']);actions=np.zeros((T+2,data.net.n));xs=[];leaves=data.net.leaves;budget=.15*data.net.mean[leaves].sum();cost=0.
 for t,demand in enumerate(sim['demand']):
  x,e,_,ledger=env.step(demand,actions[t]);xx,_=data.transform_online(x,e,t);xs.append(xx)
  if start<=t<end-2 and scale>0 and len(xs)>=data.window:
   p,q=model.predict_window(data,np.asarray(xs[-data.window:]));rem=budget
   for v in leaves[np.argsort(-p[leaves],kind='stable')]:
    if p[v]<threshold:continue
    a=min(rem,max(0,scale*q[v]*data.net.mean[v]));actions[t+2,v]=a;rem-=a
    if rem<=1e-12:break
  if start<=t<end:cost+=.1*ledger['holding']+5*ledger['backlog']+2*ledger['expedite']
  if t>=end-1:break
 return float(cost)
def tune_policy(model,data):
 choices=[(model.threshold,0.)]+[(t,s) for s in [.25,.5,1.] for t in sorted(set([model.threshold,.5,.65,.8]))];scored=[(validation_policy_cost(model,data,t,s),t,s) for t,s in choices];base=scored[0];mid=(data.sim['c1']+data.sim['c2'])//2;basehalves=[validation_policy_cost(model,data,base[1],0.,data.sim['c1'],mid),validation_policy_cost(model,data,base[1],0.,mid,data.sim['c2'])];eligible=[]
 for c,t,s in scored[1:]:
  halves=[validation_policy_cost(model,data,t,s,data.sim['c1'],mid),validation_policy_cost(model,data,t,s,mid,data.sim['c2'])]
  if c<base[0]*.99 and all(a<b*.99 for a,b in zip(halves,basehalves)):eligible.append((c,t,s))
 cost,t,s=min(eligible,key=lambda z:(z[0],z[2],z[1])) if eligible else base;model.policy_threshold=float(t);model.policy_scale=float(s);return dict(validation_policy_cost=cost,policy_threshold=t,policy_scale=s,candidates=[dict(cost=c,threshold=t,scale=s) for c,t,s in scored])

def fit_igrb(data,seed,out=None,use_adi=True,error_weighting=False):
 tr,va=data.parts['train'],data.parts['val'];y=data.y[tr+2].ravel();b=data.b[tr+2].ravel();yv=data.y[va+2];X,V=data.tabular(tr),data.tabular(va);G,GV=data.graph_tabular(tr),data.graph_tabular(va);logs=[];best=None
 for depth in [3,6]:
  c=_classifier(seed,depth).fit(X,y);r=_regressor(seed,depth).fit(X[y>0],b[y>0]);p=c.predict_proba(V)[:,1].reshape(len(va),data.net.n);q=np.maximum(0,r.predict(V)).reshape(len(va),data.net.n);t,f=_threshold(yv,p);ap=average_precision_score(yv.ravel(),p.ravel());score=f+.05*ap;logs.append(dict(stage='local',depth=depth,threshold=t,f1=f,auprc=ap,score=score));
  if best is None or score>best[0]:best=(score,depth,c,r,p,q,t)
 _,ld,lc,lr,p0,q0,lt=best;sw=np.ones((len(tr),data.net.n));rw=np.ones_like(sw)
 if error_weighting:
  oofp,oofq=_rolling_oof(data,seed,ld);mask=np.isfinite(oofp[:,0]);sw[mask]=1+2*np.abs(data.y[tr[mask]+2]-oofp[mask]);rw[mask]=1+np.minimum(3,np.abs(data.b[tr[mask]+2]-oofq[mask]))
 localcost=_allocation_proxy(data,va,p0,q0,lt);localf=f1_score(yv.ravel(),(p0>=lt).ravel(),average='macro');localap=average_precision_score(yv.ravel(),p0.ravel());localbr=brier_score_loss(yv.ravel(),p0.ravel());localobj=localf+.05*localap-.02*localbr-.03;candidates=[]
 for depth in [2,4]:
  gc=_classifier(seed+101,depth).fit(G,y,sample_weight=sw.ravel());pos=y>0;gr=_regressor(seed+211,depth).fit(G[pos],b[pos],sample_weight=rw.ravel()[pos]);pg=gc.predict_proba(GV)[:,1].reshape(len(va),data.net.n);qg=np.maximum(0,gr.predict(GV)).reshape(len(va),data.net.n)
  for a in np.linspace(0,1,11):
   p=expit(logit(np.clip(p0,1e-5,1-1e-5))+a*(logit(np.clip(pg,1e-5,1-1e-5))-logit(np.clip(p0,1e-5,1-1e-5))));q=np.maximum(0,q0+a*(qg-q0));t,f=_threshold(yv,p);ap=average_precision_score(yv.ravel(),p.ravel());br=brier_score_loss(yv.ravel(),p.ravel());cost=_allocation_proxy(data,va,p,q,t);obj=f+.05*ap-.02*br-.03*cost/max(localcost,1e-9);candidates.append(dict(depth=depth,alpha=float(a),threshold=t,f1=f,auprc=ap,brier=br,proxy_cost=cost,objective=obj))
 choice=max(candidates,key=lambda z:(z['objective'],-z['alpha'],-z['depth']))
 if choice['objective']<=localobj+1e-4:choice=dict(depth=2,alpha=0.,threshold=lt,objective=localobj)
 cal=data.sim['demand'][:data.sim['c1']];adi=float(np.median(len(cal)/np.maximum((cal>0).sum(0),1)))
 if use_adi and adi>1.32:choice=dict(depth=2,alpha=0.,threshold=lt,objective=localobj)
 gc=_classifier(seed+101,choice['depth']).fit(G,y,sample_weight=sw.ravel());pos=y>0;gr=_regressor(seed+211,choice['depth']).fit(G[pos],b[pos],sample_weight=rw.ravel()[pos]);model=IGRBModel(lc,lr,gc,gr,float(choice['alpha']),float(choice['threshold']),float(lt),ld,int(choice['depth']))
 if out:
  out.mkdir(parents=True,exist_ok=True);lc.save_model(out/'local_classifier.ubj');lr.save_model(out/'local_regressor.ubj');gc.save_model(out/'graph_classifier.ubj');gr.save_model(out/'graph_regressor.ubj');(out/'training.json').write_text(json.dumps(dict(selected=model.metadata(),training_adi=adi,use_adi=use_adi,error_weighting=error_weighting,local_objective=localobj,candidates=logs+candidates),indent=2))
 return model,dict(selected_alpha=model.alpha,selected_threshold=model.threshold,local_depth=ld,graph_depth=model.graph_depth,training_adi=adi,use_adi=use_adi,error_weighting=error_weighting)

def fit_iges(data,seed,out=None):
 """Final hard expert selector: graph only when dense and validation-superior."""
 tr,va=data.parts['train'],data.parts['val'];y=data.y[tr+2].ravel();b=data.b[tr+2].ravel();yv=data.y[va+2];X,V=data.tabular(tr),data.tabular(va);G,GV=data.graph_tabular(tr),data.graph_tabular(va);logs=[];best=None
 for depth in [3,6]:
  c=_classifier(seed,depth).fit(X,y);r=_regressor(seed,depth).fit(X[y>0],b[y>0]);p=c.predict_proba(V)[:,1].reshape(len(va),data.net.n);q=np.maximum(0,r.predict(V)).reshape(len(va),data.net.n);t,f=_threshold(yv,p);ap=average_precision_score(yv.ravel(),p.ravel());br=brier_score_loss(yv.ravel(),p.ravel());cost=_allocation_proxy(data,va,p,q,t);score=f+.05*ap-.02*br;logs.append(dict(stage='local',depth=depth,threshold=t,f1=f,auprc=ap,brier=br,proxy_cost=cost,score=score))
  if best is None or score>best[0]:best=(score,depth,c,r,p,q,t,cost)
 localobj,ld,lc,lr,p0,q0,lt,localcost=best;gbest=None
 for depth in [2,4]:
  gc=_classifier(seed+101,depth).fit(G,y);pos=y>0;gr=_regressor(seed+211,depth).fit(G[pos],b[pos]);p=gc.predict_proba(GV)[:,1].reshape(len(va),data.net.n);q=np.maximum(0,gr.predict(GV)).reshape(len(va),data.net.n);t,f=_threshold(yv,p);ap=average_precision_score(yv.ravel(),p.ravel());br=brier_score_loss(yv.ravel(),p.ravel());cost=_allocation_proxy(data,va,p,q,t);score=f+.05*ap-.02*br-.03*cost/max(localcost,1e-9);logs.append(dict(stage='graph',depth=depth,threshold=t,f1=f,auprc=ap,brier=br,proxy_cost=cost,score=score))
  if gbest is None or score>gbest[0]:gbest=(score,depth,gc,gr,t)
 gobj,gd,gc,gr,gt=gbest;cal=data.sim['demand'][:data.sim['c1']];adi=float(np.median(len(cal)/np.maximum((cal>0).sum(0),1)));use_graph=bool(adi<=1.32 and gobj>localobj-.03+1e-4);alpha=1. if use_graph else 0.;threshold=gt if use_graph else lt;model=IGRBModel(lc,lr,gc,gr,alpha,float(threshold),float(lt),ld,gd)
 if out:
  out.mkdir(parents=True,exist_ok=True);lc.save_model(out/'local_classifier.ubj');lr.save_model(out/'local_regressor.ubj');gc.save_model(out/'graph_classifier.ubj');gr.save_model(out/'graph_regressor.ubj');(out/'training.json').write_text(json.dumps(dict(selected=model.metadata(),training_adi=adi,use_graph=use_graph,local_score=localobj,graph_score=gobj,candidates=logs),indent=2))
 return model,dict(selected_alpha=alpha,selected_threshold=threshold,local_depth=ld,graph_depth=gd,training_adi=adi,use_graph=use_graph)
