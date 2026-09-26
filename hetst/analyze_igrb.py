from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier,XGBRegressor
from .simulator import synthetic,simulate
from .data import Prepared
from .evaluation import metrics
ROOT=Path('results');OUT=ROOT/'igrb_analysis';OUT.mkdir(parents=True,exist_ok=True);CASES=['seasonal','intermittent','correlated','retail']
def ci(x,reps=20000):
 x=np.asarray(x);r=np.random.default_rng(907);return np.quantile(r.choice(x,(reps,len(x)),replace=True).mean(1),[.025,.975])
main=pd.read_csv(ROOT/'igrb_final/metrics.csv');general=pd.concat([pd.read_csv(ROOT/f'igrb_gen_{x}/metrics.csv') for x in ['s','i','c']],ignore_index=True);policy=pd.read_csv(ROOT/'igrb_final/policy.csv');old=pd.concat([pd.read_csv(ROOT/c/'metrics.csv') for c in CASES])
summary=main.groupby(['case','shift','method']).agg(f1=('f1','mean'),f1_sd=('f1','std'),auprc=('auprc','mean'),brier=('brier','mean'),rmse_pos=('rmse_pos','mean'),alpha=('selected_alpha','mean'),adi=('training_adi','mean')).reset_index();summary.to_csv(OUT/'main_summary.csv',index=False)
def paired(data,index):
 p=data.pivot_table(index=index,columns='method',values='f1').reset_index();p['delta']=p.IGRB-p.LocalXGBTuned;rows=[]
 for keys,f in p.groupby(index[:-1]):
  keys=(keys,) if not isinstance(keys,tuple) else keys;lo,hi=ci(f.delta);rows.append(dict(zip(index[:-1],keys),n=len(f),local_f1=f.LocalXGBTuned.mean(),igrb_f1=f.IGRB.mean(),delta=f.delta.mean(),ci_low=lo,ci_high=hi,wins=int((f.delta>1e-12).sum()),ties=int((abs(f.delta)<=1e-12).sum()),losses=int((f.delta<-1e-12).sum())))
 return pd.DataFrame(rows)
me=paired(main,['case','shift','seed']);ie=paired(general,['case','shift','demand_seed']);me.to_csv(OUT/'main_effects.csv',index=False);ie.to_csv(OUT/'independent_effects.csv',index=False)
oldn=old[old['shift']=='none'].groupby(['case','method']).f1.mean().reset_index();best=oldn.loc[oldn.groupby('case').f1.idxmax()].rename(columns={'method':'previous_best','f1':'previous_f1'});new=summary.query("shift=='none' and method=='IGRB'")[['case','f1']].rename(columns={'f1':'igrb_f1'});comp=best.merge(new,on='case');comp['delta']=comp.igrb_f1-comp.previous_f1;comp.to_csv(OUT/'previous_best_comparison.csv',index=False)
ps=policy.groupby(['case','shift','method']).agg(cost=('cost','mean'),cost_sd=('cost','std'),policy_scale=('policy_scale','mean'),fill_rate=('fill_rate','mean')).reset_index();ps.to_csv(OUT/'policy_summary.csv',index=False)
# Graph-only control from the fitted final graph expert.
grows=[]
for case in CASES:
 demand=np.load('data/retail_demand.npy') if case=='retail' else synthetic(case,101);data=Prepared(simulate(demand,seed=1001));va,ts=data.parts['val'],data.parts['test']
 for seed in [11,23,37,53,71]:
  run=ROOT/'igrb_final'/case/f'IGRB_{seed}';gc=XGBClassifier();gr=XGBRegressor();gc.load_model(run/'graph_classifier.ubj');gr.load_model(run/'graph_regressor.ubj');pv=gc.predict_proba(data.graph_tabular(va))[:,1].reshape(len(va),data.net.n);grid=np.linspace(.15,.85,141);s=np.array([f1_score(data.y[va+2].ravel(),(pv>=t).ravel(),average='macro',zero_division=0) for t in grid]);ids=np.where(s>=s.max()-1e-12)[0];thr=float(grid[ids[np.argmin(abs(grid[ids]-.5))]]);p=gc.predict_proba(data.graph_tabular(ts))[:,1].reshape(len(ts),data.net.n);q=np.maximum(0,gr.predict(data.graph_tabular(ts))).reshape(len(ts),data.net.n);grows.append(dict(case=case,variant='GraphOnly',seed=seed,**metrics(data.y[ts+2],data.b[ts+2],p,q,thr)))
base=main.query("shift=='none'");local=base.query("method=='LocalXGBTuned'").assign(variant='LocalOnly');igrb=base.query("method=='IGRB'").assign(variant='IGRB');err=pd.read_csv(ROOT/'igrb_errorweighted/metrics.csv').query("method=='IGRB'").assign(variant='ErrorWeighted');noadi=pd.read_csv(ROOT/'igrb_noadi/metrics.csv').query("method=='IGRB'").assign(variant='NoADI');ab=pd.concat([local,igrb,pd.DataFrame(grows),err,noadi],ignore_index=True,sort=False);ab.to_csv(OUT/'ablation_metrics.csv',index=False);absum=ab.groupby(['case','variant']).agg(f1=('f1','mean'),f1_sd=('f1','std')).reset_index();absum.to_csv(OUT/'ablation_summary.csv',index=False)
(OUT/'summary.json').write_text(json.dumps(dict(main=summary.to_dict('records'),independent=ie.to_dict('records'),comparison=comp.to_dict('records'),policy=ps.to_dict('records'),ablation=absum.to_dict('records')),indent=2));print(ie.query("shift=='none'").to_string(index=False));print(comp.to_string(index=False))
if __name__=='__main__':pass
