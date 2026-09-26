import numpy as np,pandas as pd
from pathlib import Path
def ci(x,reps=30000):
 r=np.random.default_rng(1907);x=np.asarray(x);return np.quantile(r.choice(x,(reps,len(x)),replace=True).mean(1),[.025,.975])
old=pd.concat([pd.read_csv(f'results/igrb_gen_{x}/metrics.csv') for x in ['s','i','c']],ignore_index=True);new=pd.concat([pd.read_csv('results/igrb_confirm_s/metrics.csv'),pd.read_csv('results/igrb_confirm_c/metrics.csv')],ignore_index=True);allx=pd.concat([old,new],ignore_index=True);p=allx.pivot_table(index=['case','shift','demand_seed'],columns='method',values='f1').reset_index();p['delta']=p.IGRB-p.LocalXGBTuned;rows=[]
for (case,shift),f in p.groupby(['case','shift']):
 lo,hi=ci(f.delta);rows.append(dict(case=case,shift=shift,n=len(f),local_f1=f.LocalXGBTuned.mean(),igrb_f1=f.IGRB.mean(),delta=f.delta.mean(),ci_low=lo,ci_high=hi,wins=int((f.delta>1e-12).sum()),ties=int((abs(f.delta)<=1e-12).sum()),losses=int((f.delta<-1e-12).sum())))
out=pd.DataFrame(rows);Path('results/igrb_analysis').mkdir(exist_ok=True);out.to_csv('results/igrb_analysis/independent_effects_all.csv',index=False);allx.to_csv('results/igrb_analysis/independent_metrics_all.csv',index=False);print(out.to_string(index=False))
