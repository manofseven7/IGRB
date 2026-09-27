import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from .simulator import synthetic,simulate
from .data import Prepared
from .evaluation import metrics,events
from .hybrid import fit_iges,tune_policy
from .experiment_igrb import closed_loop
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='results/iges_final');p.add_argument('--seeds',default='11,23,37,53,71');p.add_argument('--cases',default='seasonal,intermittent,correlated,retail');p.add_argument('--shifts',default='none,lead,policy');p.add_argument('--skip-policy',action='store_true');p.add_argument('--demand-seeds',default='');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[];pol=[];mseeds=list(map(int,a.seeds.split(',')));dseeds=list(map(int,a.demand_seeds.split(','))) if a.demand_seeds else [101]
 for case in a.cases.split(','):
  for ix,dseed in enumerate(dseeds):
   seed=mseeds[ix%len(mseeds)];demand=np.load('data/retail_demand.npy') if case=='retail' else synthetic(case,dseed);simseed=1001 if not a.demand_seeds else 3000+dseed;ref=Prepared(simulate(demand,seed=simseed));variants={s:Prepared(simulate(demand,seed=simseed,shift=s),reference=ref) for s in a.shifts.split(',')}
   seeds=mseeds if not a.demand_seeds else [seed]
   for mseed in seeds:
    tag=f'IGES_{mseed}' if not a.demand_seeds else f'replicate_{dseed}';run=out/case/tag;run.mkdir(parents=True,exist_ok=True);done=run/'complete.json'
    if done.exists():z=json.loads(done.read_text());rows+=z['rows'];pol+=z.get('policies',[]);continue
    print('START',case,dseed,mseed,flush=True);tic=time.perf_counter();model,info=fit_iges(ref,mseed,run);local=model.local_only();
    if not a.skip_policy:tune_policy(local,ref);tune_policy(model,ref)
    cur=[];cp=[]
    for name,m in [('LocalXGBTuned',local),('IGES',model)]:
     for shift,d in variants.items():
      ts=d.parts['test'];p1,q=m.predict(d,ts);res=metrics(d.y[ts+2],d.b[ts+2],p1,q,m.threshold);cur.append(dict(case=case,demand_seed=dseed,model_seed=mseed,seed=mseed,method=name,shift=shift,elapsed=time.perf_counter()-tic,threshold=m.threshold,policy_threshold=m.policy_threshold,policy_scale=m.policy_scale,**info,**res));np.savez_compressed(run/f'predictions_{name}_{shift}.npz',time=ts,y=d.y[ts+2],b=d.b[ts+2],p=p1,q=q)
      if not a.skip_policy:
       val,ledger=closed_loop(m,d);cp.append(dict(case=case,demand_seed=dseed,method=name,seed=mseed,shift=shift,policy_threshold=m.policy_threshold,policy_scale=m.policy_scale,**val));ledger.to_csv(run/f'policy_{name}_{shift}.csv',index=False)
      print('RESULT',case,dseed,name,shift,m.alpha,round(res['f1'],4),flush=True)
    rows+=cur;pol+=cp;done.write_text(json.dumps(dict(rows=cur,policies=cp),indent=2));pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(pol).to_csv(out/'policy.csv',index=False)
if __name__=='__main__':main()
