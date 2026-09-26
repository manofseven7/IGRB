import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from .simulator import synthetic,simulate,Inventory
from .data import Prepared
from .evaluation import metrics,events
from .hybrid import fit_igrb,tune_policy

def closed_loop(model,data):
 sim=data.sim;T=len(sim['x']);env=Inventory(data.net,T,sim['seed'],shift_at=sim['c2'],shift=sim['shift']);actions=np.zeros((T+2,data.net.n));xs=[];records=[];leaves=data.net.leaves;budget=.15*data.net.mean[leaves].sum()
 for t,demand in enumerate(sim['demand']):
  x,e,_,ledger=env.step(demand,actions[t]);xx,_=data.transform_online(x,e,t);xs.append(xx)
  if sim['c2']<=t<T-2 and len(xs)>=data.window:
   p,q=model.predict_window(data,np.asarray(xs[-data.window:]));rem=budget
   for v in leaves[np.argsort(-p[leaves],kind='stable')]:
    if p[v]<model.policy_threshold:continue
    a=min(rem,max(0,model.policy_scale*q[v]*data.net.mean[v]));actions[t+2,v]=a;rem-=a
    if rem<=1e-12:break
  if t>=sim['c2']:ledger.update(t=t,cost=.1*ledger['holding']+5*ledger['backlog']+2*ledger['expedite']);records.append(ledger)
 frame=pd.DataFrame(records);return dict(cost=float(frame.cost.sum()),fill_rate=float(frame.served_current.sum()/max(frame.demand.sum(),1e-8)),backlog_unit_cycles=float(frame.backlog.sum()),emergency_units=float(frame.expedite.sum()),holding_unit_cycles=float(frame.holding.sum())),frame
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='results/igrb_final');p.add_argument('--seeds',default='11,23,37,53,71');p.add_argument('--cases',default='seasonal,intermittent,correlated,retail');p.add_argument('--shifts',default='none,lead,policy');p.add_argument('--skip-policy',action='store_true');p.add_argument('--no-adi',action='store_true');p.add_argument('--error-weighting',action='store_true');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[];policies=[]
 for case in a.cases.split(','):
  demand=np.load('data/retail_demand.npy') if case=='retail' else synthetic(case,101);base=simulate(demand,seed=1001);ref=Prepared(base);variants={s:Prepared(simulate(demand,seed=1001,shift=s),reference=ref) for s in a.shifts.split(',')}
  for seed in map(int,a.seeds.split(',')):
   run=out/case/f'IGRB_{seed}';run.mkdir(parents=True,exist_ok=True);done=run/'complete.json'
   if done.exists():saved=json.loads(done.read_text());rows+=saved['rows'];policies+=saved['policies'];continue
   print('START',case,seed,flush=True);tic=time.perf_counter();model,info=fit_igrb(ref,seed,run,use_adi=not a.no_adi,error_weighting=a.error_weighting);local=model.local_only();pinfo={} if a.skip_policy else {'LocalXGBTuned':tune_policy(local,ref),'IGRB':tune_policy(model,ref)};(run/'policy_tuning.json').write_text(json.dumps(pinfo,indent=2));current=[];cp=[]
   for name,m in {'LocalXGBTuned':local,'IGRB':model}.items():
    for shift,d in variants.items():
     ts=d.parts['test'];pr,q=m.predict(d,ts);truth=d.y[ts+2];back=d.b[ts+2];res=metrics(truth,back,pr,q,m.threshold);warn=events(d.y[d.sim['c2']:],np.r_[pr,np.zeros((2,d.net.n))],threshold=m.threshold,previous=d.y[d.sim['c2']-1]);current.append(dict(case=case,method=name,seed=seed,shift=shift,threshold=m.threshold,policy_threshold=m.policy_threshold,policy_scale=m.policy_scale,train_seconds=time.perf_counter()-tic,**info,**res,**warn));np.savez_compressed(run/f'predictions_{name}_{shift}.npz',time=ts,y=truth,b=back,p=pr,q=q,threshold=m.threshold)
     if not a.skip_policy:
      val,ledger=closed_loop(m,d);cp.append(dict(case=case,method=name,seed=seed,shift=shift,policy_threshold=m.policy_threshold,policy_scale=m.policy_scale,**val));ledger.to_csv(run/f'policy_{name}_{shift}.csv',index=False)
     print('RESULT',case,name,shift,'alpha',m.alpha,'F1',round(res['f1'],4),flush=True)
   rows+=current;policies+=cp;done.write_text(json.dumps(dict(rows=current,policies=cp,model=model.metadata()),indent=2));pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(policies).to_csv(out/'policy.csv',index=False)
if __name__=='__main__':main()
