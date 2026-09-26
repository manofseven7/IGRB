import argparse,json,time
from pathlib import Path
import pandas as pd
from .simulator import synthetic,simulate
from .data import Prepared
from .evaluation import metrics
from .hybrid import fit_igrb
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='results/igrb_generalization');p.add_argument('--demand-seeds',default='202,303,404,505,606');p.add_argument('--model-seeds',default='11,23,37,53,71');p.add_argument('--cases',default='seasonal,intermittent,correlated');p.add_argument('--shifts',default='none,lead,policy');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);ds=list(map(int,a.demand_seeds.split(',')));ms=list(map(int,a.model_seeds.split(',')));rows=[]
 for case in a.cases.split(','):
  for dseed,mseed in zip(ds,ms):
   run=out/case/f'replicate_{dseed}';run.mkdir(parents=True,exist_ok=True);done=run/'complete.json'
   if done.exists():rows+=json.loads(done.read_text());continue
   demand=synthetic(case,dseed);ref=Prepared(simulate(demand,seed=2000+dseed));variants={s:Prepared(simulate(demand,seed=2000+dseed,shift=s),reference=ref) for s in a.shifts.split(',')};print('START',case,dseed,flush=True);tic=time.perf_counter();model,info=fit_igrb(ref,mseed,run);current=[]
   for name,m in {'LocalXGBTuned':model.local_only(),'IGRB':model}.items():
    for shift,d in variants.items():
     ts=d.parts['test'];pr,q=m.predict(d,ts);res=metrics(d.y[ts+2],d.b[ts+2],pr,q,m.threshold);current.append(dict(case=case,demand_seed=dseed,model_seed=mseed,method=name,shift=shift,elapsed=time.perf_counter()-tic,**info,**res));print('RESULT',case,dseed,name,shift,m.alpha,round(res['f1'],4),flush=True)
   rows+=current;done.write_text(json.dumps(current,indent=2));pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
if __name__=='__main__':main()
