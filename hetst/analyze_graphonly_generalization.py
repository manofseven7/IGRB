from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier,XGBRegressor
from .simulator import synthetic,simulate
from .data import Prepared
from .evaluation import metrics
rows=[]
for case,key in [('seasonal','s'),('intermittent','i'),('correlated','c')]:
 for dseed,mseed in zip([202,303,404,505,606],[11,23,37,53,71]):
  demand=synthetic(case,dseed);ref=Prepared(simulate(demand,seed=2000+dseed));run=Path('results')/f'igrb_gen_{key}'/case/f'replicate_{dseed}';gc=XGBClassifier();gr=XGBRegressor();gc.load_model(run/'graph_classifier.ubj');gr.load_model(run/'graph_regressor.ubj');va=ref.parts['val'];pv=gc.predict_proba(ref.graph_tabular(va))[:,1].reshape(len(va),ref.net.n);grid=np.linspace(.15,.85,141);s=np.array([f1_score(ref.y[va+2].ravel(),(pv>=t).ravel(),average='macro',zero_division=0) for t in grid]);ids=np.where(s>=s.max()-1e-12)[0];thr=float(grid[ids[np.argmin(abs(grid[ids]-.5))]])
  for shift in ['none','lead','policy']:
   d=Prepared(simulate(demand,seed=2000+dseed,shift=shift),reference=ref);ts=d.parts['test'];p=gc.predict_proba(d.graph_tabular(ts))[:,1].reshape(len(ts),d.net.n);q=np.maximum(0,gr.predict(d.graph_tabular(ts))).reshape(len(ts),d.net.n);rows.append(dict(case=case,demand_seed=dseed,model_seed=mseed,method='GraphOnly',shift=shift,threshold=thr,**metrics(d.y[ts+2],d.b[ts+2],p,q,thr)))
out=pd.DataFrame(rows);out.to_csv('results/igrb_analysis/graph_only_independent.csv',index=False);print(out.groupby(['case','shift']).f1.mean().to_string())
