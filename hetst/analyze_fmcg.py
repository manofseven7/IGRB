"""Create the manuscript-facing summary for the FMCG-inspired robustness check."""
from pathlib import Path
import pandas as pd

ROOT=Path('results')
OUT=ROOT/'igrb_analysis'
OUT.mkdir(parents=True,exist_ok=True)

igrb=pd.read_csv(ROOT/'igrb_fmcg'/'metrics.csv').query("shift == 'none'")
base=pd.read_csv(ROOT/'fmcg_baselines'/'metrics.csv').query("shift == 'none'")
frames=[]
for data in (base,igrb):
    frames.append(data.groupby('method').agg(f1=('f1','mean'),f1_sd=('f1','std'),
        auprc=('auprc','mean'),brier=('brier','mean'),rmse_pos=('rmse_pos','mean')).reset_index())
summary=pd.concat(frames,ignore_index=True)
order=['LocalBalance','Persistence','XGBoost','LocalXGBTuned','IGRB']
summary['rank']=summary.method.map({m:i for i,m in enumerate(order)})
summary=summary.sort_values('rank').drop(columns='rank')
summary.to_csv(OUT/'fmcg_benchmark.csv',index=False)

shift=pd.read_csv(ROOT/'igrb_fmcg'/'metrics.csv')
p=shift.pivot_table(index=['seed','shift'],columns='method',values='f1').reset_index()
p['delta']=p.IGRB-p.LocalXGBTuned
p.groupby('shift').delta.agg(['mean','std','min','max']).reset_index().to_csv(
    OUT/'fmcg_shift_effects.csv',index=False)

policy=pd.read_csv(ROOT/'igrb_fmcg'/'policy.csv')
policy.groupby(['shift','method']).agg(cost=('cost','mean'),fill_rate=('fill_rate','mean'),
    policy_scale=('policy_scale','mean')).reset_index().to_csv(OUT/'fmcg_policy.csv',index=False)
print(summary.to_string(index=False))
