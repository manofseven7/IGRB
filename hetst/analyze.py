"""Aggregate only executed runs; regenerate every reported table/figure."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .simulator import simulate,synthetic
from .data import Prepared
from .models import Model
from .experiment import predict
from .evaluation import metrics

torch.set_num_threads(1)
CASES=['seasonal','intermittent','correlated','retail'];SEEDS=[11,23,37,53,71]
METHODS=['Persistence','LocalBalance','XGBoost','LSTM','GCN','RGCN','TGRU','HomGAT','RelGRU','HetST']
LABELS={'Persistence':'Persistence','LocalBalance':'Local balance','XGBoost':'XGBoost','LSTM':'Pooled LSTM','GCN':'Mean GNN','RGCN':'Relational mean GNN','TGRU':'Mean GNN + GRU','HomGAT':'Homogeneous attention','RelGRU':'Relational attention + GRU','HetST':'HetST-GNN'}

def f1_fast(y,p):
 y=y.astype(bool).ravel();p=(p>=.5).ravel();tp=np.sum(y&p);tn=np.sum(~y&~p);fp=np.sum(~y&p);fn=np.sum(y&~p)
 return .5*((2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn else 0)+.5*((2*tn)/(2*tn+fp+fn) if 2*tn+fp+fn else 0)

def boot(y,a,b,reps=1000):
 # a,b [seed,time,node]; same seed indices/time blocks used for both methods.
 rng=np.random.default_rng(812);S,T,N=a.shape;vals=[]
 for _ in range(reps):
  seeds=rng.integers(0,S,S);starts=rng.integers(0,T,int(np.ceil(T/7)));ix=np.concatenate([(s+np.arange(7))%T for s in starts])[:T]
  vals.append(np.mean([f1_fast(y[ix],a[s,ix])-f1_fast(y[ix],b[s,ix]) for s in seeds]))
 return np.quantile(vals,[.025,.975])

def main():
 out=Path('results/analysis');out.mkdir(exist_ok=True);paper=Path('paper');paper.mkdir(exist_ok=True)
 frames=[pd.read_csv(Path('results')/case/'metrics.csv') for case in CASES];m=pd.concat(frames,ignore_index=True)
 p=pd.concat([pd.read_csv(Path('results')/case/'policy.csv') for case in CASES],ignore_index=True)
 m.to_csv(out/'all_metrics.csv',index=False);p.to_csv(out/'all_policy.csv',index=False)
 summary=m.groupby(['case','method','shift']).agg({k:['mean','std'] for k in ['f1','auroc','auprc','brier','rmse_pos','rmse_all','event_recall','alert_precision','false_alerts_per_100','ewlt','train_seconds']});summary.to_csv(out/'summary.csv')
 base=p[p.method=='NoAction'].set_index(['case','shift']).cost
 p['saving_pct']=[100*(1-row.cost/base.loc[(row.case,row['shift'])]) for _,row in p.iterrows()];p.to_csv(out/'policy_comparison.csv',index=False)
 cis=[]
 for case in CASES:
  for other in ['XGBoost','RelGRU']:
   a=[];b=[]
   for seed in SEEDS:
    pa=np.load(Path('results')/case/case/f'HetST_{seed}'/'predictions_none.npz');pb=np.load(Path('results')/case/case/f'{other}_{seed}'/'predictions_none.npz');a.append(pa['p']);b.append(pb['p']);y=pa['y']
   lo,hi=boot(y,np.array(a),np.array(b));delta=np.mean([f1_fast(y,x)-f1_fast(y,z) for x,z in zip(a,b)])
   cis.append(dict(case=case,reference=other,delta=delta,lo=lo,hi=hi))
 pd.DataFrame(cis).to_csv(out/'paired_bootstrap.csv',index=False)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
 fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
 for ax,case in zip(axs.flat,CASES):
  ss=m[(m.case==case)&(m['shift']=='none')].groupby('method').f1.agg(['mean','std']).reindex(METHODS)
  ax.barh(np.arange(len(METHODS)),ss['mean'],xerr=ss['std'],color=['#cf6a32' if k=='HetST' else '#427e9d' for k in METHODS]);ax.set_yticks(np.arange(len(METHODS)),[LABELS[k] for k in METHODS],fontsize=8);ax.set_xlim(0,1);ax.set_title(case.capitalize());ax.set_xlabel('Macro F1 (mean +/- seed SD)');ax.invert_yaxis()
 fig.savefig(paper/'prediction.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
 for ax,shift in zip(axs,['none','lead','policy']):
  for method in ['Persistence','XGBoost','RelGRU','HetST']:
   ss=m[(m.method==method)&(m['shift']==shift)].groupby('case').f1.mean().reindex(CASES);ax.plot(np.arange(4),ss,'o-',label=LABELS[method])
  ax.set_xticks(np.arange(4),[x[:6] for x in CASES],rotation=20);ax.set_ylim(0,1);ax.set_title({'none':'Unchanged','lead':'Lead-time shift','policy':'Base-stock shift'}[shift]);ax.set_ylabel('Macro F1')
 axs[0].legend(fontsize=7);fig.savefig(paper/'shift.pdf');plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
 for ax,shift in zip(axs,['none','lead','policy']):
  for method in ['Persistence','Uniform','XGBoost','RelGRU','HetST']:
   ss=p[(p.method==method)&(p['shift']==shift)].groupby('case').saving_pct.mean().reindex(CASES);ax.plot(np.arange(4),ss,'o-',label=LABELS.get(method,method))
  ax.axhline(0,color='black',lw=.7);ax.set_xticks(np.arange(4),[x[:6] for x in CASES],rotation=20);ax.set_title(shift);ax.set_ylabel('Cost saving vs no action (%)')
 axs[0].legend(fontsize=7);fig.savefig(paper/'decision.pdf');plt.close(fig)
 # Write compact exact-number LaTeX tables without manual transcription.
 lines=['\\begin{tabular}{lrrrr}','\\toprule','Method & Seasonal & Intermittent & Correlated & Retail \\\\','\\midrule']
 for method in METHODS:
  cells=[]
  for case in CASES:
   v=m[(m.case==case)&(m.method==method)&(m['shift']=='none')].f1;cells.append(f'{v.mean():.3f} $\\pm$ {v.std():.3f}')
  lines.append(LABELS[method]+' & '+' & '.join(cells)+' \\\\')
 lines+=['\\bottomrule','\\end{tabular}'];(paper/'table_f1.tex').write_text('\n'.join(lines))
 lines=['\\begin{tabular}{llrrrr}','\\toprule','Case & Method & AUROC & AP & Brier & RMSE$^{+}$ \\\\','\\midrule']
 for case in CASES:
  for method in ['Persistence','XGBoost','RelGRU','HetST']:
   ss=m[(m.case==case)&(m.method==method)&(m['shift']=='none')];cells=[f'{ss[k].mean():.3f}' for k in ['auroc','auprc','brier','rmse_pos']];lines.append(case.capitalize()+' & '+LABELS[method]+' & '+' & '.join(cells)+' \\\\')
 lines+=['\\bottomrule','\\end{tabular}'];(paper/'table_metrics.tex').write_text('\n'.join(lines))
 lines=['\\begin{tabular}{llrrr}','\\toprule','Case & Reference & $\\Delta$F1 & Lower & Upper \\\\','\\midrule']
 for r in cis:lines.append(f'{r["case"].capitalize()} & {LABELS[r["reference"]]} & {r["delta"]:.3f} & {r["lo"]:.3f} & {r["hi"]:.3f} \\\\')
 lines+=['\\bottomrule','\\end{tabular}'];(paper/'table_ci.tex').write_text('\n'.join(lines))
 lines=['\\begin{tabular}{lrrrr}','\\toprule','Method & Seasonal & Intermittent & Correlated & Retail \\\\','\\midrule']
 for method in ['Uniform','Persistence','LocalBalance','XGBoost','LSTM','GCN','RGCN','TGRU','HomGAT','RelGRU','HetST']:
  cells=[f'{p[(p.case==case)&(p.method==method)&(p["shift"]=="none")].saving_pct.mean():.1f}' for case in CASES];lines.append(LABELS.get(method,method)+' & '+' & '.join(cells)+' \\\\')
 lines+=['\\bottomrule','\\end{tabular}'];(paper/'table_policy.tex').write_text('\n'.join(lines))
 a=pd.read_csv('results/ablation/metrics.csv');a=pd.concat([a,m[(m.case=='retail')&(m.method.isin(['HetST','HomGAT']))&(m['shift']=='none')]])
 a.to_csv(out/'ablation.csv',index=False)
 lines=['\\begin{tabular}{lrr}','\\toprule','Variant & Macro F1 & Conditional RMSE \\\\','\\midrule']
 for kind in ['HetST','NoTime','NoEdge','HomGAT','SingleHead','NoSelf','ClsOnly','RegOnly']:
  ss=a[a.method==kind];cells=['---' if ss[k].isna().all() else f'{ss[k].mean():.3f} $\\pm$ {ss[k].std():.3f}' for k in ['f1','rmse_pos']];lines.append(kind+' & '+' & '.join(cells)+' \\\\')
 lines+=['\\bottomrule','\\end{tabular}'];(paper/'table_ablation.tex').write_text('\n'.join(lines))
 lines=['\\begin{tabular}{llrrrr}','\\toprule','Case & Method & Recall & Precision & FA/100 & Lead \\\\','\\midrule']
 for case in CASES:
  for method in ['XGBoost','RelGRU','HetST']:
   ss=m[(m.case==case)&(m.method==method)&(m['shift']=='none')];cells=['---' if ss[k].isna().all() else f'{ss[k].mean():.3f}' for k in ['event_recall','alert_precision','false_alerts_per_100','ewlt']];lines.append(case.capitalize()+' & '+LABELS[method]+' & '+' & '.join(cells)+' \\\\')
 lines+=['\\bottomrule','\\end{tabular}'];(paper/'table_events.tex').write_text('\n'.join(lines))
 print(m[m['shift']=='none'].groupby(['case','method']).f1.mean().to_string());print(pd.DataFrame(cis).to_string(index=False));print(p[p['shift']=='none'].groupby(['case','method']).saving_pct.mean().to_string())

if __name__=='__main__':main()
