from pathlib import Path
import pandas as pd
root=Path('..')/'results'/'igrb_analysis';main=pd.read_csv(root/'main_summary.csv');ind=pd.read_csv(root/'independent_effects_all.csv');prev=pd.read_csv(root/'previous_best_comparison.csv');pol=pd.read_csv(root/'policy_summary.csv');abl=pd.read_csv(root/'ablation_summary.csv');order=['seasonal','intermittent','correlated','retail'];labels={'seasonal':'Seasonal','intermittent':'Intermittent','correlated':'Correlated','retail':'Retail-driven'};L=[]
L.append('Table~\\ref{tab:primary} reports the fixed-realization comparison. IGRB selects nonzero graph corrections for seasonal and correlated demand and returns exactly to the local expert for intermittent and retail-driven demand. The graph contribution is selective by construction.')
L += ['\\begin{table}[tb]\\centering\\scriptsize','\\caption{Unchanged-condition predictive results on the fixed benchmark realization (mean over five model seeds). RMSE$^+$ is normalized severity RMSE on positive targets.}\\label{tab:primary}','\\resizebox{\\linewidth}{!}{\\begin{tabular}{lrrrrrr}\\toprule Case & Method & Macro F1 & AP & Brier & RMSE$^+$ & $\\alpha$ \\\\\\midrule']
for case in order:
 for method,name in [('LocalXGBTuned','Local expert'),('IGRB','IGRB')]:
  r=main[(main.case==case)&(main['shift']=='none')&(main.method==method)].iloc[0];L.append(f"{labels[case]} & {name} & {r.f1:.4f} & {r.auprc:.4f} & {r.brier:.4f} & {r.rmse_pos:.4f} & {r.alpha if method=='IGRB' else 0:.2f} \\\\")
 L.append('\\addlinespace')
L += ['\\bottomrule\\end{tabular}}\\end{table}','Against the strongest method in the reconstructed benchmark, IGRB improves macro F1 in seasonal, intermittent and correlated demand and is practically tied in retail-driven demand. The intermittent gain is attributable to the retuned local expert because graph correction is disabled.']
L += ['\\begin{table}[tb]\\centering\\small','\\caption{IGRB versus the strongest previously evaluated method, unchanged fixed realization.}\\label{tab:previous}','\\begin{tabular}{llrrr}\\toprule Case & Previous best & Previous F1 & IGRB F1 & Difference \\\\\\midrule']
for case in order:
 r=prev[prev.case==case].iloc[0];L.append(f"{labels[case]} & {r.previous_best} & {r.previous_f1:.4f} & {r.igrb_f1:.4f} & {r.delta:+.4f} \\\\")
L += ['\\bottomrule\\end{tabular}\\end{table}','Independent records provide the stronger generalization check (Table~\\ref{tab:independent}). The unchanged correlated-demand interval excludes zero across ten records; the seasonal interval crosses zero after adding five confirmatory records. Intermittent demand is an exact fallback.']
L += ['\\begin{table}[tb]\\centering\\small','\\caption{Paired IGRB minus local-expert macro-F1 differences over independent demand replications; percentile bootstrap intervals use 30,000 paired resamples.}\\label{tab:independent}','\\begin{tabular}{llrrrrr}\\toprule Case & Condition & $n$ & Difference & 95\\% interval & Wins & Losses \\\\\\midrule']
for case in ['seasonal','intermittent','correlated']:
 for shift in ['none','lead','policy']:
  r=ind[(ind.case==case)&(ind['shift']==shift)].iloc[0];sl={'none':'Unchanged','lead':'Lead-time shift','policy':'Policy shift'}[shift];L.append(f"{labels[case]} & {sl} & {int(r.n)} & {r.delta:+.4f} & [{r.ci_low:+.4f}, {r.ci_high:+.4f}] & {int(r.wins)} & {int(r.losses)} \\\\")
 L.append('\\addlinespace')
L += ['\\bottomrule\\end{tabular}\\end{table}','The component study in Table~\\ref{tab:ablation} shows that forcing the graph expert harms intermittent demand and removing the ADI screen reduces its mean F1. Graph-only is slightly higher in the fixed dense cases, but it removes the fallback protection and its advantage is not evidence of general robustness. Error weighting is not uniformly beneficial and is excluded from the final method.']
L += ['\\begin{table}[tb]\\centering\\scriptsize','\\caption{Component ablations on the unchanged fixed realization (mean $\\pm$ standard deviation over five model seeds).}\\label{tab:ablation}','\\resizebox{\\linewidth}{!}{\\begin{tabular}{lrrrr}\\toprule Case & Local only & Graph only & IGRB & Targeted ablation \\\\\\midrule']
def cell(case,v):
 r=abl[(abl.case==case)&(abl.variant==v)];return '--' if r.empty else f"{r.iloc[0].f1:.4f} $\\pm$ {r.iloc[0].f1_sd:.4f}"
for case in order:
 target='Error-weighted: '+cell(case,'ErrorWeighted') if case in ['seasonal','correlated'] else 'No ADI: '+cell(case,'NoADI');L.append(f"{labels[case]} & {cell(case,'LocalOnly')} & {cell(case,'GraphOnly')} & {cell(case,'IGRB')} & {target} \\\\")
L += ['\\bottomrule\\end{tabular}}\\end{table}','The decision guardrail abstains in seasonal, intermittent and retail-driven cases. In correlated demand, IGRB lowers mean cost under all conditions (Table~\\ref{tab:policy}); relative reductions are approximately 0.14\\%, 1.27\\% and 2.58\\%. These are conditional simulator results, not universal policy superiority.']
L += ['\\begin{table}[tb]\\centering\\small','\\caption{Correlated-demand closed-loop results (mean over five model seeds).}\\label{tab:policy}','\\begin{tabular}{lrrrr}\\toprule Condition & Local cost & IGRB cost & Local fill & IGRB fill \\\\\\midrule']
for shift in ['none','lead','policy']:
 a=pol[(pol.case=='correlated')&(pol['shift']==shift)&(pol.method=='LocalXGBTuned')].iloc[0];b=pol[(pol.case=='correlated')&(pol['shift']==shift)&(pol.method=='IGRB')].iloc[0];sl={'none':'Unchanged','lead':'Lead-time shift','policy':'Policy shift'}[shift];L.append(f"{sl} & {a.cost:.0f} & {b.cost:.0f} & {a.fill_rate:.4f} & {b.fill_rate:.4f} \\\\")
L += ['\\bottomrule\\end{tabular}\\end{table}','\\FloatBarrier'];Path('igrb_tables.tex').write_text('\n'.join(L)+'\n')
