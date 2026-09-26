"""Rebuild aggregate CSV files from resumable per-run completion records."""
from pathlib import Path
import json,pandas as pd
def collect(root):
 rows=[];pol=[]
 for f in sorted(Path(root).glob('*/*/complete.json')):
  x=json.loads(f.read_text())
  if isinstance(x,dict):rows+=x.get('rows',[]);pol+=x.get('policies',[])
  elif isinstance(x,list):rows+=x
 pd.DataFrame(rows).to_csv(Path(root)/'metrics.csv',index=False)
 if pol:pd.DataFrame(pol).to_csv(Path(root)/'policy.csv',index=False)
 return len(rows),len(pol)
def main():
 print('main',collect('results/igrb_final'))
 for x in ['s','i','c']:print('generalization',x,collect(f'results/igrb_gen_{x}'))
 print('error_weighted',collect('results/igrb_errorweighted'));print('no_adi',collect('results/igrb_noadi'))
if __name__=='__main__':main()
