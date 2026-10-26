"""Mirrored lower-probability quadrature checks; do not overwrite completed PDFs."""
import json,concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from independent_block_t_fit import OUT,SRC,geometry,dump
from independent_block_t_numeric import specs,projected_parts,terms,scale_nodes
from independent_cf_inversion import inverse_pdf,moments


def check(task):
 k,name=task;g=geometry();ss=specs(g);s=next(z for z in ss if z['name']==name);j=ss.index(s) if False else next(i for i,z in enumerate(ss) if z['name']==name);folder=OUT/'distributions'/f'K{k}'/'Block_TMM';old=pd.read_csv(folder/f'{name}.csv');info=json.load(open(folder/f'{name}.json'));model=dict(np.load(OUT/f'Block_TMM_K{k}.npz'));parts=list(projected_parts(model,'Block_TMM',g,s));tm=terms(parts,s,info['strength_level'],mirror_lower=True);r=np.load(SRC/'ac_results.npz')['reference'][:,j];x=np.unique(np.r_[old.value.iloc[int(np.argmax(old.pdf))],np.quantile(r,np.r_[np.linspace(.01,.05,81),np.linspace(.95,.99,81)])]);q=x*x if s['kind']=='V' else x;pdf,cdf,_=inverse_pdf(tm,q)
 if s['kind']=='V':pdf*=2*x
 pold=np.interp(x,old.value,old.pdf);fold=np.interp(x,old.value,old.cdf);sd=float(np.sqrt(info['exact_quadratic_variance']))
 if s['kind']=='V':sd=float(np.std(r))
 dp=float(abs(pdf-pold).max()*sd);dc=float(abs(cdf-fold).max());return dict(K=k,model='Block_TMM',object=name,compared_points=len(x),scaled_PDF_difference=dp,maximum_CDF_difference=dc,within_registered_tolerance=bool(dp<2e-4 and dc<2e-5),completed_PDF_overwritten=False)


def run():
 checks=[]
 for mirror in [False,True]:
  ss,w=scale_nodes(4.1,2,mirror_lower=mirror);tm=[(float(wi),0.,np.zeros(1),np.array([np.sqrt(si)])) for wi,si in zip(w,ss)];x=np.linspace(-8,8,101);p,f,_=inverse_pdf(tm,x);checks.append(dict(mirror_lower=mirror,states=len(ss),PDF_error=float(abs(p-stats.t.pdf(x,4.1)).max()),CDF_error=float(abs(f-stats.t.cdf(x,4.1)).max())))
 tasks=[(k,name) for k in [1,2,4] for name in ['V18','V33','P1','P16','P32','Q17']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(check,tasks))
 dump(OUT/'scale_mirror_checks.json',dict(status='passed' if all(z['within_registered_tolerance'] for z in rows) else 'requires_affected_object_refinement',known_t=checks,representative_objects=rows,coverage='18 key non-loss distributions, AC tail points and native PDF peak; no claim of every grid point',existing_distributions_changed=False))
 print('KNOWN',checks,flush=True)
 for r in rows:print(r,flush=True)

if __name__=='__main__':run()
