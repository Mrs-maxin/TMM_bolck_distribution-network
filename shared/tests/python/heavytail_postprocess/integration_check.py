"""Refine CF inversion only where the saved terminal CF is not negligible.

This changes numerical integration resolution, never any model parameters.
"""
import sys
sys.dont_write_bytecode=True
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import recompute as r

def check(payload):
    item,terms,x,oldcdf=payload
    roots=np.array(item['quantiles'])**2 if item['j']<32 else np.array(item['quantiles'])
    xx=x[:100]**2 if item['j']<32 else x[:100]
    current=r.CDFKernel(terms);factor=1
    for _ in range(6):
        high=r.CDFKernel(terms,cutoff_multiplier=factor*2)
        probes=np.r_[xx,roots];difference=float(abs(high.cdf(probes)-current.cdf(probes)).max())
        if difference<=1e-5:break
        factor*=2;current=high
        for i,p in enumerate(r.PROBS):
            delta=max(current.sd,1e-12);lo=roots[i]-delta;hi=roots[i]+delta
            if item['j']<32:lo=max(0.,lo)
            for _ in range(25):
                a,b=current.cdf([lo,hi])
                if a<=p<=b:break
                delta*=2;lo=roots[i]-delta;hi=roots[i]+delta
                if item['j']<32:lo=max(0.,lo)
            else:raise RuntimeError('refined quantile has no bracket')
            roots[i]=brentq(lambda v:float(current.cdf([v])[0])-p,lo,hi,
                           xtol=max(current.sd*1e-9,1e-13),rtol=1e-12,maxiter=120)
    else:raise RuntimeError((item['scene'],item['K'],item['model'],item['j'],'frequency refinement did not converge'))
    residual=abs(current.cdf(roots)-r.PROBS)
    assert residual.max()<1e-6
    record=dict(scene=item['scene'],K=item['K'],model=item['model'],j=item['j'],
        cutoff_factor_used=factor,cutoff_factor_checked=factor*2,max_CDF_refinement_difference=difference,
        CDF100=current.cdf(xx).tolist(),quantiles=(np.sqrt(np.maximum(roots,0)) if item['j']<32 else roots).tolist(),
        CDF_residual=residual.tolist())
    updated=dict(item,quantiles=record['quantiles'],CDF_residual=residual.tolist(),
        CF_end=current.CF_end,frequency_points=len(current.tau),cutoff_multiplier=factor,
        frequency_refinement_max_difference=difference,frequency_refinement_checked=True)
    return record,updated

def main():
    done={}
    for line in (r.ROOT/'computation_records/analytic_quantile_log.jsonl').read_text().splitlines():
        item=json.loads(line);done[(item['scene'],item['K'],item['model'],item['j'])]=item
    assert len(done)==4656
    selected={key for key,item in done.items() if item['CF_end']>1e-7}
    ns=r.numerical_functions();futures=[];count=0
    with ProcessPoolExecutor(max_workers=6,mp_context=get_context('spawn')) as pool:
        for scene in r.SCENES:
            forms=ns['quadratic_forms'](dict(np.load(r.OLD/scene/'voltage_coordinates.npz')))
            for k in [1,2,4,6]:
                models=r.read_models(r.OLD/scene/f'models_K{k}.pkl')
                for model,kind in r.MODELS.items():
                    js=[j for j in range(97) if (scene,k,model,j) in selected]
                    if not js:continue
                    terms=ns['terms'](models[model],kind,forms)
                    a=np.load(r.OLD/scene/f'CDF_{model}_K{k}.npz');xs=a['thresholds'];cdf=a['CDF']
                    for j in js:futures.append(pool.submit(check,(done[(scene,k,model,j)],terms[j],xs[j],cdf[j])))
        with (r.ROOT/'computation_records/frequency_integration_refinement_log.jsonl').open('w') as stream,(r.ROOT/'computation_records/analytic_quantile_log.jsonl').open('a') as mainlog:
            for future in as_completed(futures):
                record,updated=future.result();stream.write(json.dumps(record)+'\n');stream.flush()
                mainlog.write(json.dumps(updated)+'\n');mainlog.flush();count+=1
                if count%10==0:print('frequency checks',count,'/',len(futures),flush=True)
    print('frequency refinement complete',count,flush=True)

if __name__=='__main__':main()
