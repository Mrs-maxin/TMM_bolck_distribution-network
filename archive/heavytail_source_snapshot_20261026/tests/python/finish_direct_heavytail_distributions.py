"""Finish missing analytic electrical distributions using existing fitted models.

Only electrical objects are evaluated concurrently; formulas, integration nodes,
input scenarios and already saved results remain unchanged.
"""
import pickle,time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import run_direct_heavytail_case33 as r

def one_object(payload):
    j,tm,ref,feature=payload;L=len(r.LEVELS)
    x=ref['thresholds'];F,end=r.compressed_cdf(tm,x*x if j<32 else x)
    lo=F[:L]-ref['actual'][:L];hi=ref['actual'][L:2*L]-F[L:2*L]
    row=dict(object=r.object_name(j),kind=feature['kind'],lower_tail_pp=100*np.mean(abs(lo)),
        upper_tail_pp=100*np.mean(abs(hi)),lower_signed_pp=100*np.mean(lo),upper_signed_pp=100*np.mean(hi),
        full_CDF_max_pp=100*np.max(abs(F[2*L:]-ref['actual'][2*L:])),CF_end_abs=end,
        extended_lower=bool(feature['extended_lower']),extended_upper=bool(feature['extended_upper']))
    return row,F

def main():
    for scene in r.SCENES:
        d=r.OUT/scene
        if (d/'comparison_complete.json').exists():continue
        refs,features=r.reference(scene);data=dict(np.load(d/'voltage_coordinates.npz'))
        forms=r.an.quadratic_forms(data);allrows=[];timings=[]
        with ProcessPoolExecutor(max_workers=6,mp_context=get_context('spawn')) as pool:
            for k in r.KS:
                with (d/f'models_K{k}.pkl').open('rb') as f:models=pickle.load(f)
                for name,kind in [('AQGMM','GMM'),('standard_TMM','TMM'),('grouped_t_shift','SHIFT')]:
                    p=d/f'objects_{name}_K{k}.csv'
                    if not p.exists():
                        t0=time.perf_counter();terms=r.an.terms(models[name],kind,forms)
                        payloads=[(j,terms[j],refs[j],features.iloc[j].to_dict()) for j in range(97)]
                        values=list(pool.map(one_object,payloads,chunksize=1))
                        tab=pd.DataFrame([v[0] for v in values]);curves=np.array([v[1] for v in values])
                        tab.to_csv(p,index=False)
                        np.savez_compressed(d/f'CDF_{name}_K{k}.npz',CDF=curves,
                            thresholds=np.array([v['thresholds'] for v in refs]),MC_CDF=np.array([v['actual'] for v in refs]))
                        timings.append(dict(scene=scene,K=k,model=name,analytic_seconds=time.perf_counter()-t0,workers=6))
                        r.event('model_done_parallel',scene=scene,K=k,model=name,seconds=time.perf_counter()-t0)
                    else:tab=pd.read_csv(p)
                    allrows.append(tab.assign(scene=scene,K=k,model=name))
        pd.concat(allrows,ignore_index=True).to_csv(d/'all_object_errors.csv',index=False)
        pd.DataFrame(timings).to_csv(d/'completion_times.csv',index=False)
        r.dump(d/'comparison_complete.json',dict(K=r.KS,models=3,objects=97,
            retained_dimension=len(data['scale']),single_realization=True,
            completed_using_saved_models=True,missing_objects_evaluated_concurrently=True))

if __name__=='__main__':main()
