"""Strength-quadrature refinement at MC-selected electrical quantities."""
import pickle,json
import numpy as np
import pandas as pd
import run_direct_heavytail_case33 as r
rows=[]
for scene in r.SCENES:
    d=r.OUT/scene
    if not (d/'models_K2.pkl').exists():continue
    with (d/'models_K2.pkl').open('rb') as f:models=pickle.load(f)
    data=dict(np.load(d/'voltage_coordinates.npz'));forms=r.an.quadratic_forms(data)
    ids=json.loads((d/'representatives_locked_before_fit.json').read_text())['indices']
    r.an.U_NODES=24
    for name,kind in [('standard_TMM','TMM'),('grouped_t_shift','SHIFT')]:
        a=np.load(d/f'CDF_{name}_K2.npz');terms=r.an.terms(models[name],kind,forms)
        for j in ids:
            x=a['thresholds'][j];F,end=r.compressed_cdf(terms[j],x*x if j<32 else x)
            dif=abs(F-a['CDF'][j]);rows.append(dict(scene=scene,model=name,K=2,object=r.object_name(j),
                max_CDF_difference=float(dif.max()),tail_mean_difference=float(dif[:100].mean()),CF_end=end,
                strength_nodes_original=48,strength_nodes_refined=96))
        print(scene,name,max(v['max_CDF_difference'] for v in rows if v['scene']==scene and v['model']==name),flush=True)
        pd.DataFrame(rows).to_csv(r.OUT/'strength_integration_refinement_K2.csv',index=False)
