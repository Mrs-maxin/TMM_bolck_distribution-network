"""Independent probability checks: analytic electrical distributions against voltage-model draws."""
from pathlib import Path
import sys,json,pickle,time
import numpy as np
import pandas as pd
import run_direct_heavytail_case33 as r

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--scene',choices=list(r.SCENES));args=parser.parse_args()
    rows=[]
    for scene in ([args.scene] if args.scene else r.SCENES):
        d=r.OUT/scene
        if not (d/'models_K2.pkl').exists():continue
        with (d/'models_K2.pkl').open('rb') as f:models=pickle.load(f)
        data=dict(np.load(d/'voltage_coordinates.npz'));r.gs.DRAWS=524288
        for name,model in models.items():
            path=d/f'CDF_{name}_K2.npz'
            if not path.exists():continue
            t0=time.perf_counter()
            if name=='grouped_t_shift':y,w=r.sh.model_outputs(model,data,2026102671,draws=524288)
            else:y,w=r.gs.model_outputs(model,data,2026102671)
            a=np.load(path)
            # All 97 electrical objects, with the same MC thresholds as the analytic calculation.
            for j in range(97):
                order=np.argsort(y[:,j]);F=np.interp(a['thresholds'][j],y[order,j],np.cumsum(w[order])/w.sum(),left=0,right=1)
                diff=F-a['CDF'][j]
                rows.append(dict(scene=scene,model=name,K=2,object=r.object_name(j),
                    max_probability_difference=float(abs(diff).max()),tail_mean_difference=float(abs(diff[:100]).mean()),
                    weighted_quadrature_points=len(w),seconds=time.perf_counter()-t0))
            print(scene,name,'max difference',max(v['max_probability_difference'] for v in rows if v['scene']==scene and v['model']==name),flush=True)
            pd.DataFrame(rows).to_csv(r.OUT/f'analytic_probability_check_{args.scene or "all"}.csv',index=False)

if __name__=='__main__':main()
