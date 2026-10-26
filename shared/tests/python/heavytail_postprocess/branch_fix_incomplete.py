"""Analytic branch-Q evaluation with the known fixed nodal Q conditions.

This uses an AC-equivalent expression on the feasible power-flow set. It does
not claim that the fitted voltage law itself has become AC-feasible.
"""
import recompute as r
from branch_check import raw_network,fixed_Q_forms
import numpy as np
import pandas as pd
import inspect,json,time,argparse
from numpy.polynomial.legendre import leggauss
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context

OUT=r.ROOT/'diagnostics/branch_check/incomplete_analytic_check'
# Center before squaring: fixed load can be much larger than its loss variation.
source=inspect.getsource(r.CDFKernel).replace('class CDFKernel:','class StableCDFKernel:')
source=source.replace('self.mean=float(w@mk)\n        self.sd=float(np.sqrt(max(w@(vk+mk**2)-self.mean**2,1e-300)))',
    'offset=float(mk[0]); mid=float(w@(mk-offset))\n        self.mean=offset+mid\n        self.sd=float(np.sqrt(max(w@(vk+(mk-offset-mid)**2),1e-300)))')
assert 'w@(vk+mk**2)-self.mean**2' not in source
exec(source,globals())

def run_model(scene,name):
    start=time.perf_counter();n=raw_network();ns=r.numerical_functions();d=r.OLD/scene
    co=dict(np.load(d/'voltage_coordinates.npz'));model=r.read_models(d/'models_K2.pkl')[name]
    forms=fixed_Q_forms(co,n);tm=ns['terms'](model,r.MODELS[name],forms)
    ref=np.load(d/'mc_outputs.npy',mmap_mode='r')
    saved=np.load(d/f'CDF_{name}_K2.npz');thresholds=saved['thresholds'][64:96,:100].copy();old=saved['CDF'][64:96,:100].copy()
    MC=saved['MC_CDF'][64:96,:100].copy();features=pd.read_csv(d/'MC_electrical_features.csv')
    rows=[];points=[];quant=[];numerical=[]
    for j in range(32):
        prev=StableCDFKernel(tm[j]);Fprev=prev.cdf(thresholds[j]);factor=1.;difference=np.inf
        for factor in [2.,4.,8.,16.]:
            kernel=StableCDFKernel(tm[j],cutoff_multiplier=factor);F=kernel.cdf(thresholds[j]);difference=float(abs(F-Fprev).max())
            if difference<2e-6:break
            Fprev=F;prev=kernel
        monotonic=float(np.maximum(-np.diff(F[np.argsort(thresholds[j])]),0).max())
        numerical.append(dict(scene=scene,model=name,K=2,branch=j+1,cutoff_factor=factor,
            refinement_max_CDF_difference=difference,monotonicity_defect=monotonic,
            CF_endpoint=kernel.CF_end,frequency_points=len(kernel.tau),mean_Mvar=kernel.mean,sd_Mvar=kernel.sd,
            numerical_convergence_passed=difference<2e-6 and monotonic<2e-6))
        for side,sl,flag in [('下尾',slice(0,50),'extended_lower'),('上尾',slice(50,100),'extended_upper')]:
            rows.append(dict(情形=r.CN[scene],方法=r.CN[name],K=2,支路=j+1,电气量=f'Q{j+1}',尾侧=side,
                MC尾部延伸=bool(features.iloc[64+j][flag]),
                原CDF_RMSE_百分点=float(np.sqrt(np.mean((old[j,sl]-MC[j,sl])**2))*100),
                固定无功条件下CDF_RMSE_百分点=float(np.sqrt(np.mean((F[sl]-MC[j,sl])**2))*100),
                数值收敛=numerical[-1]['numerical_convergence_passed']))
            for ii in range(50):
                index=(0 if side=='下尾' else 50)+ii
                points.append(dict(scene=scene,model=name,K=2,branch=j+1,side=side,tail_probability=r.LEVELS[ii],threshold_Mvar=thresholds[j,index],MC_CDF=MC[j,index],old_CDF=old[j,index],fixed_Q_CDF=F[index]))
        values=np.array(ref[:,64+j]);qref=np.quantile(values,r.PROBS)
        def solve(p):
            low=kernel.mean-16*kernel.sd;high=kernel.mean+16*kernel.sd
            for _ in range(20):
                if kernel.cdf([low])[0]<=p:break
                low-=2*max(high-low,1e-10)
            for _ in range(20):
                if kernel.cdf([high])[0]>=p:break
                high+=2*max(high-low,1e-10)
            root=r.optimize.brentq(lambda x:kernel.cdf([x])[0]-p,low,high,xtol=1e-13)
            return root,float(abs(kernel.cdf([root])[0]-p))
        for prob,target in zip(r.PROBS,qref):
            root,res=solve(prob)
            quant.append(dict(情形=r.CN[scene],方法=r.CN[name],K=2,支路=j+1,电气量=f'Q{j+1}',分位水平=prob,
                MC分位值_kvar=target*1000,固定无功条件下分位值_kvar=root*1000,
                分位误差_kvar=(root-target)*1000,绝对分位误差_kvar=abs(root-target)*1000,CDF反求残差=res,
                数值收敛=numerical[-1]['numerical_convergence_passed']))
    payload=dict(CDF=rows,points=points,quantiles=quant,numerical=numerical,seconds=time.perf_counter()-start)
    r.write_json(OUT/f'{scene}_{name}.json',payload)
    return scene,name,len(rows),payload['seconds'],max(x['refinement_max_CDF_difference'] for x in numerical)

def collect():
    payload=[json.loads((OUT/f'{s}_{name}.json').read_text()) for s in r.SCENES for name in r.MODELS]
    for key,filename in [('CDF','reactive_cdf.csv'),('points','reactive_tail_points.csv'),('quantiles','reactive_quantiles.csv'),('numerical','reactive_integration.csv')]:
        pd.DataFrame([x for p in payload for x in p[key]]).to_csv(OUT/filename,index=False,encoding='utf-8-sig')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--scene',choices=r.SCENES);p.add_argument('--model',choices=list(r.MODELS));args=p.parse_args()
    if args.scene:
        print(run_model(args.scene,args.model),flush=True)
    else:
        tasks=[(s,name) for s in r.SCENES for name in r.MODELS if not (OUT/f'{s}_{name}.json').exists()]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
            jobs=[pool.submit(run_model,*task) for task in tasks]
            for job in as_completed(jobs):print(job.result(),flush=True)
        collect()
