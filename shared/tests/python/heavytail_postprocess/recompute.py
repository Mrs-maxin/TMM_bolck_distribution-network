"""Re-evaluate saved IEEE33 voltage laws using tail CDF-RMSE and extreme quantiles.

Only saved MC outputs and saved models are read. No injections, power flows,
fitting, or archived source files are changed. Quantiles invert the same analytic
electrical CDF; cached characteristic functions avoid repeating its construction.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import ast,csv,json,pickle,re,time,hashlib,argparse
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
import pandas as pd
from scipy import stats,optimize
from numpy.polynomial.legendre import leggauss

ROOT=Path(__file__).resolve().parents[3]/'result/heavytail_injection/case33bw_20261026'
ARCH=ROOT/'../../../../archive/heavytail_source_snapshot_20261026'
OLD=ROOT/'run_outputs'
OUT=ROOT/'results'
SCENES=['pareto','lognormal','loglogistic','mixed']
MODELS={'AQGMM':'GMM','standard_TMM':'TMM','grouped_t_shift':'SHIFT'}
CN={'pareto':'Pareto','lognormal':'对数正态','loglogistic':'对数逻辑斯蒂','mixed':'混合',
    'AQGMM':'AQGMM','standard_TMM':'标准TMM','grouped_t_shift':'分组t'}
LEVELS=np.linspace(.001,.05,50)
PROBS=np.array([.001,.005,.01,.99,.995,.999])

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=1,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))

def extract(path,names,namespace):
    """Load specified pure functions without executing archived module entry code."""
    source=Path(path).read_text();found=set()
    for node in ast.parse(source).body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            exec(compile(ast.get_source_segment(source,node),str(path),'exec'),namespace)
            found.add(node.name)
    assert found==set(names),(path,set(names)-found)

class StrengthRecord:
    pass

class SavedModelReader(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='grouped_strength_tmm_case33' and name=='TruncatedStrength':return StrengthRecord
        return super().find_class(module,name)

def read_models(path):
    with Path(path).open('rb') as f:return SavedModelReader(f).load()

def numerical_functions():
    ns=dict(np=np,re=re,ROOT=ARCH,stats=stats,leggauss=leggauss,U_NODES=12,
        U_PIECES=[(0.,.9),(.9,.99),(.99,.999),(.999,1.)])
    extract(ARCH/'tests/python/archived/cloud_numeric_core.py',['admittance','evaluate_outputs'],ns)
    extract(ARCH/'tests/python/archived/run_typhoon_comparison_K136.py',['network','voltage'],ns)
    b,bus,branch=ns['network']()
    ns['base']=SimpleNamespace(BASE=b,BUS=bus,BRANCH=branch,evaluate_outputs=ns['evaluate_outputs'],voltage=ns['voltage'])
    extract(ARCH/'tests/python/analytic_grouped_t_case33.py',
        ['ac_quantities','quadratic_forms','u_nodes','gaussian_states','terms','cdf_at'],ns)
    def t_nodes(nu):
        if not np.isfinite(nu):return np.ones(1),np.ones(1)
        u,w=ns['u_nodes']();return nu/stats.chi2.isf(u,nu),w
    ns['t_nodes']=t_nodes
    return ns

def object_info(j):
    if j<32:return f'V{j+2}','V','p.u.',1.
    if j<64:return f'P{j-31}','P','kW',1000.
    if j<96:return f'Q{j-63}','Q','kvar',1000.
    return 'loss','loss','kW',1000.

def status(scene,k,model):
    if model=='grouped_t_shift':return '参数一致性未通过；诊断结果'
    if model=='standard_TMM' and k==1 and scene in ['pareto','mixed']:
        return '自由度触下限；四阶匹配未满足'
    return '按已保存模型计算'

def calculate_rmse():
    rows=[]
    overrides={}
    if (ROOT/'computation_records/frequency_integration_refinement_log.jsonl').exists():
        for line in (ROOT/'computation_records/frequency_integration_refinement_log.jsonl').read_text().splitlines():
            item=json.loads(line)
            if item['cutoff_factor_used']>1:overrides[(item['scene'],item['K'],item['model'],item['j'])]=item
    with (OUT/'tail_cdf_per_threshold.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['情形','方法','K','电气量','类型','尾侧','尾概率','物理阈值','阈值单位','MC_CDF','方法CDF','MC尾概率','方法尾概率','尾概率偏差_百分点']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for scene in SCENES:
            feature=pd.read_csv(OLD/scene/'MC_electrical_features.csv')
            for k in [1,2,4,6]:
                for model in MODELS:
                    a=np.load(OLD/scene/f'CDF_{model}_K{k}.npz')
                    saved_F=a['CDF'];saved_MC=a['MC_CDF'];saved_thresholds=a['thresholds']
                    assert saved_F.shape==(97,351)
                    for j in range(97):
                        obj,kind,unit,scale=object_info(j)
                        key=(scene,k,model,j)
                        corrected=np.array(overrides[key]['CDF100']) if key in overrides else None
                        for side,sl in [('下尾',slice(0,50)),('上尾',slice(50,100))]:
                            F=(corrected[sl] if corrected is not None else saved_F[j,sl]);MC=saved_MC[j,sl]
                            delta=F-MC if side=='下尾' else MC-F
                            rmse=float(np.sqrt(np.mean((F-MC)**2)))
                            rows.append(dict(情形=CN[scene],方法=CN[model],K=k,电气量=obj,类型=kind,尾侧=side,
                                CDF_RMSE=rmse,CDF_RMSE_百分点=100*rmse,平均尾概率偏差_百分点=100*float(delta.mean()),
                                MC尾部延伸=bool(feature.iloc[j]['extended_lower' if side=='下尾' else 'extended_upper']),
                                模型状态=status(scene,k,model),积分范围倍数=overrides[key]['cutoff_factor_used'] if key in overrides else 1))
                            for i,p in enumerate(LEVELS):
                                pm=F[i] if side=='下尾' else 1-F[i]
                                pc=MC[i] if side=='下尾' else 1-MC[i]
                                writer.writerow(dict(zip(fields,[CN[scene],CN[model],k,obj,kind,side,float(p),
                                    float(saved_thresholds[j,sl][i])*scale,unit,float(MC[i]),float(F[i]),float(pc),float(pm),100*float(delta[i])])))
    table=pd.DataFrame(rows);assert len(table)==9312
    table.to_csv(OUT/'tail_cdf_rmse_all_quantities.csv',index=False,encoding='utf-8-sig')
    return table

class CDFKernel:
    """The archived Gil-Pelaez quadrature, cached for repeated threshold queries."""
    def __init__(self,terms,cutoff_multiplier=1.):
        compact=[]
        for w,m,lam,beta in terms:
            keep=abs(lam)>max(float(np.max(abs(lam)))*1e-10,1e-18)
            compact.append((w,m,lam[keep],beta[keep],float(np.sum(beta[~keep]**2))))
        width=max(len(t[2]) for t in compact)+1
        lam=np.zeros((len(compact),width));beta=np.zeros_like(lam)
        for j,(_,_,l,b,v) in enumerate(compact):
            lam[j,:len(l)]=l;beta[j,:len(b)]=b;beta[j,-1]=np.sqrt(v)
        w=np.array([t[0] for t in compact]);m0=np.array([t[1] for t in compact])
        mk=m0+lam.sum(1);vk=2*(lam**2).sum(1)+(beta**2).sum(1)
        self.mean=float(w@mk)
        self.sd=float(np.sqrt(max(w@(vk+mk**2)-self.mean**2,1e-300)))
        keep=w>1e-14
        tmax=cutoff_multiplier*12/np.sqrt(vk[keep]/self.sd**2).min()
        nodes,weights=leggauss(64);edges=np.linspace(0,tmax,int(np.ceil(tmax*4))+2)
        tau=((edges[1:]-edges[:-1])[:,None]*(nodes+1)/2+edges[:-1,None]).ravel()
        wt=((edges[1:]-edges[:-1])[:,None]*weights/2).ravel()
        t=tau/self.sd;phi=np.zeros(len(t),complex)
        tail_weights=np.cumsum((wt/tau)[::-1])[::-1]
        omission_bound=0.;evaluated_points=0
        for j in np.flatnonzero(keep):
            # The magnitude of each conditional Gaussian quadratic CF decreases
            # monotonically. Omit only a tail whose discrete CDF error has a
            # direct bound below numerical precision; retained quadrature is unchanged.
            def log_modulus(index):
                value=t[index];den=1+4*value*value*lam[j]**2
                return float(-.25*np.log(den).sum()-.5*value*value*np.sum(beta[j]**2/den))
            cutoff=len(t)
            if log_modulus(len(t)-1)<np.log(1e-13):
                left,right=0,len(t)-1
                while left<right:
                    middle=(left+right)//2
                    if log_modulus(middle)<np.log(1e-13):right=middle
                    else:left=middle+1
                cutoff=min(len(t),left+1)
                if cutoff<len(t):omission_bound+=w[j]*np.exp(log_modulus(cutoff))*tail_weights[cutoff]/np.pi
            tj=t[:cutoff];z=1-2j*np.outer(tj,lam[j]);evaluated_points+=cutoff
            phi[:cutoff]+=w[j]*np.exp(1j*tj*(m0[j]-self.mean)-.5*np.log(z).sum(1)-
                (np.outer(tj*tj,beta[j]**2)/(2*z)).sum(1))
        self.tau=tau;self.coeff=phi*wt/tau/np.pi
        self.CF_end=float(abs(phi[-1]))
        self.omission_bound=float(omission_bound)
        self.evaluated_points=evaluated_points

    def cdf(self,values):
        y=(np.atleast_1d(values)-self.mean)/self.sd
        result=np.empty(len(y))
        for start in range(0,len(y),16):
            result[start:start+16]=.5-np.imag(np.exp(-1j*np.outer(y[start:start+16],self.tau))@self.coeff)
        return np.clip(result,0,1)

def solve_object(payload):
    scene,k,model,j,terms,thresholds,saved_cdf=payload
    t0=time.perf_counter();kernel=CDFKernel(terms)
    x=np.asarray(thresholds)**2 if j<32 else np.asarray(thresholds)
    order=np.argsort(x);xx,keep=np.unique(x[order],return_index=True);ff=np.asarray(saved_cdf)[order][keep]
    # Cross-check cached CDF against previously saved values before solving quantiles.
    probe=np.unique(np.r_[0,49,50,99,np.arange(100,351,25)]).astype(int)
    agreement=float(np.max(abs(kernel.cdf(x[probe])-np.asarray(saved_cdf)[probe])))
    assert agreement<2e-7,(scene,k,model,j,'CDF mismatch',agreement)
    q=[];res=[];brackets=[]
    for p in PROBS:
        loc=int(np.searchsorted(ff,p))
        lo=xx[max(0,loc-1)];hi=xx[min(len(xx)-1,loc)]
        if loc==0:lo=min(lo,kernel.mean-8*kernel.sd)
        if loc==len(xx):hi=max(hi,kernel.mean+8*kernel.sd)
        if j<32:lo=max(0.,lo)
        flo=float(kernel.cdf([lo])[0]);fhi=float(kernel.cdf([hi])[0])
        step=max(kernel.sd,hi-lo,1e-12)
        for _ in range(25):
            if flo<=p<=fhi and hi>lo:break
            if flo>p:
                lo=lo-step
                if j<32:lo=max(0.,lo)
                flo=float(kernel.cdf([lo])[0])
            if fhi<p or hi<=lo:
                hi+=step;fhi=float(kernel.cdf([hi])[0])
            step*=2
        else:raise RuntimeError((scene,k,model,j,p,'no bracket',lo,hi,flo,fhi))
        root=optimize.brentq(lambda value:float(kernel.cdf([value])[0])-p,lo,hi,
                            xtol=max(kernel.sd*1e-9,1e-13),rtol=1e-12,maxiter=120)
        residual=abs(float(kernel.cdf([root])[0])-p)
        assert residual<1e-6,(scene,k,model,j,p,residual)
        q.append(float(np.sqrt(max(root,0))) if j<32 else float(root));res.append(residual);brackets.append([float(lo),float(hi)])
    return dict(scene=scene,K=k,model=model,j=j,quantiles=q,CDF_residual=res,
        old_CDF_match_max=agreement,CF_end=kernel.CF_end,frequency_points=len(kernel.tau),
        seconds=time.perf_counter()-t0,brackets=brackets,
        conditional_CF_evaluations=kernel.evaluated_points,omitted_CDF_bound=kernel.omission_bound)

def reference_quantiles():
    result={};rows=[]
    for scene in SCENES:
        mc=np.load(OLD/scene/'mc_outputs.npy',mmap_mode='r');values=[]
        n=len(mc);rank_lo=np.maximum(0,stats.binom.ppf(.025,n,PROBS).astype(int)-1)
        rank_hi=np.minimum(n-1,stats.binom.ppf(.975,n,PROBS).astype(int))
        for j in range(97):
            raw=np.array(mc[:,j]);q=np.quantile(raw,PROBS,method='linear');values.append(q)
            ordered=np.partition(raw,np.unique(np.r_[rank_lo,rank_hi]));lo=ordered[rank_lo];hi=ordered[rank_hi]
            obj,kind,unit,scale=object_info(j)
            for p,v,l,h in zip(PROBS,q,lo,hi):rows.append(dict(情形=CN[scene],电气量=obj,类型=kind,分位水平=p,
                MC分位值=v*scale,MC分位下限_95=l*scale,MC分位上限_95=h*scale,单位=unit))
        result[scene]=np.array(values)
    pd.DataFrame(rows).to_csv(OUT/'mc_extreme_quantiles_ci95.csv',index=False,encoding='utf-8-sig')
    return result

def calculate_quantiles(workers=6,only_scene=None,only_k=None):
    log=ROOT/'computation_records/analytic_quantile_log.jsonl';done={}
    if log.exists():
        for line in log.read_text().splitlines():
            if line:
                item=json.loads(line);done[(item['scene'],item['K'],item['model'],item['j'])]=item
    ns=numerical_functions();count=len(done)
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool,log.open('a') as stream:
        pending={};remaining={}
        def accept(future):
            nonlocal count
            item=future.result();key=(item['scene'],item['K'],item['model'],item['j'])
            done[key]=item;stream.write(json.dumps(item)+'\n');stream.flush();count+=1
            modelkey=key[:3];remaining[modelkey]-=1
            if count%50==0:print(json.dumps(dict(stage='quantiles',completed=count,total=4656,scene=key[0],K=key[1],model=key[2])),flush=True)
            if remaining[modelkey]==0:print(json.dumps(dict(stage='model_quantiles_done',scene=key[0],K=key[1],model=key[2],completed=count)),flush=True)
            del pending[future]
        for scene in ([only_scene] if only_scene else SCENES):
            coords=dict(np.load(OLD/scene/'voltage_coordinates.npz'));forms=ns['quadratic_forms'](coords)
            assert forms['check']<1e-8
            for k in ([only_k] if only_k else [1,2,4,6]):
                models=read_models(OLD/scene/f'models_K{k}.pkl')
                for model,kind in MODELS.items():
                    missing=[j for j in range(97) if (scene,k,model,j) not in done]
                    if not missing:continue
                    saved=np.load(OLD/scene/f'CDF_{model}_K{k}.npz');thresholds=saved['thresholds'];cdf=saved['CDF']
                    tm=ns['terms'](models[model],kind,forms)
                    remaining[(scene,k,model)]=len(missing)
                    for j in missing:
                        pending[pool.submit(solve_object,(scene,k,model,j,tm[j],thresholds[j],cdf[j]))]=(scene,k,model,j)
                    for future in [f for f in pending if f.done()]:accept(future)
        for future in as_completed(list(pending)):accept(future)
    return done

def export_quantiles(done):
    ref=reference_quantiles();rows=[];checks=[]
    for key,item in sorted(done.items()):
        scene,k,model,j=key;obj,kind,unit,scale=object_info(j)
        assert np.all(np.diff(item['quantiles'])>=0)
        checks.append(dict(情形=CN[scene],方法=CN[model],K=k,电气量=obj,
            最大CDF反求残差=max(item['CDF_residual']),原积分设置CDF复现差=item['old_CDF_match_max'],
            特征函数终点模=item['CF_end'],积分点数=item['frequency_points'],计算秒=item['seconds'],
            积分范围倍数=item.get('cutoff_multiplier',1),频域复核状态='已通过' if item.get('frequency_refinement_checked') else '未触发',
            频域加密复核差=item.get('frequency_refinement_max_difference'),遗漏项CDF上界=item.get('omitted_CDF_bound',0.)))
        for i,(p,value) in enumerate(zip(PROBS,item['quantiles'])):
            mc=float(ref[scene][j,i]);error=(value-mc)*scale
            rows.append(dict(情形=CN[scene],方法=CN[model],K=k,电气量=obj,类型=kind,尾侧='下尾' if p<.5 else '上尾',
                尾概率=float(round(p if p<.5 else 1-p,12)),分位水平=float(p),MC分位值=mc*scale,方法分位值=value*scale,
                分位误差=error,绝对分位误差=abs(error),单位=unit,CDF反求残差=item['CDF_residual'][i],模型状态=status(scene,k,model)))
    pd.DataFrame(rows).to_csv(OUT/'extreme_quantile_error_all_quantities.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(ROOT/'computation_records/analytic_cdf_quantile_solve_check.csv',index=False,encoding='utf-8-sig')
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['rmse','quantiles','export','all'],default='all')
    parser.add_argument('--workers',type=int,default=6);parser.add_argument('--scene',choices=SCENES);parser.add_argument('--k',type=int,choices=[1,2,4,6]);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.phase in ['rmse','all']:
        table=calculate_rmse();print(json.dumps(dict(stage='CDF_RMSE_done',rows=len(table))),flush=True)
    if args.phase in ['quantiles','all']:
        done=calculate_quantiles(args.workers,args.scene,args.k)
        if len(done)==4656:export_quantiles(done)
    if args.phase=='export':
        done={}
        for line in (ROOT/'computation_records/analytic_quantile_log.jsonl').read_text().splitlines():
            item=json.loads(line);done[(item['scene'],item['K'],item['model'],item['j'])]=item
        assert len(done)==4656,len(done);export_quantiles(done)

if __name__=='__main__':main()
