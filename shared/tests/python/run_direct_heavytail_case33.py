"""Independent IEEE33 theoretical injection study; original project remains read-only.

Full positive generation distributions, fixed original P/Q demand, prescribed
Pearson correlation. Reuses the snapshotted voltage models and electrical forms.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import sys
sys.dont_write_bytecode = True
import json, time, hashlib, platform, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, special, optimize
from scipy.special import roots_hermitenorm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'result/heavytail_injection/case33bw_20261026/run_outputs'
OUT.mkdir(parents=True, exist_ok=True)
# The archived voltage-model module creates a diagnostic child on import.
# Create its empty parent inside this independent copy; no historical data are used.
(ROOT/'result/case33bw/pv_nrel_case33_newsplit_20261022/nrel_ha4_40_60').mkdir(parents=True, exist_ok=True)
sys.argv_saved = sys.argv[:]
sys.argv = sys.argv[:1]
import grouped_strength_tmm_case33 as gs
import shift_grouped_t_case33 as sh
import analytic_grouped_t_case33 as an
from fast_voltage_map import fast_map
from cloud_numeric_core import initialize, fit, log_density
sys.argv = sys.argv_saved
base = gs.base
NODES = np.array([18,17,16,15,14,13,33,22])
SCENES = {'pareto':['pareto']*8, 'lognormal':['lognormal']*8,
          'loglogistic':['loglogistic']*8,
          'mixed':['pareto','lognormal','loglogistic','pareto','lognormal','loglogistic','pareto','lognormal']}
MEAN, SD, RHO = .25, .25/np.sqrt(15), .3
LEVELS = np.linspace(.001,.05,50)
GRID = np.linspace(.001,.999,251)
KS = [1,2,4,6]

def dump(path, data):
    def convert(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,Path): return str(v)
        raise TypeError(type(v).__name__)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=1,default=convert))

def event(stage, **kw):
    row=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    print(json.dumps(row,ensure_ascii=False,default=str),flush=True)
    with (OUT/'progress.jsonl').open('a') as f: f.write(json.dumps(row,default=str)+'\n')

def laws():
    cv=SD/MEAN
    shape=optimize.brentq(lambda a: np.sqrt(stats.fisk(a).var())/stats.fisk(a).mean()-cv,4.01,50)
    sigma=np.sqrt(np.log1p(cv**2))
    return {'pareto':stats.pareto(5,scale=.2),
            'lognormal':stats.lognorm(sigma,scale=MEAN*np.exp(-sigma**2/2)),
            'loglogistic':stats.fisk(shape,scale=MEAN/stats.fisk(shape).mean())}

LAWS=laws()

def quantile_z(name,z):
    rv=LAWS[name]
    if name=='pareto': return rv.kwds['scale']*np.exp(-special.log_ndtr(-z)/5)
    if name=='lognormal': return rv.kwds['scale']*np.exp(rv.args[0]*z)
    return rv.kwds['scale']*np.exp((special.log_ndtr(z)-special.log_ndtr(-z))/rv.args[0])

def corr_integral(a,b,rho,order=80):
    z,w=roots_hermitenorm(order); w=w/np.sqrt(2*np.pi)
    x=(quantile_z(a,z)-MEAN)/SD
    y=(quantile_z(b,rho*z[:,None]+np.sqrt(1-rho*rho)*z[None,:])-MEAN)/SD
    return float((w*x)@y@w)

def correlation(names):
    R=np.eye(8); cache={}; checks=[]
    for i in range(8):
        for j in range(i):
            pair=tuple(sorted([names[i],names[j]]))
            if pair not in cache:
                r=optimize.brentq(lambda v:corr_integral(*pair,v)-RHO,.00001,.95,xtol=1e-10)
                check=corr_integral(*pair,r,120)
                assert abs(check-RHO)<2e-5,(pair,check)
                cache[pair]=r;checks.append(dict(pair=pair,normal_correlation=r,power_correlation_check=check))
            R[i,j]=R[j,i]=cache[pair]
    assert np.linalg.eigvalsh(R).min()>0
    return R,checks

def injection(power):
    t=np.tile(np.r_[-base.BUS[1:,2],-base.BUS[1:,3]]/base.BASE,(len(power),1))
    t[:,NODES-2]+=power/base.BASE
    return t

def draw(names,R,n,seed):
    z=np.random.default_rng(seed).standard_normal((n,8))@np.linalg.cholesky(R).T
    p=np.column_stack([quantile_z(name,z[:,j]) for j,name in enumerate(names)])
    assert np.isfinite(p).all() and (p>=0).all()
    return p

ACTIVE=base.BRANCH[base.BRANCH[:,10]>0]
assert len(ACTIVE)==32 and np.all(ACTIVE[:,4]==0) and np.all(ACTIVE[:,8]==0)
assert np.all(base.BUS[:,4:6]==0)
F=ACTIVE[:,0].astype(int)-1; T=ACTIVE[:,1].astype(int)-1
Z=ACTIVE[:,2]+1j*ACTIVE[:,3]
assert np.all(F<T) and len(set(T))==32

def radial_ac(target,tolerance=2e-11):
    """Separate full nonlinear AC solution for each scenario by radial sweeps."""
    s=np.column_stack([np.zeros(len(target)),target[:,:32]+1j*target[:,32:]])
    v=np.ones((len(target),33),complex)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        for it in range(150):
            old=v.copy(); currents=-np.conj(s/v)
            for k in range(31,-1,-1): currents[:,F[k]]+=currents[:,T[k]]
            for k in range(32): v[:,T[k]]=v[:,F[k]]-Z[k]*currents[:,T[k]]
            change=np.max(abs(v-old),axis=1)
            if np.isfinite(change).all() and change.max()<tolerance: break
    calc=v*np.conj(v@base.Y.T)
    residual=np.max(abs(np.c_[calc[:,1:].real,calc[:,1:].imag]-target),axis=1)
    failed=np.flatnonzero((residual>1e-10)|~np.isfinite(residual))
    failures=[]
    for idx in failed:
        try: v[idx],_=base.polar_ac(target[idx],base.Y,tolerance=1e-11)
        except (RuntimeError,np.linalg.LinAlgError): failures.append(int(idx))
    if failures: raise RuntimeError(f'AC failed; preserve injections and stop, indices={failures[:20]}')
    calc=v*np.conj(v@base.Y.T)
    residual=np.max(abs(np.c_[calc[:,1:].real,calc[:,1:].imag]-target))
    assert residual<1e-10,residual
    return v,dict(max_power_residual_pu=float(residual),newton_fallback=len(failed),iterations=it+1)

def protocol():
    p=OUT/'protocol.json'
    record=dict(version='direct-generation-heavy-tails-v1',nodes=NODES,load_multiplier=1.,load_randomness=False,
        load_P_MW=base.BUS[1:,2],load_Q_MVAr=base.BUS[1:,3],load_totals=[base.BUS[:,2].sum(),base.BUS[:,3].sum()],
        generation_Q_MVAr=0.,scenes=SCENES,station_mean_MW=MEAN,station_sd_MW=SD,power_Pearson_correlation=RHO,
        marginal_parameters={n:dict(args=rv.args,kwds=rv.kwds,mean=rv.mean(),sd=rv.std(),skew=rv.stats(moments='s'),
                                      excess_kurtosis=rv.stats(moments='k')) for n,rv in LAWS.items()},
        splits=dict(estimation=8192,selection=4096,mc=1000000),K=KS,gmm_starts=6,
        models=['AQGMM','standard_TMM_moment','grouped_t_shift_current'],
        input_clipping=False,input_recentering=False,input_outlier_removal=False,
        correlation_assumption='Gaussian copula, Nataf calibration to injection-power Pearson coefficients; not a tail-dependence study',
        voltage_coordinates='SVD retained singular values > 1e-8 times largest; independent selection reconstruction checked',
        MC_solver='per-scenario nonlinear radial AC sweeps, residual <1e-10 pu, independent polar Newton audit',
        fit_voltage='MATLAB fast mapping with original default tolerances; only independent snapshot used',
        grouped_model_scope='Existing method unchanged apart from numerical rank support; internal strength cap estimated only from estimation+selection injections; input laws remain unbounded. This cap is a model approximation, not a generator limit.',
        tail_screen='MC only before model error: q0.5%/q99.5% distance from median divided by SD*normal q99.5% >1.15; finite-range extension, not proof of asymptotic heavy tail',
        metric='At 50 MC thresholds from 0.1% to 5% per side: mean absolute probability error; report percentage points; full CDF grid error retained for every object',
        selection_rule='Gaussian joint voltage likelihood on independent selection scenarios; no MC output errors used',
        representative_rule='Before fitting: strongest extended V, P, Q side plus V18,V33,P1,P17,P24,Q17,loss; all objects retained',
        uncertainty='One independent scenario realization per input law; no claim of repeat stability; MC95% pointwise Wilson intervals at each tail threshold',
        frozen_before_computation=True,source_snapshot_manifest='source_snapshot_manifest.json',
        environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__),
        sources=['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html',
                 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisk.html',
                 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html',
                 'https://uqpyproject.readthedocs.io/en/latest/transformations/nataf.html'])
    if p.exists():
        assert json.loads(p.read_text())['version']==record['version']
    else:
        dump(p,record)
        dump(ROOT/'01_IDEA/direct_heavytail_protocol_20261026.json',record)
    return record

def prepare(scene):
    d=OUT/scene;d.mkdir(exist_ok=True)
    if (d/'prepared.json').exists(): return
    names=SCENES[scene]; R,checks=correlation(names); si=list(SCENES).index(scene)
    dump(d/'correlation.json',dict(power_target=(1-RHO)*np.eye(8)+RHO,latent_normal=R,checks=checks))
    event('input_law_locked',scene=scene)
    splits=[]
    for k,(label,n) in enumerate([('estimation',8192),('selection',4096)]):
        p=draw(names,R,n,2026102600+100*si+k);splits.append(p)
    power=np.concatenate(splits);t=injection(power)
    v,info=fast_map(t)
    audit=[]
    ids=np.r_[np.arange(64),np.argsort(power.max(1))[-64:]]
    for idx in ids:
        vr,_=base.polar_ac(t[idx],base.Y,tolerance=1e-11)
        audit.append(float(np.max(abs(vr-v[idx]))))
    dump(d/'mapping_audit.json',dict(**info,polar_check_count=len(ids),max_voltage_difference_pu=max(audit)))
    assert max(audit)<2e-6
    for k,label in enumerate(['estimation','selection']):
        sl=slice(0,8192) if k==0 else slice(8192,None)
        np.savez_compressed(d/f'{label}.npz',power_MW=power[sl],target=t[sl],voltage=v[sl])
    event('mapping_done',scene=scene,**info)
    n=1000000;p=draw(names,R,n,2026102600+100*si+2)
    np.save(d/'mc_injection_MW.npy',p)
    mc=np.lib.format.open_memmap(d/'mc_outputs.npy',mode='w+',dtype='float64',shape=(n,97))
    maxres=0.; fallback=0;t0=time.perf_counter();newtondiff=[]
    for i in range(0,n,8192):
        target=injection(p[i:i+8192]);vol,inf=radial_ac(target)
        if i==0:
            for j in range(64):
                vr,_=base.polar_ac(target[j],base.Y,tolerance=1e-11)
                newtondiff.append(float(np.max(abs(vr-vol[j]))))
            assert max(newtondiff)<1e-8
            assert np.max(abs(base.evaluate_outputs(vol[:64],base.BASE,base.BUS,base.BRANCH)-
                                 base.independent_outputs(vol[:64],base.BASE,base.BRANCH)))<1e-8 if hasattr(base,'independent_outputs') else True
        mc[i:i+len(vol)]=base.outputs(vol)
        maxres=max(maxres,inf['max_power_residual_pu']);fallback+=inf['newton_fallback']
        if i%131072==0: event('MC_AC',scene=scene,completed=min(i+8192,n),seconds=round(time.perf_counter()-t0,1))
    mc.flush()
    corr=np.corrcoef(p.T)
    pd.DataFrame(corr,index=NODES,columns=NODES).to_csv(d/'realized_power_correlation.csv')
    assert np.max(abs(corr-((1-RHO)*np.eye(8)+RHO)))<.025
    rows=[]
    for j,node in enumerate(NODES):
        rows.append(dict(node=node,distribution=names[j],mean_MW=p[:,j].mean(),sd_MW=p[:,j].std(),
                         q001_MW=np.quantile(p[:,j],.001),q999_MW=np.quantile(p[:,j],.999),max_MW=p[:,j].max(),
                         theoretical_mean_MW=MEAN,theoretical_sd_MW=SD))
    pd.DataFrame(rows).to_csv(d/'injection_statistics.csv',index=False)
    dump(d/'prepared.json',dict(mc_scenarios=n,max_power_residual_pu=maxres,newton_fallback=fallback,
        radial_vs_polar_max_pu=max(newtondiff),mc_seconds=time.perf_counter()-t0,
        voltage_min_pu=float(mc[:,:32].min()),voltage_max_pu=float(mc[:,:32].max()),
        power_corr_max_error=float(np.max(abs(corr-((1-RHO)*np.eye(8)+RHO))))))
    event('prepared',scene=scene,seconds=round(time.perf_counter()-t0,1))

def object_name(j):
    if j<32:return f'V{j+2}'
    if j<64:return f'P{j-31}'
    if j<96:return f'Q{j-63}'
    return 'loss'

def reference(scene):
    d=OUT/scene;mc=np.load(d/'mc_outputs.npy',mmap_mode='r');refs=[];rows=[]
    for j in range(97):
        r=np.sort(np.array(mc[:,j]));sd=r.std();med=np.median(r)
        qs=np.quantile(r,np.r_[LEVELS,1-LEVELS,GRID]);L=len(LEVELS)
        thresholds=qs
        actual=np.searchsorted(r,thresholds,side='right')/len(r)
        lo=(med-np.quantile(r,.005))/sd/stats.norm.ppf(.995)
        hi=(np.quantile(r,.995)-med)/sd/stats.norm.ppf(.995)
        refs.append(dict(thresholds=thresholds,actual=actual,sd=sd))
        rows.append(dict(object=object_name(j),kind='V' if j<32 else 'P' if j<64 else 'Q' if j<96 else 'loss',
            mean=r.mean(),sd=sd,skew=stats.skew(r),excess_kurtosis=stats.kurtosis(r),extension_lower=lo,extension_upper=hi,
            extended_lower=bool(lo>1.15),extended_upper=bool(hi>1.15),q001=np.quantile(r,.001),q999=np.quantile(r,.999)))
    tab=pd.DataFrame(rows);tab.to_csv(d/'MC_electrical_features.csv',index=False)
    reps=[16,31,32,48,55,80,96]
    for a,b in [(0,32),(32,64),(64,96)]:
        score=tab.iloc[a:b][['extension_lower','extension_upper']].max(axis=1)
        reps.append(int(score.idxmax()))
    dump(d/'representatives_locked_before_fit.json',dict(indices=sorted(set(reps)),objects=[object_name(i) for i in sorted(set(reps))]))
    return refs,tab

def coordinates(scene):
    d=OUT/scene;a=np.load(d/'estimation.npz');b=np.load(d/'selection.npz')
    state=lambda v: np.c_[v[:,1:].real,v[:,1:].imag]
    st,sv=state(a['voltage']),state(b['voltage']);center=st.mean(0)
    _,s,vt=np.linalg.svd(st-center,full_matrices=False);keep=s>s[0]*1e-8
    basis=vt[keep].T;scale=s[keep]/np.sqrt(len(st)-1)
    x=(st-center)@basis/scale;xv=(sv-center)@basis/scale
    reconstruction=float(abs((xv*scale)@basis.T+center-sv).max())
    assert reconstruction<1e-7
    np.savez_compressed(d/'voltage_coordinates.npz',center=center,basis=basis,scale=scale,singular_values=s)
    dump(d/'coordinate_audit.json',dict(retained=int(keep.sum()),original=64,selection_reconstruction_max_pu=reconstruction))
    return dict(train=a['target'],every=np.r_[a['target'],b['target']],x=x,xv=xv,basis=basis,scale=scale,center=center,voltage=a['voltage'])

def fit_gmm(data,k):
    best=None;records=[]
    for start in range(1 if k==1 else 6):
        ini,_,info=initialize(data['x'],k,start,data['scale'],population_covariance=True)
        m,a=fit(data['x'],ini,actual_covariance_floor=True,relative_tolerance=True)
        score=float(log_density(data['xv'],m)[0].mean())
        records.append(dict(start=start,status=a['status'],iterations=a['iterations'],selection_ll=score,rule=info['rule']))
        if best is None or score>best[0]:best=(score,m,start)
    return best[1],dict(chosen_start=best[2],attempts=records)

def compressed_cdf(tm,thresholds):
    """Exact low-rank compression of Gaussian quadratic forms, retaining linear variance."""
    packed=[]
    for w,m,lam,beta in tm:
        tol=max(np.max(abs(lam))*1e-10,1e-18)
        keep=abs(lam)>tol
        packed.append((w,m,lam[keep],beta[keep],float(np.sum(beta[~keep]**2))))
    width=max(len(t[2]) for t in packed)+1
    compact=[]
    for w,m,lam,beta,var in packed:
        ll=np.zeros(width);bb=np.zeros(width);ll[:len(lam)]=lam;bb[:len(beta)]=beta;bb[-1]=np.sqrt(var)
        compact.append((w,m,ll,bb))
    return an.cdf_at(compact,thresholds)

def t_nodes(nu):
    if not np.isfinite(nu):return np.ones(1),np.ones(1)
    u,w=an.u_nodes()
    return nu/stats.chi2.isf(u,nu),w

an.t_nodes=t_nodes

def evaluate(model,kind,forms,refs,features):
    terms=an.terms(model,kind,forms);rows=[];curve=[];L=len(LEVELS)
    for j,r in enumerate(refs):
        threshold=r['thresholds'];F,decay=compressed_cdf(terms[j],threshold**2 if j<32 else threshold)
        lower=F[:L]-r['actual'][:L];upper=r['actual'][L:2*L]-F[L:2*L]
        rows.append(dict(object=object_name(j),kind=features.iloc[j]['kind'],
            lower_tail_pp=100*np.mean(abs(lower)),upper_tail_pp=100*np.mean(abs(upper)),
            lower_signed_pp=100*np.mean(lower),upper_signed_pp=100*np.mean(upper),
            full_CDF_max_pp=100*np.max(abs(F[2*L:]-r['actual'][2*L:])),CF_end_abs=decay,
            extended_lower=bool(features.iloc[j].extended_lower),extended_upper=bool(features.iloc[j].extended_upper)))
        curve.append(F)
    return pd.DataFrame(rows),np.array(curve)

def compare(scene):
    d=OUT/scene
    if (d/'comparison_complete.json').exists():return
    refs,features=reference(scene);data=coordinates(scene);rng=np.random.default_rng(2026102699)
    anchor=int(np.argmin(((data['train']-data['train'].mean(0))**2).sum(1)))
    split,Q1,Q2=gs.chk.split(data['train'],data['x'],data['basis'],data['scale'],rng,scene,
        group_target=data['every'],return_q=True,v0=data['voltage'][anchor])
    groups=[np.array(g) for g in split['group_cols']]
    if not len(groups):
        raise RuntimeError('No heavy-tail direction identified by existing method; retain MC results and diagnose before changing method')
    caps=gs.ts.strength_cap(data['every'],split['groups']) if hasattr(gs,'ts') else None
    if caps is None:
        import truncated_strength_case33 as ts
        caps=ts.strength_cap(data['every'],split['groups'])
    dump(d/'split.json',dict(split,internal_strength_caps=caps,cap_basis='estimation+selection only',anchor_index=anchor))
    forms=an.quadratic_forms(data);assert forms['check']<1e-8
    dump(d/'quadratic_audit.json',dict(physical_forms_max_error=forms['check']))
    event('coordinates_ready',scene=scene,dimension=len(data['scale']),groups=split['groups'])
    allrows=[];timings=[]
    for k in KS:
        t0=time.perf_counter();g,info=fit_gmm(data,k);np.savez_compressed(d/f'AQGMM_K{k}.npz',**g)
        dump(d/f'fit_K{k}.json',info)
        standard,_,_=gs.mm.moment_models(data['x'],g,Q1,Q2)
        shifted=sh.shift_grouped_model(data['x'],g,Q1,Q2,groups,caps)
        import pickle
        with (d/f'models_K{k}.pkl').open('wb') as f:pickle.dump(dict(AQGMM=g,standard_TMM=standard,grouped_t_shift=shifted),f)
        fitsec=time.perf_counter()-t0
        for name,kind,model in [('AQGMM','GMM',g),('standard_TMM','TMM',standard),('grouped_t_shift','SHIFT',shifted)]:
            path=d/f'objects_{name}_K{k}.csv'
            if path.exists():
                per=pd.read_csv(path)
            else:
                t1=time.perf_counter();per,curves=evaluate(model,kind,forms,refs,features)
                per.to_csv(path,index=False)
                np.savez_compressed(d/f'CDF_{name}_K{k}.npz',CDF=curves,thresholds=np.array([r['thresholds'] for r in refs]),
                                    MC_CDF=np.array([r['actual'] for r in refs]))
                timings.append(dict(K=k,model=name,analytic_seconds=time.perf_counter()-t1,common_fit_seconds=fitsec))
                event('model_done',scene=scene,K=k,model=name,seconds=round(time.perf_counter()-t1,1))
            allrows.append(per.assign(scene=scene,K=k,model=name))
    pd.concat(allrows,ignore_index=True).to_csv(d/'all_object_errors.csv',index=False)
    pd.DataFrame(timings).to_csv(d/'computation_times.csv',index=False)
    dump(d/'comparison_complete.json',dict(K=KS,models=3,objects=97,retained_dimension=len(data['scale']),single_realization=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['protocol','prepare','compare','all'])
    parser.add_argument('--scene',choices=list(SCENES));a=parser.parse_args()
    protocol()
    if a.phase=='protocol':return
    for scene in ([a.scene] if a.scene else SCENES):
        if a.phase in ['prepare','all']:prepare(scene)
        if a.phase in ['compare','all']:compare(scene)
    event('requested_phase_done',phase=a.phase)

if __name__=='__main__':main()
