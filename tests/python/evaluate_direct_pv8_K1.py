"""Evaluate saved K=1 analytic power flow; no source/load, AC or fitting rerun.

Definitions: 01_IDEA/plans/direct_pv8_K1_evaluation_20261026.md.
"""
from pathlib import Path
import hashlib
import json
import math
import platform
import time
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.integrate import simpson,quad
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'result/case33bw/direct_pv8_python_cloud_20261026'
OUT=SOURCE/'K1_evaluation_20261026'
MODELS=['AQGMM','Standard_TMM','Grouped_TMM']
PROBS=[.001,.005,.01,.99,.995,.999]


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def quadratic_moments(a):
    w,m,lam,beta=(a[k] for k in ('weights','m0','lam','beta'))
    mk=m+lam.sum(1)
    vk=2*(lam**2).sum(1)+(beta**2).sum(1)
    mean=float(w@mk)
    return mean,float(w@(vk+(mk-mean)**2))


def amplitude_mean(a):
    w,m,lam,beta=(a[k] for k in ('weights','m0','lam','beta'))
    def log_laplace(t):
        z=1+2*t*lam
        assert z.min()>0
        return -t*m-.5*np.log(z).sum(1)+.5*t*t*(beta**2/z).sum(1)
    def integrand(u):
        return float(w@(-np.expm1(log_laplace(u*u))))/(u*u*np.sqrt(np.pi))
    limit=64.
    log_missing=float(logsumexp(np.log(w)+log_laplace(limit*limit))-np.log(limit*np.sqrt(np.pi)))
    missing=max(float(np.exp(log_missing)),float(np.nextafter(0.,1.)))
    while missing>=1e-11 and limit<8192:
        limit*=2
        log_missing=float(logsumexp(np.log(w)+log_laplace(limit*limit))-np.log(limit*np.sqrt(np.pi)))
        missing=max(float(np.exp(log_missing)),float(np.nextafter(0.,1.)))
    assert missing<1e-11,missing
    value,error=quad(integrand,0,limit,epsabs=2e-12,epsrel=2e-12,limit=200)
    coarse,_=quad(integrand,0,limit,epsabs=1e-9,epsrel=1e-9,limit=200)
    return value+1/(limit*np.sqrt(np.pi)),error,missing,abs(value-coarse)


def raw_q8(a):
    w,m,lam,beta=(a[k] for k in ('weights','m0','lam','beta'))
    cumulants=[np.zeros(len(w)),m+lam.sum(1)]
    for r in range(2,9):
        cumulants.append(2**(r-1)*math.factorial(r-1)*(lam**r).sum(1)
                         +math.factorial(r)*2**(r-3)*(beta**2*lam**(r-2)).sum(1))
    raw=[np.ones(len(w))]
    for n in range(1,9):raw.append(sum(math.comb(n-1,j-1)*cumulants[j]*raw[n-j] for j in range(1,n+1)))
    return float(w@raw[8])


def markdown(frame):
    # No optional table-formatting dependency.
    cols=list(frame.columns)
    fmt=lambda v:f'{v:.8g}' if isinstance(v,(float,np.floating)) else str(v)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+''.join('| '+' | '.join(fmt(v) for v in row)+' |\n' for row in frame.itertuples(index=False,name=None))


def main():
    started=time.perf_counter();OUT.mkdir(exist_ok=True)
    files=[SOURCE/'ac_results.npz',SOURCE/'models/K1.pkl']
    before={str(p.relative_to(ROOT)):digest(p) for p in files}
    inventory=pd.read_csv(SOURCE/'tail_inventory_all97.csv')
    ac=np.load(SOURCE/'ac_results.npz')['reference'];assert ac.shape==(3504,97)
    tails=[];quantiles=[];moments=[];checks=[];points=[];hashes={}
    for obj in inventory.to_dict('records'):
        index,name,kind=int(obj['index']),obj['object'],obj['kind']
        values=np.sort(ac[:,index]);acmean=float(values.mean());acvar=float(values.var(ddof=0))
        identity=dict(object=name,index=index,kind=kind,from_bus=int(obj['from_bus']),to_bus=int(obj['to_bus']),unit='p.u.' if kind=='V' else 'MVAr' if kind=='Q' else 'MW')
        for model in MODELS:
            p=SOURCE/'analytic_only/K1'/model/f'{name}.csv'
            data=pd.read_csv(p,float_precision='round_trip');x=data.value.to_numpy();pdf=data.pdf.to_numpy();F0=data.cdf.to_numpy()
            unique,first=np.unique(x,return_index=True)
            duplicate_count=len(x)-len(unique)
            if duplicate_count:
                assert max(np.ptp(F0[x==z]) for z in x[:-1][np.diff(x)==0])<1e-10
                x,pdf,F0=x[first],pdf[first],F0[first]
            F=np.maximum.accumulate(F0)
            correction=float(abs(F-F0).max());assert correction<1e-12
            fun=PchipInterpolator(x,F,extrapolate=False)
            qlinear=lambda prob:float(np.interp(prob,F,x))
            levels=(np.linspace(.01,.05,81),np.linspace(.95,.99,81))
            tailrow=dict(identity,model=model)
            for side,probs in zip(('left','right'),levels):
                thresholds=np.quantile(values,probs,method='linear')
                assert thresholds.min()>=x[0] and thresholds.max()<=x[-1]
                actual=np.searchsorted(values,thresholds,side='right')/len(values)
                predicted=fun(thresholds)
                linear=np.interp(thresholds,x,F)
                rmse=float(np.sqrt(np.mean((predicted-actual)**2)))
                sensitivity=float(abs(predicted-linear).max())
                tailrow.update({f'{side}_CDF_RMSE':rmse,f'{side}_CDF_RMSE_percentage_points':100*rmse,
                    f'{side}_max_interpolation_sensitivity':sensitivity,
                    f'{side}_AC_nominal_CDF_max_difference':float(abs(actual-probs).max())})
                for prob,z,fa,fm in zip(probs,thresholds,actual,predicted):
                    points.append(dict(identity,model=model,side=side,nominal_probability=prob,AC_quantile=z,AC_right_continuous_CDF=fa,model_CDF=fm,signed_CDF_error=fm-fa))
            tails.append(tailrow)
            max_residual=0.;max_qsens=0.
            for prob in PROBS:
                assert F[0]<prob<F[-1],(name,model,prob)
                j=int(np.searchsorted(F,prob,side='left'));lo,hi=x[j-1],x[j]
                q=float(brentq(lambda z:float(fun(z))-prob,lo,hi,xtol=1e-14,rtol=1e-14))
                qa=float(np.quantile(values,prob,method='linear'));error=q-qa
                residual=abs(float(fun(q))-prob);sensitivity=abs(q-qlinear(prob))
                max_residual=max(max_residual,residual);max_qsens=max(max_qsens,sensitivity)
                quantiles.append(dict(identity,model=model,probability=prob,role='primary' if prob in (.01,.99) else 'auxiliary' if prob in (.005,.995) else 'exploratory_only',AC_quantile=qa,model_quantile=q,signed_error=error,absolute_error=abs(error),CDF_inverse_residual=residual,linear_vs_PCHIP_quantile_difference=sensitivity))
            a=np.load(p.with_name(name+'_characteristic_terms.npz'))
            qmean,qvar=quadratic_moments(a)
            norm=float(np.trapezoid(pdf,x));norm_s=float(simpson(pdf,x=x))
            mean_bound=0.;sensitivity=0.;quaderr=0.;laplace_tail=0.;laplace_refinement=0.;pdf_mean=0.
            if kind=='V':
                anchor=acmean
                mean=anchor+float(simpson((x-anchor)*pdf,x=x))/norm_s
                trap=anchor+float(np.trapezoid((x-anchor)*pdf,x))/norm
                take=np.unique(np.r_[np.arange(0,len(x),2),len(x)-1]).astype(int)
                coarse=anchor+float(simpson((x[take]-anchor)*pdf[take],x=x[take]))/float(simpson(pdf[take],x=x[take]))
                sensitivity=max(abs(mean-trap),abs(mean-coarse))
                # E[V^2] is analytic. This is Var(|V|), not Var(|V|^2).
                pdf_mean=mean
                mean,quaderr,laplace_tail,laplace_refinement=amplitude_mean(a)
                sensitivity=max(sensitivity,abs(mean-pdf_mean))
                var=qmean-mean**2
                left_mass=max(float(F[0]),0.);right_mass=max(float(1-F[-1]),0.)
                missing_mean_bound=x[0]*left_mass+max(raw_q8(a),0.)**(1/16)*right_mass**(15/16)
                # Normalization correction is explicit; the strict tail bound is
                # for the finite conditional mixture, not the continuous strength law.
                unnormalized=float(simpson(x*pdf,x=x))
                mean_bound=missing_mean_bound+abs(mean-unnormalized)
                source='amplitude mean from saved quadratic CF Laplace integral; full PDF integral cross-check; E[V^2] analytic'
                rawmean=unnormalized
                var_bound=2*abs(mean)*mean_bound+mean_bound**2
            else:
                mean,var=qmean,qvar;source='conditional Gaussian quadratic moments with saved mixture/strength weights';rawmean=mean;var_bound=0.
            assert var>=0
            meanerr=mean-acmean;varerr=var-acvar
            moments.append(dict(identity,model=model,AC_mean=acmean,model_mean=mean,mean_signed_error=meanerr,mean_absolute_error=abs(meanerr),AC_variance=acvar,model_variance=var,variance_signed_error=varerr,variance_absolute_error=abs(varerr),variance_relative_error=varerr/acvar,variance_absolute_relative_error=abs(varerr)/acvar,moment_source=source,PDF_mass_trapezoid=norm,PDF_mass_Simpson=norm_s,unnormalized_PDF_mean=rawmean,observed_mean_integration_sensitivity=sensitivity,PDF_voltage_mean_tail_and_normalization_bound=mean_bound,PDF_voltage_variance_tail_and_normalization_bound=var_bound,normalized_PDF_amplitude_mean=pdf_mean,amplitude_Laplace_quadrature_estimated_error=quaderr,amplitude_Laplace_truncation_upper_bound=laplace_tail,amplitude_Laplace_tolerance_refinement_difference=laplace_refinement))
            checks.append(dict(identity,model=model,points=len(x),identical_coordinate_duplicates_collapsed=duplicate_count,minimum_CDF_increment=float(np.diff(F0).min()),monotonic_roundoff_correction=correction,CDF_lower=float(F0[0]),CDF_upper=float(F0[-1]),maximum_inverse_residual=max_residual,maximum_quantile_interpolation_sensitivity=max_qsens,conditional_weight_sum=float(a['weights'].sum())))
            hashes[str(p.relative_to(ROOT))]=digest(p)
    tables={'tail_CDF_RMSE_all97.csv':pd.DataFrame(tails),'extreme_quantiles_all97.csv':pd.DataFrame(quantiles),'global_moments_all97.csv':pd.DataFrame(moments),'evaluation_numerical_checks.csv':pd.DataFrame(checks)}
    for name,t in tables.items():t.to_csv(OUT/name,index=False,float_format='%.17g')
    pd.DataFrame(points).to_csv(OUT/'tail_CDF_evaluation_points.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    t=tables['tail_CDF_RMSE_all97.csv'];q=tables['extreme_quantiles_all97.csv'];m=tables['global_moments_all97.csv']
    focus=['V18','V33','P1','P16','P32','Q1','Q16','Q32','loss','P22','Q22']
    pivot=t.pivot(index='object',columns='model',values='right_CDF_RMSE')
    delta=pivot['Grouped_TMM']-pivot['AQGMM']
    extra=list(delta.nsmallest(2).index)+list(delta.nlargest(2).index)
    focus+=list(dict.fromkeys(v for v in extra if v not in focus))
    display=t.set_index('object').loc[focus].reset_index()
    body='# K=1 当前结果：逐对象定量比较\n\n只复用现存291组解析结果和独立3504组交流潮流；不重做源荷、交流潮流、参数估计，不扩算K=2、4。G=AQGMM，S=标准TMM，B=本文分组TMM。所有误差均为模型减交流潮流。以下支路编号保留原清单，起止节点见表。\n\n'
    rows=[]
    for name in focus:
        sub=t[t.object==name];base=sub.iloc[0]
        row=dict(object=name,branch=f"{base.from_bus}→{base.to_bus}" if base.kind in ('P','Q') else '-',unit=base.unit)
        for model,short in zip(MODELS,['G','S','B']):
            v=sub[sub.model==model].iloc[0];row[short+'_left_RMSE_pp']=v.left_CDF_RMSE_percentage_points;row[short+'_right_RMSE_pp']=v.right_CDF_RMSE_percentage_points
        rows.append(row)
    body+='## 左右尾CDF-RMSE（百分点）\n\n'+markdown(pd.DataFrame(rows))
    for prob in [.01,.99,.005,.995]:
        rows=[]
        for name in focus:
            sub=q[(q.object==name)&(q.probability==prob)];row=dict(object=name,unit=sub.iloc[0].unit,AC=sub.iloc[0].AC_quantile)
            for model,short in zip(MODELS,['G','S','B']):
                r=sub[sub.model==model].iloc[0];row[short]=r.model_quantile;row[short+'_signed_error']=r.signed_error;row[short+'_absolute_error']=r.absolute_error
            rows.append(row)
        body+=f'\n## q{prob:g}（物理单位）\n\n'+markdown(pd.DataFrame(rows))
    rows=[]
    for name in focus:
        sub=m[m.object==name]
        for r in sub.itertuples():rows.append(dict(object=name,model=r.model,unit=r.unit,AC_mean=r.AC_mean,model_mean=r.model_mean,mean_signed_error=r.mean_signed_error,AC_variance=r.AC_variance,model_variance=r.model_variance,variance_relative_error_percent=100*r.variance_relative_error))
    body+='\n## 幅值/功率的均值与方差\n\n方差单位为物理单位平方，AC方差ddof=0。电压幅值均值使用同一条件特征函数的Laplace积分，PDF积分作为核对；E[V²]解析计算后减幅值均值平方求方差。\n\n'+markdown(pd.DataFrame(rows))
    body+='\n## 适用范围与数值口径\n\n尾部各81个概率水平；经验CDF用右连续实际值，右尾没有除以接近1的CDF。模型分位反求自身CDF。q001/q999仅保存在完整CSV作探索性结果，不据约3—4个交流潮流尾端记录判主要优劣。插值与幅值积分诊断见evaluation_numerical_checks.csv及global_moments_all97.csv。有限混合尾部矩界不包含连续强度积分误差；加密差是数值敏感性，不是严格总误差上界。全部97对象逐模型CSV保留好、相当、差的数值，不用全网平均误差给出方法排名。\n'
    (OUT/'comparison_readable.md').write_text(body)
    summary=dict(status='complete',K=1,models=MODELS,quantities=97,reference_AC_scenarios=3504,tail_levels_per_side=81,quantiles=PROBS,variance_ddof=0,source_load_AC_and_parameters_recomputed=False,scope='Current date split, descriptive object-wise comparison only',plan='01_IDEA/plans/direct_pv8_K1_evaluation_20261026.md',input_hashes=dict(before,**hashes),python=platform.python_version(),seconds=time.perf_counter()-started,max_CDF_roundoff_correction=float(max(r['monotonic_roundoff_correction'] for r in checks)),max_inverse_residual=float(max(r['maximum_inverse_residual'] for r in checks)),max_tail_CDF_interpolation_sensitivity=float(max(max(r['left_max_interpolation_sensitivity'],r['right_max_interpolation_sensitivity']) for r in tails)),max_voltage_mean_observed_integration_sensitivity=float(max(r['observed_mean_integration_sensitivity'] for r in moments)),focus_objects=focus,outputs=list(tables)+['tail_CDF_evaluation_points.csv.gz','comparison_readable.md'])
    for p in files:assert digest(p)==before[str(p.relative_to(ROOT))]
    (OUT/'source_hashes.json').write_text(json.dumps(summary.pop('input_hashes'),indent=2)+'\n')
    summary['input_hash_manifest']='source_hashes.json'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_hashes'},indent=2))


if __name__=='__main__':main()
