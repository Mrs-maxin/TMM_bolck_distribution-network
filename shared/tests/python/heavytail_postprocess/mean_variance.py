"""Full-law electrical means/variances; no finite voltage draws for model moments."""
import recompute as r
import numpy as np
import pandas as pd
from scipy import stats,integrate
from numpy.polynomial.hermite import hermgauss
import json,time,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
OUT=r.ROOT/'tail_distribution/computation_records'
POINTS=[.9,.99,.999,.9999,.99999]

def gaussian_quadratic(forms,mean,cov):
    A,b,c=(forms[key] for key in ['A','b','c'])
    am=np.einsum('qij,j->qi',A,mean);linear=2*am+b
    ac=np.einsum('qij,jk->qik',A,cov,optimize=True)
    m=am@mean+b@mean+c+np.trace(ac,axis1=1,axis2=2)
    v=2*np.einsum('qij,qji->q',ac,ac)+np.einsum('qi,ij,qj->q',linear,cov,linear,optimize=True)
    return m,v

class VoltageMagnitude:
    def __init__(self,coords,reference,nodes):
        self.v0=coords['center'][:32]+1j*coords['center'][32:]
        self.T=(coords['basis'][:32]+1j*coords['basis'][32:])*coords['scale']
        self.reference=self.v0+self.T@reference;self.a=abs(self.reference)
        self.unit=np.conj(self.reference)/self.a
        self.R=(self.unit[:,None]*self.T).real;self.I=(self.unit[:,None]*self.T).imag
        z,w=hermgauss(nodes);self.z=np.sqrt(2)*z;self.w=np.outer(w,w)/np.pi
    def terms(self,mean,cov):
        mu=self.v0+self.T@mean;aligned=mu*self.unit
        vr=np.einsum('ni,ij,nj->n',self.R,cov,self.R,optimize=True)
        vi=np.einsum('ni,ij,nj->n',self.I,cov,self.I,optimize=True)
        cross=np.einsum('ni,ij,nj->n',self.R,cov,self.I,optimize=True)
        l00=np.sqrt(np.maximum(vr,0));l10=np.divide(cross,l00,out=np.zeros_like(cross),where=l00>1e-150)
        l11=np.sqrt(np.maximum(vi-l10*l10,0))
        real=aligned.real[:,None,None]+l00[:,None,None]*self.z[None,:,None]
        imag=aligned.imag[:,None,None]+l10[:,None,None]*self.z[None,:,None]+l11[:,None,None]*self.z[None,None,:]
        magnitude=np.hypot(real,imag);denominator=magnitude+real
        correction=np.divide(imag**2,denominator,out=(magnitude-real).copy(),where=denominator>1e-14)
        corr=np.einsum('nij,ij->n',correction,self.w)
        projection=aligned.real-self.a
        distance=vr+vi+abs(mu-self.reference)**2
        return projection,corr,distance

def quad_checked(func,eps):
    result,error,info=integrate.quad_vec(func,0.,1.,epsabs=eps,epsrel=eps,norm='max',points=POINTS,limit=2000,full_output=True)
    assert info.success,(info.message,error,info.neval)
    return result,float(error),int(info.neval)

def component_Gaussian_or_t(model,index,forms,coords):
    mu=model['means'][index];scatter=model['scatter'][index];nu=model['df'][index]
    cov=scatter*(nu/(nu-2) if np.isfinite(nu) else 1.)
    mean,var=gaussian_quadratic(forms,mu,cov)
    if np.isfinite(nu):
        A=forms['A'];ac=np.einsum('qij,jk->qik',A,cov,optimize=True);trace=np.trace(ac,axis1=1,axis2=2)
        factor=(nu-2)/(nu-4)
        var+=(factor-1)*(2*np.einsum('qij,qji->q',ac,ac)+trace**2)
    results=[];checks=[]
    for gh in [20,32]:
        vm=VoltageMagnitude(coords,mu,gh)
        if not np.isfinite(nu):projection,corr,distance=vm.terms(mu,cov);err=0.;neval=1
        else:
            scale=np.maximum(vm.terms(mu,scatter)[1],1e-20)
            def f(u):
                s=nu/stats.chi2.isf(min(u,np.nextafter(1.,0.)),nu)
                return vm.terms(mu,scatter*s)[1]/scale
            corr,err,neval=quad_checked(f,2e-8 if gh==20 else 3e-10);corr*=scale
            projection=np.zeros(32)
            distance=np.einsum('ni,ij,nj->n',vm.R,cov,vm.R,optimize=True)+np.einsum('ni,ij,nj->n',vm.I,cov,vm.I,optimize=True)
        mv=vm.a+projection+corr;vv=distance-2*vm.a*corr-(projection+corr)**2
        assert vv.min()>-1e-11,('voltage_variance',vv.min())
        results.append((mv,np.maximum(vv,0)))
        checks.append(dict(GH=gh,integration_scaled_error=err,evaluations=neval))
    delta_mean=float(abs(results[1][0]-results[0][0]).max());delta_var=float(abs(results[1][1]-results[0][1]).max())
    mean[:32]=results[-1][0];var[:32]=results[-1][1]
    return mean,var,dict(voltage_GH_refinement_mean_max_pu=delta_mean,voltage_GH_refinement_variance_max_pu2=delta_var,integrations=checks)

def component_grouped(c,model,forms,coords):
    Q1,Q2=model['Q1'],model['Q2'];M=Q1+Q2@c['coefficient'];offset=Q2@c['intercept'];residual=Q2@c['residual']@Q2.T
    ref=offset+M@c['mu'];anchor,_=gaussian_quadratic(forms,ref,M@c['scatter']@M.T+residual)
    final=[];checks=[]
    for gh in [20,32]:
        vm=VoltageMagnitude(coords,ref,gh)
        def f(u):
            strengths=np.array([1. if t.gauss else t.nu/stats.chi2.isf(min(u*t.pmax,np.nextafter(1.,0.)),t.nu)/t.mean for t in c['trunc']])
            s=strengths[c['gid']];conditional=c['mu']+c['shift']*(s-1)
            Su=np.sqrt(s)[:,None]*c['scatter']*np.sqrt(s)[None,:]
            mean=offset+M@conditional;cov=M@Su@M.T+residual
            qm,qv=gaussian_quadratic(forms,mean,cov);delta=qm-anchor
            projection,corr,distance=vm.terms(mean,cov)
            return np.r_[delta,qv+delta**2,projection,corr,distance]
        probes=np.array([f(x) for x in [.25,.75,.95,.995,.9999]])
        scale=np.maximum(abs(probes).max(0),1e-22)
        out,error,neval=quad_checked(lambda u:f(u)/scale,2e-8 if gh==20 else 3e-10);out*=scale
        qm=anchor+out[:97];qv=out[97:194]-out[:97]**2
        projection=out[194:226];corr=out[226:258];distance=out[258:290]
        qm[:32]=vm.a+projection+corr;qv[:32]=distance-2*vm.a*corr-(projection+corr)**2
        assert qv.min()>-1e-11,('group_variance',qv.min())
        final.append((qm,np.maximum(qv,0)))
        checks.append(dict(GH=gh,scaled_error=error,evaluations=neval))
    return *final[-1],dict(mean_refinement_abs_max=float(abs(final[1][0]-final[0][0]).max()),
        variance_refinement_abs_max=float(abs(final[1][1]-final[0][1]).max()),integrations=checks)

def model_job(scene,k,name):
    start=time.perf_counter();path=OUT/f'{scene}_K{k}_{name}_moments.json'
    if path.exists():return scene,k,name,'cached'
    ns=r.numerical_functions();d=r.OLD/scene;coords=dict(np.load(d/'voltage_coordinates.npz'));forms=ns['quadratic_forms'](coords)
    model=r.read_models(d/f'models_K{k}.pkl')[name];mc=json.loads((r.ROOT/'tail_distribution'/f'{scene}_MC.json').read_text())['moments']
    weights=[];means=[];variances=[];checks=[]
    if name=='grouped_t_shift':
        total=sum(c['weight'] for c in model['shifted'])
        for ci,c in enumerate(model['shifted']):
            m,v,check=component_grouped(c,model,forms,coords);weights.append(c['weight']/total);means.append(m);variances.append(v);checks.append(dict(component=ci,**check))
    else:
        for ci,w in enumerate(model['weights']):
            m,v,check=component_Gaussian_or_t(model,ci,forms,coords);weights.append(w);means.append(m);variances.append(v);checks.append(dict(component=ci,**check))
    weights=np.array(weights);means=np.array(means);variances=np.array(variances)
    mean=weights@means;variance=weights@(variances+(means-mean)**2);rows=[]
    for j in range(97):
        obj,kind,unit,scale=r.object_info(j);ref=mc[j]
        rows.append(dict(情形=r.CN[scene],方法=r.CN[name],K=k,电气量=obj,类型=kind,
            MC均值=ref['mean']*scale,方法均值=mean[j]*scale,均值误差=(mean[j]-ref['mean'])*scale,
            均值绝对误差=abs(mean[j]-ref['mean'])*scale,均值误差除以MC标准差=(mean[j]-ref['mean'])/np.sqrt(ref['variance']),
            MC方差=ref['variance']*scale**2,方法方差=variance[j]*scale**2,方差误差=(variance[j]-ref['variance'])*scale**2,
            方差相对误差=variance[j]/ref['variance']-1,MC标准差=np.sqrt(ref['variance'])*scale,方法标准差=np.sqrt(variance[j])*scale,
            单位=unit,方差单位=unit+'²',模型状态=r.status(scene,k,name),
            矩计算='二次型精确矩；电压幅值为条件积分' if name!='grouped_t_shift' else '已保存截断强度分布的自适应积分'))
    r.write_json(path,dict(rows=rows,checks=checks,seconds=time.perf_counter()-start))
    return scene,k,name,time.perf_counter()-start

def collect():
    rows=[]
    for s in r.SCENES:
        for k in [1,2,4,6]:
            for name in r.MODELS:rows.extend(json.loads((OUT/f'{s}_K{k}_{name}_moments.json').read_text())['rows'])
    frame=pd.DataFrame(rows);assert len(frame)==4656
    frame['方差比_方法除以MC']=frame['方法方差']/frame['MC方差']
    frame.to_csv(r.OUT/'full_mean_variance_all_quantities.csv',index=False,encoding='utf-8-sig')
    idx=['情形','K','电气量','类型','单位','方差单位','MC均值','MC方差','MC标准差']
    wide=frame.pivot(index=idx,columns='方法',values=['方法均值','均值误差','方法方差','方差相对误差','方差比_方法除以MC','方法标准差']).reset_index()
    wide.columns=['_'.join(str(v) for v in col if v!='') if isinstance(col,tuple) else col for col in wide.columns]
    wide.to_csv(r.OUT/'full_mean_variance_method_compare.csv',index=False,encoding='utf-8-sig')
    selected=pd.read_csv(r.ROOT/'tail_distribution/focus_quantities_mc_selection.csv')[['情形','电气量']]
    wide.merge(selected,on=['情形','电气量'],validate='m:1').to_csv(r.ROOT/'tail_distribution/focus_quantities_full_mean_variance_method_compare.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);a=p.parse_args();OUT.mkdir(exist_ok=True)
    tasks=[(s,k,name) for s in r.SCENES for k in [1,2,4,6] for name in r.MODELS if not (OUT/f'{s}_K{k}_{name}_moments.json').exists()]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(model_job,*task) for task in tasks]):print(future.result(),flush=True)
    collect()
