"""Compare an equivalent statewise inversion to the existing mixture inversion."""
import numpy as np,pickle,time
import pandas as pd
from numpy.polynomial.legendre import leggauss
import run_direct_heavytail_case33 as r

def statewise(tm,thresholds):
    thresholds=np.asarray(thresholds);out=np.zeros(len(thresholds));endmax=0.
    for w,m,lam,beta in tm:
        if w<1e-14:continue
        sd=np.sqrt(2*np.sum(lam*lam)+np.sum(beta*beta));mean=m+lam.sum()
        if sd<1e-20:
            out+=w*(thresholds>=mean);continue
        lam=lam/sd;beta=beta/sd;y=(thresholds-mean)/sd
        lo,hi=12.,12.
        def cf(t):
            z=1-2j*np.outer(t,lam)
            return np.exp(-1j*t*lam.sum()-.5*np.log(z).sum(1)-(np.outer(t*t,beta*beta)/(2*z)).sum(1))
        while abs(cf(np.array([hi]))[0])>1e-12 and hi<4096:hi*=2
        # Match frequency resolution to the largest standardized threshold.
        nodes,ww=leggauss(32);pieces=int(np.ceil(hi*max(2.,np.max(abs(y))/32)))
        edges=np.linspace(0,hi,pieces+1)
        t=((edges[1:]-edges[:-1])[:,None]*(nodes+1)/2+edges[:-1,None]).ravel()
        wt=((edges[1:]-edges[:-1])[:,None]*ww/2).ravel();phi=cf(t)
        value=np.clip(.5-np.imag(np.exp(-1j*np.outer(y,t))*phi)/t@wt/np.pi,0,1)
        out+=w*value;endmax=max(endmax,float(w*abs(phi[-1])))
    return out,endmax

if __name__=='__main__':
    d=r.OUT/'pareto';data=dict(np.load(d/'voltage_coordinates.npz'));forms=r.an.quadratic_forms(data)
    with (d/'models_K4.pkl').open('rb') as f:models=pickle.load(f)
    rows=[]
    for name,kind in [('AQGMM','GMM'),('standard_TMM','TMM')]:
        terms=r.an.terms(models[name],kind,forms);a=np.load(d/f'CDF_{name}_K4.npz')
        for j in [16,31,32,48,55,80,96]:
            x=a['thresholds'][j];t=time.perf_counter();F,end=statewise(terms[j],x*x if j<32 else x)
            err=float(abs(F-a['CDF'][j]).max());rows.append(dict(model=name,object=r.object_name(j),max_difference=err,seconds=time.perf_counter()-t))
            print(rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(r.OUT/'statewise_inversion_check.csv',index=False)
