"""Analytic distributions and object-wise independent/estimation-date evaluation."""
from pathlib import Path
import os,json,time,traceback,concurrent.futures
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from independent_block_t_fit import OUT,SRC,geometry,dump
from independent_block_t_numeric import specs,projected_parts,exact_moments,terms,decompose
from direct_pv8_inversion import inverse_pdf,moments as numerical_moments
from evaluate_direct_pv8_K1 import amplitude_mean
from independent_tail_w1 import tail_w1
G=None;SPECS=None;REF=None


def initialize_worker():
 global G,SPECS,REF
 G=geometry();SPECS=specs(G);REF=np.load(SRC/'ac_results.npz')['reference']


class Quantile:
 def __init__(self,x,cdf):
  self.x=x;self.cdf=np.maximum.accumulate(cdf);self.F=PchipInterpolator(x,self.cdf,extrapolate=False);self.maximum_residual=0.
 def __call__(self,p):
  p=np.asarray(p);shape=p.shape;pp=p.ravel();j=np.searchsorted(self.cdf,pp);assert j.min()>0 and j.max()<len(self.x)
  lo=self.x[j-1].copy();hi=self.x[j].copy();q=np.interp(pp,self.cdf,self.x)
  for _ in range(30):
   e=self.F(q)-pp;done=abs(e)<2e-12
   if done.all():break
   lo=np.where((e<0)&~done,q,lo);hi=np.where((e>0)&~done,q,hi);der=self.F.derivative()(q);new=q-e/np.maximum(der,1e-300);new=np.where((new<=lo)|(new>=hi)|~np.isfinite(new),(lo+hi)/2,new);q=np.where(done,q,new)
  residual=float(abs(self.F(q)-pp).max());self.maximum_residual=max(self.maximum_residual,residual);assert residual<2e-9,residual
  return q.reshape(shape)


def work(task):
 family,k,j=task;spec=SPECS[j];name=spec['name'];folder=OUT/'distributions'/f'K{k}'/family;folder.mkdir(parents=True,exist_ok=True);infofile=folder/f'{name}.json'
 if infofile.exists():
  old=json.load(open(infofile))
  if old.get('status')=='complete':return old
 start=time.perf_counter();model=dict(np.load(OUT/f'{family}_K{k}.npz'));parts=list(projected_parts(model,family,G,spec));mean,var=exact_moments(parts,spec['A'],spec['b'],spec['c'])
 physical=lambda xx:np.einsum('ni,ij,nj->n',xx@spec['R'].T+spec['o'],spec['A'],xx@spec['R'].T+spec['o'])+(xx@spec['R'].T+spec['o'])@spec['b']+spec['c']
 estq=physical(G['x']);est=np.sqrt(np.maximum(estq,0)) if spec['kind']=='V' else estq;ref=REF[:,j];levels=np.r_[np.linspace(.01,.05,81),np.linspace(.95,.99,81)];refqs=np.quantile(ref,levels);estqs=np.quantile(est,levels)
 # Strength convergence is checked at AC and estimation-date thresholds plus a central physical grid.
 coarse=terms(parts,spec,1);muq,sdq,_=numerical_moments(coarse);coreq=np.r_[np.linspace(ref.min(),ref.max(),81),refqs,estqs]
 probe=coreq**2 if spec['kind']=='V' else coreq
 pc,fc,_=inverse_pdf(coarse,probe);history=[];level=2
 while True:
  tm=terms(parts,spec,level);pf,ff,_=inverse_pdf(tm,probe);changeF=float(abs(ff-fc).max());changeP=float(abs(pf-pc).max()*sdq);history.append(dict(level=level,conditional_states=len(tm),maximum_CDF_change=changeF,scaled_PDF_change=changeP))
  if changeF<2e-5 and changeP<2e-4:break
  if level>=16:raise RuntimeError((task,'strength quadrature unresolved',history))
  pc,fc=pf,ff;level*=2
 muq,sdq,_=numerical_moments(tm);qlo=min(muq-12*sdq,float(ref.min())**2 if spec['kind']=='V' else float(ref.min()));qhi=max(muq+12*sdq,float(ref.max())**2 if spec['kind']=='V' else float(ref.max()))
 if spec['kind']=='V':lo=np.sqrt(max(qlo,1e-12));hi=np.sqrt(qhi)
 else:lo,hi=qlo,qhi
 grid=np.unique(np.r_[np.linspace(lo,hi,2401),np.linspace(ref.min(),ref.max(),1601),refqs,estqs]);attempts=[]
 for extend in range(7):
  qgrid=grid**2 if spec['kind']=='V' else grid;pdf,cdf,intinfo=inverse_pdf(tm,qgrid)
  if spec['kind']=='V':pdf*=2*grid
  attempts.append(dict(lower=float(grid[0]),upper=float(grid[-1]),CDF_lower=float(cdf[0]),CDF_upper=float(cdf[-1])))
  if cdf[0]<1e-6 and cdf[-1]>1-1e-6:break
  span=grid[-1]-grid[0];lower=grid[0]-span;upper=grid[-1]+span
  if spec['kind']=='V':lower=max(lower,1e-8)
  grid=np.unique(np.r_[grid,np.linspace(lower,grid[0],1001),np.linspace(grid[-1],upper,1001)])
 else:raise RuntimeError((task,'boundary unresolved',attempts))
 mass=float(np.trapezoid(pdf,grid));masserror=abs(mass-float(cdf[-1]-cdf[0]));assert masserror<3e-4,(task,masserror)
 correction=float(abs(np.maximum.accumulate(cdf)-cdf).max());assert correction<2e-7,(task,'CDF oscillation',correction)
 inverse=Quantile(grid,cdf);quantilevalues={p:float(inverse(np.array([p]))[0]) for p in [.005,.01,.99,.995]}
 coarsegrid=np.unique(np.r_[np.arange(0,len(grid),2),len(grid)-1]).astype(int);coarse_inv=Quantile(grid[coarsegrid],cdf[coarsegrid]);ps=np.r_[np.linspace(.01,.05,81),np.linspace(.95,.99,81)];qsensitivity=float(abs(inverse(ps)-coarse_inv(ps)).max())
 resultrows=[];qrows=[];pointrows=[]
 amplitude_diag={}
 if spec['kind']=='V':
  a=dict(weights=np.array([z[0] for z in tm]),m0=np.array([z[1] for z in tm]),lam=np.array([z[2] for z in tm]),beta=np.array([z[3] for z in tm]));amp,e,bound,refine=amplitude_mean(a);coarse_a={key:np.array([z[i] for z in coarse]) for i,key in enumerate(['weights','m0','lam','beta'])};oldamp,*_=amplitude_mean(coarse_a);amplitude_diag=dict(quadrature_error=e,Laplace_tail_bound=bound,tolerance_change=refine,strength_refinement_change=abs(amp-oldamp));modelmean=amp;modelvar=mean-amp**2
 else:modelmean,modelvar=mean,var
 for label,values,thresholds in [('independent_AC',ref,refqs),('estimation_dates',est,estqs)]:
  Fs=np.interp(thresholds,grid,cdf);actual=np.searchsorted(np.sort(values),thresholds,side='right')/len(values);left=float(np.sqrt(np.mean((Fs[:81]-actual[:81])**2)));right=float(np.sqrt(np.mean((Fs[81:]-actual[81:])**2)))
  wl,wl_info=tail_w1(inverse,values,.01,.05,inverse.F);wr,wr_info=tail_w1(inverse,values,.95,.99,inverse.F)
  rm=float(values.mean());rv=float(values.var());me=modelmean-rm;ve=modelvar-rv
  resultrows.append(dict(object=name,kind=spec['kind'],from_bus=spec['from_bus'],to_bus=spec['to_bus'],K=k,model=family,reference=label,left_CDF_RMSE=left,right_CDF_RMSE=right,left_CDF_RMSE_pp=100*left,right_CDF_RMSE_pp=100*right,left_trimmed_W1=wl,right_trimmed_W1=wr,left_W1_unnormalized=wl_info['unnormalized_integral'],right_W1_unnormalized=wr_info['unnormalized_integral'],reference_mean=rm,model_mean=modelmean,mean_signed_error=me,mean_absolute_error=abs(me),mean_relative_error=me/rm if abs(rm)>1e-14 else None,reference_variance=rv,model_variance=modelvar,variance_signed_error=ve,variance_absolute_error=abs(ve),variance_relative_error=ve/rv,variance_absolute_relative_error=abs(ve)/rv,W1_diagnostics=dict(left=wl_info,right=wr_info)))
  for prob,value in quantilevalues.items():
   qref=float(np.quantile(values,prob,method='linear'));qrows.append(dict(object=name,K=k,model=family,reference=label,probability=prob,reference_quantile=qref,model_quantile=value,signed_error=value-qref,absolute_error=abs(value-qref)))
  pointrows.append(dict(reference=label,nominal_probabilities=levels,thresholds=thresholds,actual_empirical_CDF=actual,model_CDF=Fs))
 pd.DataFrame(dict(value=grid,pdf=pdf,cdf=cdf)).to_csv(folder/f'{name}.csv',index=False)
 np.savez_compressed(folder/f'{name}_CF.npz',weights=np.array([z[0] for z in tm]),m0=np.array([z[1] for z in tm]),lam=np.array([z[2] for z in tm]),beta=np.array([z[3] for z in tm]))
 numerical_mean,numerical_sd,_=numerical_moments(tm)
 info=dict(status='complete',family=family,K=k,object=name,unit='p.u.' if spec['kind']=='V' else 'MVAr' if spec['kind']=='Q' else 'MW',seconds=time.perf_counter()-start,strength_history=history,strength_level=level,inversion=intinfo,boundary_history=attempts,PDF_mass=mass,PDF_CDF_mass_difference=masserror,CDF_roundoff_monotonic_correction=correction,model_quantile_grid_sensitivity=qsensitivity,maximum_CDF_inverse_residual=inverse.maximum_residual,exact_quadratic_mean=mean,exact_quadratic_variance=var,finite_strength_quadrature_mean=numerical_mean,finite_strength_quadrature_variance=numerical_sd**2,quadrature_variance_relative_difference=(numerical_sd**2-var)/var,amplitude_diagnostics=amplitude_diag,metrics=resultrows,quantiles=qrows,tail_points=pointrows)
 if name in ['Q17','loss']:
  info['model_variance_decomposition']=decompose(parts,spec);delta=G['x']@spec['R'].T;ell=2*spec['A']@spec['o']+spec['b'];L=delta@ell;H=np.einsum('ni,ij,nj->n',delta,spec['A'],delta);info['estimation_variance_decomposition']=dict(VarL=float(L.var()),VarH=float(H.var()),twice_covariance=float(2*np.mean((L-L.mean())*(H-H.mean()))))
 if name=='Q17':
  qi=SPECS[97];info['implicit_Q18_mean_variance']=exact_moments(list(projected_parts(model,family,G,qi)),qi['A'],qi['b'],qi['c'])
 dump(infofile,info);return info


def run():
 initialize_worker();OUT.mkdir(exist_ok=True);tasks=[(family,k,j) for k in [1,2,4] for family in ['AQGMM','Standard_TMM','Block_TMM'] for j in range(97)];completed=[];start=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,initializer=initialize_worker) as pool:
  pending={pool.submit(work,t):t for t in tasks}
  for future in concurrent.futures.as_completed(pending):
   task=pending[future]
   try:info=future.result()
   except Exception:
    dump(OUT/'distribution_failure.json',dict(task=task,traceback=traceback.format_exc()));raise
   completed.append(info);print('DISTRIBUTION',len(completed),'/873',task,round(info['seconds'],2),flush=True);dump(OUT/'progress.json',dict(status='running',completed=len(completed),total=873,seconds=time.perf_counter()-start,last=task))
 allrows=[];allq=[]
 for info in completed:
  for row in info['metrics']:
   row=dict(row);row.pop('W1_diagnostics');allrows.append(row)
  allq.extend(info['quantiles'])
 pd.DataFrame(allrows).to_csv(OUT/'per_object_errors_and_moments.csv',index=False);pd.DataFrame(allq).to_csv(OUT/'extreme_quantiles.csv',index=False)
 dump(OUT/'calculation_complete.json',dict(status='complete',models=9,quantities=97,distributions=len(completed),main_reference='3504 independent AC',auxiliary_reference='10512 estimation-date existing voltages',seconds=time.perf_counter()-start,original_scenarios_and_AC_recomputed=False))

if __name__=='__main__':run()
