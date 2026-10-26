"""Independent raw-voltage initialization and likelihood estimation for three families."""
from pathlib import Path
import ast,json,time,re
import numpy as np
from scipy.linalg import cholesky,solve_triangular
from scipy.special import gammaln,digamma,logsumexp
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'result/case33bw/direct_pv8_python_cloud_20261026'
OUT=SRC/'independent_block_t_20261026'
FLOOR=1e-7


def dump(path,value):path.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')


def core_functions():
 ns=dict(np=np,time=time,cholesky=cholesky,solve_triangular=solve_triangular,gammaln=gammaln,digamma=digamma,logsumexp=logsumexp,brentq=brentq,FLOOR=FLOOR)
 tree=ast.parse((ROOT/'tests/python/archived/cloud_numeric_core.py').read_text())
 for name in ['floor_matrix','px_scale','log_density','fit','admittance','rect_jacobian']:
  z=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name==name);exec(compile(ast.Module(body=[z],type_ignores=[]),'<reused numerical function>','exec'),ns)
 return ns
CORE=core_functions()


def geometry():
 OUT.mkdir(exist_ok=True);file=OUT/'geometry.npz'
 if file.exists():return np.load(file)
 ac=np.load(SRC/'ac_results.npz');split=json.load(open(SRC/'date_split.json'))['row_indices'];states=ac['fitting_states'];n=len(split['estimation']);center=states[:n].mean(0);_,singular,vt=np.linalg.svd(states[:n]-center,full_matrices=False);scale=singular/np.sqrt(n-1);basis=vt.T;x=(states[:n]-center)@basis/scale;xv=(states[n:]-center)@basis/scale
 text=re.sub(r'%[^\n]*','',(ROOT/'data/cases/case33bw_modify.m').read_text());tables={}
 for name in ['bus','branch']:
  q=re.search(r'mpc\.'+name+r'\s*=\s*\[(.*?)\];',text,re.S)[1];tables[name]=np.array([[float(v) for v in row.split()] for row in q.split(';') if row.strip()])
 base=float(re.search(r'mpc.baseMVA\s*=\s*([\d.]+)',text)[1]);bus=tables['bus'];branch=tables['branch'];bus[:,2:4]/=1000;branch[:,2:4]/=(bus[0,9]*1000)**2/(base*1e6);Y,_,_=CORE['admittance'](base,bus,branch)
 target=np.load(SRC/'inputs.npz')['target'][split['estimation']];anchor=int(np.argmin(((target-target.mean(0))**2).sum(1)));state=states[anchor];v=np.r_[1.,state[:32]+1j*state[32:]];Ji=np.linalg.inv(CORE['rect_jacobian'](v,Y));groups=json.load(open(SRC/'voltage_groups.json'));assert groups['leak_directions']==0
 names=groups['groups'][0];columns=np.array([int(z[1:])-2+(32 if z[0]=='Q' else 0) for z in names]);h=(basis.T/scale[:,None])@Ji[:,columns];u,s,_=np.linalg.svd(h,full_matrices=False);Q1=u[:,:np.sum(s>s[0]*1e-10)];Q2=np.linalg.svd(np.eye(64)-Q1@Q1.T)[0][:,:64-Q1.shape[1]];share=float(np.sum((scale[:,None]*Q1)**2)/np.sum(scale**2));assert Q1.shape==(64,8) and abs(share-groups['voltage_share_first_order'])<1e-9
 np.savez_compressed(file,x=x,xv=xv,center=center,basis=basis,scale=scale,Q1=Q1,Q2=Q2,base=base,bus=bus,branch=branch,Y=Y)
 dump(OUT/'geometry_audit.json',dict(rows=[len(x),len(xv)],block_dimensions=[8,56],projection_orthogonality=float(abs(np.c_[Q1,Q2].T@np.c_[Q1,Q2]-np.eye(64)).max()),physical_share=share,old_parameter_files_read=False,source='saved raw voltages, inputs, registered injection groups only'))
 return np.load(file)


def initialize_raw(x,k,partition,nu,geometry,family):
 n=len(x);labels=np.empty(n,int);labels[np.argsort(x[:,partition],kind='stable')]=np.minimum(k-1,np.arange(n)*k//n)
 if family!='Block_TMM':
  model=dict(weights=[],means=[],scatter=[],df=[])
  for j in range(k):
   z=x[labels==j];C=np.cov(z,rowvar=False,bias=True);scatter=C if np.isinf(nu) else C*(nu-2)/nu
   model['weights'].append(len(z)/n);model['means'].append(z.mean(0));model['scatter'].append(CORE['floor_matrix'](scatter,FLOOR if np.isinf(nu) else FLOOR*(nu-2)/nu)[0]);model['df'].append(nu)
 else:
  U=x@geometry['Q1'];W=x@geometry['Q2'];model={key:[] for key in ['weights','means','scatter','df','intercept','coefficient','residual']}
  for j in range(k):
   ix=labels==j;u,w=U[ix],W[ix];design=np.c_[np.ones(len(u)),u];coef=np.linalg.lstsq(design,w,rcond=None)[0];e=w-design@coef
   for key,val in dict(weights=ix.mean(),means=u.mean(0),scatter=CORE['floor_matrix'](np.cov(u,rowvar=False,bias=True)*(nu-2)/nu,FLOOR*(nu-2)/nu)[0],df=nu,intercept=coef[0],coefficient=coef[1:].T,residual=CORE['floor_matrix'](e.T@e/len(e))[0]).items():model[key].append(val)
 return {key:np.array(v) for key,v in model.items()}


def block_density(U,W,model):
 n,r=U.shape;dw=W.shape[1];logs=[];dist=[]
 for j,nu in enumerate(model['df']):
  L=cholesky(model['scatter'][j],lower=True);z=solve_triangular(L,(U-model['means'][j]).T,lower=True,check_finite=False).T;delta=(z*z).sum(1)
  t=-.5*(r*np.log(2*np.pi)+delta) if np.isinf(nu) else gammaln((nu+r)/2)-gammaln(nu/2)-r/2*np.log(nu*np.pi)-(nu+r)/2*np.log1p(delta/nu)
  LO=cholesky(model['residual'][j],lower=True);e=solve_triangular(LO,(W-model['intercept'][j]-U@model['coefficient'][j].T).T,lower=True,check_finite=False).T
  logs.append(np.log(model['weights'][j])+t-np.log(np.diag(L)).sum()-.5*(dw*np.log(2*np.pi)+(e*e).sum(1))-np.log(np.diag(LO)).sum());dist.append(delta)
 logs=np.array(logs).T;density=logsumexp(logs,axis=1);return density,np.exp(logs-density[:,None]),np.array(dist).T


def fit_block(U,W,initial,maximum=1200,tolerance=1e-6):
 start=time.perf_counter();m={k:v.copy() for k,v in initial.items()};n,r=U.shape;design=np.c_[np.ones(n),U];history=[];floors=[];density,resp,delta=block_density(U,W,m);history.append(float(density.mean()));status='maximum_iterations'
 for iteration in range(1,maximum+1):
  count=resp.sum(0)
  if count.min()<=r+W.shape[1]+2:status='collapsed_component';break
  new={k:v.copy() for k,v in m.items()};new['weights']=count/n;ff=[]
  for j,nu in enumerate(m['df']):
   tau=(nu+r)/(nu+delta[:,j]);w=resp[:,j]*tau;new['means'][j]=w@U/w.sum();e=U-new['means'][j];scatter=e.T@(e*w[:,None])/count[j]
   new['scatter'][j],fc=CORE['floor_matrix'](scatter,FLOOR*(nu-2)/nu)
   statistic=resp[:,j]@(digamma((nu+r)/2)-np.log((nu+delta[:,j])/2)-tau)/count[j];score=lambda v:np.log(v/2)-digamma(v/2)+1+statistic
   new['df'][j]=4.1 if score(4.1)<=0 else 1e5 if score(1e5)>=0 else brentq(score,4.1,1e5,xtol=1e-8)
   coeff=np.linalg.solve(design.T@(design*resp[:,j,None]),design.T@(W*resp[:,j,None]));new['intercept'][j]=coeff[0];new['coefficient'][j]=coeff[1:].T;e=W-design@coeff;new['residual'][j],fr=CORE['floor_matrix'](e.T@(e*resp[:,j,None])/count[j]);ff.append([fc,fr])
  density,nresp,ndelta=block_density(U,W,new);value=float(density.mean());gain=value-history[-1]
  if gain < -1e-8:status='rejected_decreasing_update';break
  m,resp,delta=new,nresp,ndelta;history.append(value);floors.append(ff)
  if abs(gain)<tolerance*max(1.,abs(history[-2])):status='converged';break
 return m,dict(status=status,iterations=iteration,history=history,train_ll=history[-1],effective_counts=resp.sum(0),df=m['df'],weights=m['weights'],floor_history=floors,floor_active_max=int(np.max(floors)) if floors else 0,seconds=time.perf_counter()-start)


def run():
 g=geometry();x,xv=g['x'],g['xv'];folder=OUT/'fits';folder.mkdir(exist_ok=True);all_selected=[]
 for k in [1,2,4]:
  for family in ['AQGMM','Standard_TMM','Block_TMM']:
   candidates=[]
   for partition in range(1 if k==1 else 2):
    for nu in ([np.inf] if family=='AQGMM' else [12.,40.,200.]):
     stem=folder/f'{family}_K{k}_PC{partition+1}_nu{nu:g}';auditfile=stem.with_suffix('.json')
     if auditfile.exists():audit=json.load(open(auditfile));model=dict(np.load(stem.with_suffix('.npz')))
     else:
      print('FIT_START',family,k,partition,nu,flush=True);initial=initialize_raw(x,k,partition,nu,g,family)
      if family=='Block_TMM':model,audit=fit_block(x@g['Q1'],x@g['Q2'],initial);selection=float(block_density(xv@g['Q1'],xv@g['Q2'],model)[0].mean())
      else:model,audit=CORE['fit'](x,initial,maximum=1200,tolerance=1e-6,actual_covariance_floor=True,relative_tolerance=True);selection=float(CORE['log_density'](xv,model)[0].mean())
      audit.update(family=family,K=k,partition=partition+1,initial_df=nu,selection_ll=selection,parameter_source='raw voltage only; separate initialization; no other model results',old_parameters_read=False)
      np.savez_compressed(stem.with_suffix('.npz'),**model);dump(auditfile,audit);print('FIT_END',family,k,audit['status'],audit['iterations'],audit['df'],audit['seconds'],flush=True)
     candidates.append((audit,model,str(stem.relative_to(OUT))))
   good=[z for z in candidates if z[0]['status']=='converged']
   if not good:raise RuntimeError(f'No converged model: {family} K={k}; audits preserved')
   audit,model,source=max(good,key=lambda z:z[0]['selection_ll']);np.savez_compressed(OUT/f'{family}_K{k}.npz',**model);all_selected.append(dict(family=family,K=k,source=source,**{key:audit[key] for key in ['status','selection_ll','train_ll','df','weights','iterations','seconds']}));dump(OUT/'selected_models.json',all_selected)
   print('SELECTED',family,k,audit['selection_ll'],flush=True)
 dump(OUT/'fit_complete.json',dict(status='complete',selected_models=9,total_attempts=len(list(folder.glob('*.json'))),original_model_files_read=False))

if __name__=='__main__':run()
