"""Prespecified complete cloud-passage curves, standard AC, joint voltage mixtures."""
from pathlib import Path
import os,sys,time,json,traceback,ast,shutil,zipfile
import numpy as np
import pandas as pd
from scipy.stats import truncnorm,skew,kurtosis
from cloud_numeric_core import ROOT,QUANTILES,sha,write_json,admittance,polar_ac,evaluate_outputs,site_regions,initialize,fit,log_density,propagate,environment
from run_cloud_study import save_fit,distribution_metrics
from run_typhoon_comparison_K136 import network,voltage,outputs
O=ROOT/'result/case33bw/pv_weather_comparison_20261019/cloud_passage_power_curves_20261019';O.mkdir(exist_ok=True)
SCENES=['weak_local','regional_passage'];KS=[1,3,6,5,12];N=65536;SPLITS={'training':64,'validation':32,'reference':128};OFF={'training':11,'validation':104729,'reference':209759}
BASE,BUS,BRANCH=network();Y,_,_=admittance(BASE,BUS,BRANCH);SITES=np.array([18,17,16,15,14,13,33,22]);CAP=1.25*BUS[:,2].clip(0).sum()/8;_,REGIONS,_=site_regions(BUS,BRANCH);REGIONS=REGIONS[1:]
OBJECTS=[dict(output='V',node=j,branch=0,unit='p.u.',priority=j in [12,13,18]) for j in range(2,34)]+[dict(output=q,node=0,branch=j,unit='MW' if q=='P' else 'MVAr',priority=j in [12,6]) for q in ['P','Q'] for j in range(1,33)]+[dict(output='loss',node=0,branch=0,unit='MW',priority=True)]
PRIMARY=np.array([j for j,o in enumerate(OBJECTS) if o['priority']]);START=time.perf_counter();CPU=time.process_time()
def csv(p,rows):
 p=Path(p);df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);b=df.to_csv(index=False).encode();t=p.with_name(p.name+'.writing')
 with t.open('wb',buffering=0) as f:
  for i in range(0,len(b),1048576):
   v=memoryview(b)[i:i+1048576]
   while len(v):n=f.write(v);assert n>0;v=v[n:]
  os.fsync(f.fileno())
 os.replace(t,p);assert len(pd.read_csv(p))==len(df)
def save_npz(p,**a):
 p=Path(p);t=p.with_name(p.stem+'.tmp.npz');np.savez_compressed(t,**a)
 with zipfile.ZipFile(t) as z:assert z.testzip() is None
 os.replace(t,p)
def event(stage,**kw):
 r=dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),PID=os.getpid(),stage=stage,candidate_records=len(list(O.glob('*/candidates/K*.json'))),selected_models=sum(len(pd.read_csv(p)) for p in O.glob('*/selected_models.csv')),completed_propagations=len(list(O.glob('*/propagation/*_complete.json'))),total_models=20,total_propagations=80,**kw);write_json(O/'progress.json',r)
 with (O/'progress_history.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
 print(json.dumps(r),flush=True)
def protocol():
 p=dict(version='cloud-passage-power-curves-v1',execution='OpenAI_cloud_only',identity='synthetic engineering steady-state snapshots from full cloud passages; zero measured observations',source_revision='0d5c999db4cbee2ea7055844d31b729d9694d747',historical_snapshot_inputs_reused=False,
 motivation=dict(DOI='10.1016/j.solener.2017.07.035',role='depth, duration, ramp smoothness and lag motivate engineering curves; ranges below are engineering assumptions, not source TMM findings'),
 network=dict(case='case33bw',baseMVA=BASE,slack='bus1=1+0j',case_SHA256=sha(ROOT/'data/cases/case33bw_modify.m'),topology_unchanged=True),
 PV=dict(buses=SITES.tolist(),total_AC_capacity_MW=8*CAP,per_site_P_limit_MW=CAP,per_site_S_MVA=1.05*CAP,Q_MVAr=0,formula='Ci*clip(hC(t)*rPV(T)*(1-depth_i*half_cosine_shading_i),0,1)',hC='.85-.025*(solar_hour-12.5)^2',temperature='constant within process: truncatedNormal(28,1,[25,32]); rPV=max(0,1-.0037*(T-25)), engineering approximation, not cell-temperature measurement',cloud_edge_enhancement=False,extra_Gaussian_PV_or_A=False),
 process=dict(duration_minutes=12,step_seconds=15,points=49,solar_start_uniform=[11,13.8],center_minutes_uniform=[5,7],plateau_minutes_uniform=[2,4],down_up_minutes_uniform=[.5,1.5],six_site_lag_minutes_uniform=[-.5,.5],ramp='half cosine, continuous first derivative at joins; pre/down/plateau/up/post fully retained',time_identity='constructed approximate local solar hour, not observed UTC'),
 scenes=dict(weak_local=dict(depth_uniform=[.05,.15],coverage='rotating two adjacent positions among [18,17,16,15,14,13] by process ID; all other sites unaffected'),regional_passage=dict(shared_depth_uniform=[.2,.8],site_offset_uniform=[-.05,.05],final_depth_bounds=[.2,.8],six_stations_covered=True,bus33='half common depth, lag+0.5min',bus22='quarter common depth, lag+1min',mapping='fixed equivalent weather grouping; bus numbers do not imply geographical coordinates')),
 load=dict(base_multiplier=.8,general_P_share=.4,general_Q_share=.4,cooling_P_share=.6,P_std=[.03,.08,.15],Q_std=[.04,.10,.20],PQ_correlation=[.35,.35,.25],P_bounds=[.5,1.6],Q_bounds=[.4,1.8],cooling_independent=True,pf_uniform=[.93,.99],Qcool='Pcool*tan(acos(pf))',weather_multiplier=False,temporal_extension='stationary Gaussian AR1 innovations, rho=exp(-15/60), same snapshot marginals as old C; one-minute correlation engineering assumption; pf independent per node/time',paired='same load/time/T/base passage shape in scene pair; no stream sharing across splits'),
 samples=dict(processes=SPLITS,points_per_process=49,per_scene={s:n*49 for s,n in SPLITS.items()},AC_total=21952,measured_records=0,split='whole independent process blocks, no process crosses splits; same pair ID has same split',reference_uncertainty='128 independent simulated reference processes;6272 correlated snapshots; no IID CI claimed; propagation ranges are not reference CIs',budget_basis='compact bounded batch fixed before method errors, below authorized maxima'),
 rng=dict(generator='MT19937',base_seed=2026101927,split_offsets=OFF,weather_stream_offset=700001,propagation_seed='2026101928+100000*scene_index+1000*K+repeat'),
 AC=dict(solver='existing polar Newton',tolerance_pu=1e-11,max_iterations=40,flat_start=True,fallback=False,PtoV=False),
 fit=dict(K_order=KS,partitions=2,K1_distinct_partitions=1,partition_rule='deterministic balanced PC1/PC2',TMM_initial_df=[12,200],TMM='standard multivariate t PX-ECM, one shared radial Gamma per component',GMM='EM',max_iterations=300,tolerance=1e-6,df_bounds=[4.1,100000],actual_covariance_floor=1e-7,scatter_floor='1e-7*(nu-2)/nu',selection='validation voltage likelihood, original converged-preferred rule; never AC output errors',preprocessing='all training-resolvable real/imag coordinates, common training-only whitening; no variance truncation',special_fitting_audit=False),
 propagation=dict(points=N,repeats=4,sampler='existing weighted component-stratified randomized Sobol',weights='pi_j/n_j',retain='only eight priority outputs rep0 raw samples and weights; every repeat direct empirical metrics, compact CDF/common histogram count+mass+weight2; no 64D point copies'),objects=OBJECTS,
 metrics=dict(quantiles=QUANTILES.tolist(),reference_quantile='ceil(q*n)-1',model_quantile='weighted right empirical inverse',CDF='right ECDF; KS over support union and W1 exact weighted empirical integral',tails=[[.005,.05],[.95,.995]],joint_output_acceptance=False),storage=dict(workspace_write_check=True,bavail_reported_zero=True,bfree_start_bytes=846462976,compact_target_bytes_under=300000000,other_task_cleanup=False,large_archive=False))
 path=O/'protocol.json'
 if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(p))
 else:write_json(path,p)
 write_json(O/'protocol_identity_before_generation.json',dict(SHA256=sha(path),UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())));write_json(O/'environment.json',environment());code=O/'code';code.mkdir(exist_ok=True)
 for name in ['run_cloud_passage_case33.py','cloud_numeric_core.py','run_cloud_study.py','run_typhoon_comparison_K136.py']:shutil.copy2(ROOT/'tests'/name,code/name)
 write_json(O/'run_manifest.json',dict(status='running',PID=os.getpid(),command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -u tests/run_cloud_passage_case33.py',protocol_SHA256=sha(path),code_SHA256={x.name:sha(x) for x in code.glob('*.py')},other_results_unchanged=True));event('protocol_saved_before_generation',AC_total=21952)
def generate(scene,split):
 d=O/scene;d.mkdir(exist_ok=True);path=d/(split+'_inputs.npz')
 if path.exists():return
 n=SPLITS[split];rng=np.random.Generator(np.random.MT19937(2026101927+OFF[split]));weather=np.random.Generator(np.random.MT19937(2026101927+OFF[split]+700001));tau=np.arange(49)*.25
 start=rng.uniform(11,13.8,n);T=truncnorm.rvs(-3,4,loc=28,scale=1,size=n,random_state=rng);hour=start[:,None]+tau/60
 def ar(width):
  a=rng.standard_normal((n,49,width));rho=np.exp(-15/60)
  for j in range(1,49):a[:,j]=rho*a[:,j-1]+np.sqrt(1-rho*rho)*a[:,j]
  return a
 gp=ar(1);gq=.35*gp+np.sqrt(1-.35**2)*ar(1);rp=ar(4);rq=.35*rp+np.sqrt(1-.35**2)*ar(4);zp=ar(32);zq=.25*zp+np.sqrt(1-.25**2)*ar(32)
 ec=.03*ar(1)+.08*ar(4)[:,:,REGIONS]+.15*ar(32);ep=.03*gp+.08*rp[:,:,REGIONS]+.15*zp;eq=.04*gq+.10*rq[:,:,REGIONS]+.20*zq
 pf=rng.uniform(.93,.99,(n,49,32));fp=np.clip(1+ep,.5,1.6);fq=np.clip(1+eq,.4,1.8);fc=np.clip(1+ec,.5,1.6)
 pg=.4*.8*BUS[None,None,1:,2]*fp;qg=.4*.8*BUS[None,None,1:,3]*fq;pc=.6*.8*BUS[None,None,1:,2]*fc;qc=pc*np.tan(np.arccos(pf));lp=pg+pc;lq=qg+qc
 center=weather.uniform(5,7,n);plateau=weather.uniform(2,4,n);down=weather.uniform(.5,1.5,n);up=weather.uniform(.5,1.5,n);lag=weather.uniform(-.5,.5,(n,8));lag[:,6]=.5;lag[:,7]=1
 rawdepth=weather.uniform(.2,.8,n);offset=weather.uniform(-.05,.05,(n,8));weakdepth=weather.uniform(.05,.15,(n,8))
 if scene=='regional_passage':
  depth=np.clip(rawdepth[:,None]+offset,.2,.8);depth[:,6]=.5*rawdepth;depth[:,7]=.25*rawdepth
 else:
  depth=np.zeros((n,8))
  for i in range(n):ids=np.array([i%6,(i+1)%6]);depth[i,ids]=weakdepth[i,ids]
 a=center[:,None]-.5*plateau[:,None]+lag;b=center[:,None]+.5*plateau[:,None]+lag;tt=tau[None,:,None];descent=np.clip((tt-(a[:,None,:]-down[:,None,None]))/down[:,None,None],0,1);recovery=np.clip((tt-b[:,None,:])/up[:,None,None],0,1)
 shade=.5*(1-np.cos(np.pi*descent))*.5*(1+np.cos(np.pi*recovery));assert (shade[:,0]==0).all() and (shade[:,-1]==0).all();reduction=shade*depth[:,None,:]
 raw=(.85-.025*(hour-12.5)**2)[:,:,None]*np.maximum(0,1-.0037*(T-25))[:,None,None]*(1-reduction);pv=CAP*np.clip(raw,0,1);pbus=np.zeros((n,49,32));pbus[:,:,SITES-2]=pv;target=np.concatenate(((pbus-lp)/BASE,-lq/BASE),axis=2)
 def flat(a):return a.reshape((n*49,)+a.shape[2:])
 save_npz(path,target=flat(target),PV_P_MW=flat(pv),load_P_MW=flat(lp),load_Q_MVAr=flat(lq),general_P_MW=flat(pg),general_Q_MVAr=flat(qg),cooling_P_MW=flat(pc),cooling_Q_MVAr=flat(qc),pf=flat(pf),hour=hour.ravel(),temperature_C=np.repeat(T,49),process_id=np.repeat(np.arange(n),49),elapsed_minutes=np.tile(tau,n),shading_fraction=flat(reduction),load_clipping=np.column_stack([(fp!=1+ep).reshape(n*49,32).sum(1),(fq!=1+eq).reshape(n*49,32).sum(1),(fc!=1+ec).reshape(n*49,32).sum(1)]),seed=np.array(2026101927+OFF[split]))
 csv(d/(split+'_process_parameters.csv'),[dict(scene=scene,split=split,process_id=i,solar_start_hour=start[i],temperature_C=T[i],center_minutes=center[i],plateau_minutes=plateau[i],down_minutes=down[i],up_minutes=up[i],depth=json.dumps(depth[i].tolist()),lag_minutes=json.dumps(lag[i].tolist())) for i in range(n)])
 check=dict(scene=scene,split=split,processes=n,snapshots=n*49,observations=0,PV_mean_MW=float(pv.sum(2).mean()),PV_min_MW=float(pv.sum(2).min()),PV_max_MW=float(pv.sum(2).max()),load_P_mean_MW=float(lp.sum(2).mean()),load_Q_mean_MVAr=float(lq.sum(2).mean()),PV_capacity_clip_fraction=float((raw>=1).mean()),PV_zero_fraction=float((pv==0).mean()),net_P_identity_max_MW=float(abs(BASE*target[:,:,:32]+lp-pbus).max()),net_Q_identity_max_MVAr=float(abs(BASE*target[:,:,32:]+lq).max()),SHA256=sha(path));assert np.isfinite(target).all() and pv.min()>=0 and pv.max()<=CAP and check['net_P_identity_max_MW']<1e-12
 write_json(d/(split+'_input_manifest.json'),check);event('inputs_saved',**check)
def ac(scene,split):
 d=O/scene;dest=d/(split+'_AC.npz');audit=d/(split+'_AC_summary.json')
 if dest.exists() and audit.exists():return
 inp=np.load(d/(split+'_inputs.npz'));target=inp['target'];n=len(target);v=np.full((n,33),np.nan+0j);out=np.full((n,65),np.nan);iterations=np.full(n,-1);residual=np.full(n,np.nan);balance=np.full(n,np.nan);failures=[];clock=time.perf_counter();first=0;cp=d/(split+'_AC_checkpoint.npz')
 if cp.exists():
  old=np.load(cp);first=len(old['voltage']);v[:first]=old['voltage'];out[:first]=old['outputs'];iterations[:first]=old['iterations'];residual[:first]=old['residual'];balance[:first]=old['balance']
 for i in range(first,n):
  try:
   vv,it=polar_ac(target[i],Y,tolerance=1e-11);s=vv*np.conj(Y@vv);v[i]=vv;out[i]=evaluate_outputs(vv[None],BASE,BUS,BRANCH)[0];iterations[i]=it;residual[i]=max(abs(np.r_[s[1:].real,s[1:].imag]-target[i]));balance[i]=BASE*s[0].real+BASE*target[i,:32].sum()-out[i,-1]
  except Exception:failures.append(dict(row=i,process_id=int(inp['process_id'][i]),error=traceback.format_exc()))
  if (i+1)%1024==0 or i+1==n:
   save_npz(cp,voltage=v[:i+1],outputs=out[:i+1],iterations=iterations[:i+1],residual=residual[:i+1],balance=balance[:i+1]);write_json(d/(split+'_AC_failures.json'),failures);event('AC_checkpoint',scene=scene,split=split,completed=i+1,total=n,failures=len(failures))
 save_npz(dest,voltage=v,outputs=out,iterations=iterations,residual=residual,active_balance_MW=balance)
 summary=dict(scene=scene,split=split,snapshots=n,processes=SPLITS[split],failures=int((iterations<0).sum()),max_iterations=int(iterations.max()),max_residual_pu=float(np.nanmax(residual)),max_balance_MW=float(np.nanmax(abs(balance))),min_voltage_pu=float(np.nanmin(abs(v))),max_voltage_pu=float(np.nanmax(abs(v))),seconds=time.perf_counter()-clock,input_SHA256=sha(d/(split+'_inputs.npz')),AC_SHA256=sha(dest));write_json(audit,summary)
 if cp.exists():write_json(d/(split+'_checkpoint_retired.json'),dict(path=str(cp),bytes=cp.stat().st_size,SHA256=sha(cp),retained_AC_SHA256=sha(dest)));cp.unlink()
 assert not summary['failures'],'failures retained; no silent sample removal'
def prepare(scene):
 d=O/scene
 if (d/'prepared.json').exists():return
 clock=time.perf_counter();acs={s:np.load(d/(s+'_AC.npz')) for s in SPLITS};states={s:np.c_[a['voltage'][:,1:].real,a['voltage'][:,1:].imag] for s,a in acs.items()};center=states['training'].mean(0);_,sv,vt=np.linalg.svd(states['training']-center,full_matrices=False);rank=int(sum(sv>max(states['training'].shape)*np.spacing(sv[0])));basis=vt[:rank].T;scale=sv[:rank]/np.sqrt(len(states['training'])-1);x=(states['training']-center)@basis/scale;z=(states['validation']-center)@basis/scale;actual=outputs(acs['validation']['voltage']);recon=outputs(voltage((z*scale)@basis.T+center))
 csv(d/'PCA_output_reconstruction.csv',[dict(**o,maximum_absolute_reconstruction_error=float(abs(actual[:,j]-recon[:,j]).max())) for j,o in enumerate(OBJECTS)]);save_npz(d/'fitting_inputs.npz',x=x,validation=z,center=center,basis=basis,scale=scale,singular_values=sv)
 ref=outputs(acs['reference']['voltage']);assert abs(ref[:,32:]-acs['reference']['outputs']).max()<1e-10;save_npz(d/'reference_outputs.npz',values=ref);std=ref.std(0);edges=np.array([np.linspace(r.min()-4*s,r.max()+4*s,241) for r,s in zip(ref.T,std)]);save_npz(d/'reference_distribution_data.npz',edges=edges,counts=np.array([np.histogram(r,e)[0] for r,e in zip(ref.T,edges)]),mean=ref.mean(0),std=std)
 inp=np.load(d/'reference_inputs.npz');v=acs['reference']['voltage'];sources=dict(PV_P_MW=inp['PV_P_MW'].sum(1),load_P_MW=inp['load_P_MW'].sum(1),load_Q_MVAr=inp['load_Q_MVAr'].sum(1),shading_fraction=inp['shading_fraction'][:,:6].mean(1),minimum_voltage_pu=abs(v[:,1:]).min(1),loss_MW=ref[:,-1],angle12_minus13_rad=np.angle(v[:,11])-np.angle(v[:,12]));sources['net_import_MW']=sources['load_P_MW']-sources['PV_P_MW'];sources['joint_voltage_deviation']=np.linalg.norm(states['reference']-center,axis=1);shapes=[];conditions=[]
 for j,o in enumerate(OBJECTS):
  rr=ref[:,j];q=np.sort(rr)[np.ceil(QUANTILES*len(rr)).astype(int)-1];sd=rr.std();shapes.append(dict(**o,scene=scene,reference_mean=rr.mean(),reference_std=sd,skewness=float(skew(rr)),excess_kurtosis=float(kurtosis(rr)),q01_standardized=(q[1]-rr.mean())/sd,q99_standardized=(q[6]-rr.mean())/sd,**{f'q{int(1000*t):03d}':val for t,val in zip(QUANTILES,q)},reference_processes=SPLITS['reference'],reference_snapshots=len(rr)))
  if o['priority']:
   for side,mask in [('lower',rr<=q[3]),('upper',rr>=q[4])]:conditions.append(dict(**o,scene=scene,side=side,snapshots=int(mask.sum()),processes=int(len(np.unique(inp['process_id'][mask]))),**{name+'_mean':float(a[mask].mean()) for name,a in sources.items()}))
 csv(d/'reference_distribution_shapes.csv',shapes);csv(d/'AC_tail_source_conditions.csv',conditions);shade=sources['shading_fraction'];groups=[]
 for name,mask in [('before_after',shade==0),('partially_shaded',(shade>0)&(shade<.2)),('deeply_shaded',shade>=.2)]:
  if mask.any():groups.append(dict(scene=scene,group=name,snapshots=int(mask.sum()),**{key+'_mean':float(a[mask].mean()) for key,a in sources.items()},**{f'V{b}_mean_pu':float(abs(v[mask,b-1]).mean()) for b in [12,13,18]},**{f'V{b}_real_mean':float(v[mask,b-1].real.mean()) for b in [12,13,18]},**{f'V{b}_imag_mean':float(v[mask,b-1].imag.mean()) for b in [12,13,18]}))
 csv(d/'physical_shading_groups.csv',groups);write_json(d/'prepared.json',dict(scene=scene,dimension=rank,coordinate_order='bus2..33 real then bus2..33 imaginary; slack1=1+0j',preprocessing_seconds=time.perf_counter()-clock,training_snapshots=len(x),validation_snapshots=len(z),reference_snapshots=len(ref),max_voltage_coordinate_reconstruction_error=float(abs((z*scale)@basis.T+center-states['validation']).max()),fitting_inputs_SHA256=sha(d/'fitting_inputs.npz')));event('prepared',scene=scene,dimension=rank)
source=ROOT/'tests/run_typhoon_comparison_K136.py';tree=ast.parse(source.read_text());pieces=[]
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in ['fits','evaluate','summarize']:
  s=ast.get_source_segment(source.read_text(),node)
  if node.name=='evaluate':
   s=s.replace('2026101922','2026101928').replace('propseconds=time.perf_counter()-clock;',"if rep==0:save_npz(dest/(stem+'_saved_outputs.npz'),values=out[:,PRIMARY],weights=w,object_indices=PRIMARY)\n            propseconds=time.perf_counter()-clock;")
   s=s.replace("save_npz(dest/(stem+'_curves.npz'),cdf=np.array(curves),", "bin_counts=np.array([np.histogram(out[:,j],edges[j])[0] for j in range(len(OBJECTS))])\n            bin_weight2=np.array([np.histogram(out[:,j],edges[j],weights=w*w)[0] for j in range(len(OBJECTS))])\n            save_npz(dest/(stem+'_curves.npz'),bin_counts=bin_counts,bin_weight2=bin_weight2,cdf=np.array(curves),")
  if node.name=='summarize':s=s.replace("'W*/","'*/")
  pieces.append(s)
exec('\n\n'.join(pieces),globals())
def first_table(k):
 g=pd.read_csv(O/'global_repeat_summary.csv');q=pd.read_csv(O/'quantile_repeat_summary.csv');rows=[]
 for scene in SCENES:
  for j in PRIMARY:
   o=OBJECTS[j];row=dict(scene=scene,object=o['output']+str(o['node'] or o['branch'] or ''),K=k,unit=o['unit'])
   for family in ['GMM','TMM']:
    a=g[(g.scene==scene)&(g.K==k)&(g.family==family)&(g.output==o['output'])&(g.node==o['node'])&(g.branch==o['branch'])].iloc[0]
    for m in ['ks','w1']:row[family+'_'+m]=a[m+'_mean']
    for t,label in [(.01,'q01'),(.99,'q99')]:
     b=q[(q.scene==scene)&(q.K==k)&(q.family==family)&(q.output==o['output'])&(q.node==o['node'])&(q.branch==o['branch'])&(q.q==t)].iloc[0];row[label+'_AC']=b.reference_quantile_mean
     for m in ['absolute_error','signed_error','model_quantile']:
      row[family+'_'+label+'_'+m]=b[m+'_mean']
      if m=='absolute_error':row[family+'_'+label+'_range']=json.dumps([b[m+'_min'],b[m+'_max']])
   rows.append(row)
 csv(O/f'priority_K{k}_eight_objects.csv',rows)
 if k==3:
  cols=['scene','object','GMM_ks','TMM_ks','GMM_q01_absolute_error','TMM_q01_absolute_error','GMM_q99_absolute_error','TMM_q99_absolute_error'];print('K3_EIGHT_OBJECTS\n'+pd.DataFrame(rows)[cols].to_csv(index=False),flush=True)
def main():
 protocol();(O/'code/adapted_routines.py').write_text('\n\n'.join(pieces))
 for scene in SCENES:
  for split in SPLITS:generate(scene,split);ac(scene,split)
  prepare(scene)
 csv(O/'AC_summary.csv',[json.loads((O/s/(p+'_AC_summary.json')).read_text()) for s in SCENES for p in SPLITS]);csv(O/'input_summary.csv',[json.loads((O/s/(p+'_input_manifest.json')).read_text()) for s in SCENES for p in SPLITS]);event('inputs_AC_complete',AC_rows=21952)
 for k in KS:
  for scene in SCENES:fits(scene,k);evaluate(scene,k)
  summarize();first_table(k);event('K_stage_complete',K=k)
 qq=pd.read_csv(O/'quantile_metrics.csv');gg=pd.read_csv(O/'global_metrics.csv');coverage=qq.groupby(['scene','K','family','propagation_replicate']).agg(rows=('q','size'),objects=('output','size')).reset_index();assert len(coverage)==80 and (coverage.rows==776).all();assert len(qq)==62080 and len(gg)==7760
 csv(O/'coverage_check.csv',coverage);write_json(O/'comparison_complete.json',dict(status='numerics_complete',models=20,propagations=80,AC=21952,raw_quantile_rows=len(qq),raw_global_rows=len(gg),wall_seconds=time.perf_counter()-START,CPU_seconds=time.process_time()-CPU));event('comparison_numerics_complete')
if __name__=='__main__':
 try:main()
 except Exception:event('failed',error=traceback.format_exc());raise
