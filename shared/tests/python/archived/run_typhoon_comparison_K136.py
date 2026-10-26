"""Cloud-only fixed typhoon inputs: standard joint-voltage GMM/TMM, K1/3/6.

No source generation, AC solve, PtoV, fitting audit, or larger K is performed.
Resume uses completed candidate JSON/NPZ and per-propagation completion markers.
"""
from pathlib import Path
import os, sys, time, json, re, traceback, shutil, platform
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from cloud_numeric_core import (ROOT, FLOOR, QUANTILES, sha, write_json,
    admittance, evaluate_outputs, initialize, fit, log_density, propagate)
from run_cloud_study import save_fit, distribution_metrics

R=ROOT/'result/case33bw/kelmarsh_wind_20261019'
I=R/'typhoon_input_preparation_20261019'
O=R/'typhoon_comparison_K136_20261019'
O.mkdir(exist_ok=True)
SCENES=['W0','W1','W2','W3']; KS=[1,3,6]; N=65536
OBJECTS=[dict(output='V',node=j,branch=0,unit='p.u.',priority=j in [18,25,33]) for j in range(2,34)]
OBJECTS += [dict(output=q,node=0,branch=j,unit='MW' if q=='P' else 'MVAr',priority=j in [1,2,17,24,32]) for q in ['P','Q'] for j in range(1,33)]
OBJECTS += [dict(output='loss',node=0,branch=0,unit='MW',priority=True)]
START=time.perf_counter(); CPU=time.process_time()
def event(stage,**kw):
    counts=dict(candidate_records=len(list(O.glob('W*/candidates/K*.json'))),
      selected_models=sum(len(pd.read_csv(p)) for p in O.glob('W*/selected_models.csv')),
      completed_propagations=len(list(O.glob('W*/propagation/*_complete.json'))),
      total_selected_models=24,total_propagations=96,total_planned_candidates=60)
    row=dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),PID=os.getpid(),
      stage=stage,**counts,**kw)
    write_json(O/'progress.json',row)
    with (O/'progress_history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def save_npz(path,**arrays):
    p=Path(path);t=p.with_name(p.stem+'.tmp.npz');np.savez_compressed(t,**arrays);os.replace(t,p)
def csv(path,rows):
    p=Path(path);t=p.with_suffix('.tmp.csv');pd.DataFrame(rows).to_csv(t,index=False);os.replace(t,p)
def network():
    text=re.sub(r'%[^\n]*','',(ROOT/'data/cases/case33bw_modify.m').read_text());tables={}
    for name in ['bus','branch']:
        block=re.search(r'mpc\.'+name+r'\s*=\s*\[(.*?)\];',text,re.S)[1]
        tables[name]=np.array([[float(v) for v in row.split()] for row in block.split(';') if row.strip()])
    base=float(re.search(r'mpc.baseMVA\s*=\s*([\d.]+)',text)[1]);bus,branch=tables['bus'],tables['branch']
    bus[:,2:4]/=1000;branch[:,2:4]/=(bus[0,9]*1000)**2/(base*1e6)
    return base,bus,branch
BASE,BUS,BRANCH=network()
def voltage(state):return np.column_stack((np.ones(len(state)),state[:,:32]+1j*state[:,32:]))
def outputs(v):return np.column_stack((abs(v[:,1:]),evaluate_outputs(v,BASE,BUS,BRANCH)))
def protocol():
    paths=[I/'source_input_config.json',I/'process_source_record_split.csv',ROOT/'data/cases/case33bw_modify.m']
    for s in SCENES:paths += [I/s/'inputs.npz',I/s/'standard_AC.npz']
    p=dict(version='typhoon-comparison-K136-v1',execution='OpenAI_cloud_only',
      environment=dict(host=platform.node(),platform=platform.platform(),python=sys.version,numpy=np.__version__),
      input_root=str(I.relative_to(ROOT)),source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in paths},
      scenes=SCENES,K=KS,models=24,propagations=96,source_or_AC_samples_added=0,
      split=dict(training=1314,validation=365,reference=365,training_processes=18,validation_processes=5,reference_processes=5,
        definition='original process IDs, same across four scenarios; no resplit or SCADA resampling'),
      state='non-slack bus2..33 real then imaginary; slack bus1=1+0j; standard AC stored voltages only; no PtoV',
      preprocessing=dict(training_only=True,PCA_discard_variance_steps=[1e-6,1e-8,0],
        maximum_validation_coordinate_reconstruction_pu=1e-4,whitening='train covariance eigenvalue square root'),
      fit=dict(GMM='ordinary EM',TMM='standard multivariate t classical PX-ECM; one shared radial Gamma per component',
        partitions=2,K1_distinct_partitions=1,partition_definition='PC1 and PC2 deterministic balanced',
        TMM_initial_df=[12,200],TMM_route='GMM warm start',df_bounds=[4.1,1e5],maximum_iterations=300,
        relative_tolerance=1e-6,actual_covariance_floor=FLOOR,TMM_scatter_floor='floor*(nu-2)/nu',
        selection='held-out joint voltage likelihood, converged candidates preferred, no AC output errors',
        no_refinement=True,no_multistart_audit=True),
      propagation=dict(method='component-weighted randomized stratified Sobol',points=N,repeats=4,
        seed='2026101922+100000*scene_index+1000*K+repeat, repeat0..3; MT19937 scrambling',
        weights='pi_j/n_j, integer counts floor desired then residual adjustment; actual counts and ESS saved',
        n_is_not_independent_observations=True),
      metrics=dict(quantiles=QUANTILES.tolist(),reference_quantile='right empirical inverse: ceil(q*n)-1',
        model_quantile='weighted right inverse empirical CDF',KS='right empirical CDF over union of all sorted values; includes both jump limits through consecutive distinct support',
        W1='exact integral of absolute CDF difference for saved weighted empirical laws',
        tail_CDF='at reference right ECDF points in [.005,.05] and [.95,.995], original probability scale',
        reference_uncertainty='365 snapshots from5 temporally structured held-out processes; q.005/q.995 about2 order records. No IID CI claimed; no resampling reference.'),
      density=dict(fixed_bins='240 common bins per variable from AC min-4std to max+4std; same for all K/families/repeats',
        outside_bin_mass_saved=True,normalize='counts / full model mass / bin width, no within-range or tail renormalization',
        points='zero and exact source control boundaries identified in input stage; electrical output zero mass explicitly saved',
        role='finite-reference histogram diagnosis; no smoothing'),
      objects=OBJECTS,scope='only existing four input sets and K1/3/6; no new AC/PtoV/inputs or fit audits',
      parameter_count='GMM=(K-1)+K*d+K*d*(d+1)/2; TMM adds K, joint model counted once')
    if (O/'comparison_protocol.json').exists():
        old=json.loads((O/'comparison_protocol.json').read_text());assert old['source_SHA256']==p['source_SHA256']
    else:write_json(O/'comparison_protocol.json',p)
    code=O/'code';code.mkdir(exist_ok=True)
    for n in ['run_typhoon_comparison_K136.py','cloud_numeric_core.py','run_cloud_study.py']:
        shutil.copy2(ROOT/'tests'/n,code/n)
    write_json(O/'run_manifest.json',dict(status='running',protocol_SHA256=sha(O/'comparison_protocol.json'),
      source_SHA256=p['source_SHA256'],code_SHA256={n:sha(code/n) for n in [x.name for x in code.glob('*.py')]},
      command=' '.join(sys.argv),PID=os.getpid(),numerical_location='OpenAI cloud Linux',new_AC_solves=0,new_source_samples=0))
    membership=pd.read_csv(I/'process_source_record_split.csv')
    assert membership.groupby('process_id').split.nunique().max()==1
    assert membership.source_record_id.nunique()==2044
    event('protocol_saved_before_fits')
def prepare(scene):
    d=O/scene;d.mkdir(exist_ok=True)
    if (d/'prepared.json').exists():return
    clock=time.perf_counter();inp=np.load(I/scene/'inputs.npz');ac=np.load(I/scene/'standard_AC.npz')
    v=ac['voltage'];assert v.shape==(2044,33) and np.isfinite(v).all() and np.max(abs(v[:,0]-1))<1e-14
    state=np.column_stack((v[:,1:].real,v[:,1:].imag));split=inp['split'];masks={s:split==s for s in ['training','validation','reference']}
    assert [int(m.sum()) for m in masks.values()]==[1314,365,365]
    # Necessary output identity, no AC solve.
    saved=ac['outputs'];forward=evaluate_outputs(v,BASE,BUS,BRANCH)
    assert np.max(abs(saved-forward))<1e-10
    states=[state[masks[s]] for s in ['training','validation']]
    center=states[0].mean(0);_,sv,vt=np.linalg.svd(states[0]-center,full_matrices=False)
    rank=int(sum(sv>max(states[0].shape)*np.spacing(sv[0])));var=sv**2/(len(states[0])-1);attempts=[]
    for tol in [1e-6,1e-8,0]:
        dim=rank if tol==0 else min(rank,int(np.searchsorted(np.cumsum(var)/var.sum(),1-tol)+1))
        basis=vt[:dim].T;scale=np.sqrt(var[:dim]);x=((states[0]-center)@basis)/scale
        z=((states[1]-center)@basis)/scale;reconstructed=(z*scale)@basis.T+center
        error=float(np.max(abs(reconstructed-states[1])));attempts.append(dict(tolerance=tol,dimension=dim,max_coordinate_error=error))
        if error<=1e-4:break
    assert error<=1e-4
    fullref=outputs(v[masks['reference']]);reconstructed_out=outputs(voltage(reconstructed));actual_out=outputs(v[masks['validation']])
    save_npz(d/'fitting_inputs.npz',x=x,validation=z,center=center,basis=basis,scale=scale,
      source_train_rows=np.flatnonzero(masks['training']),source_validation_rows=np.flatnonzero(masks['validation']))
    save_npz(d/'reference_outputs.npz',values=fullref,source_rows=np.flatnonzero(masks['reference']))
    reconstruction=[{**o,'maximum_absolute_reconstruction_error':float(np.max(abs(reconstructed_out[:,j]-actual_out[:,j])))} for j,o in enumerate(OBJECTS)]
    csv(d/'PCA_output_reconstruction.csv',reconstruction)
    write_json(d/'prepared.json',dict(scene=scene,dimension=dim,numerical_rank=rank,PCA_attempts=attempts,
      output_identity_max_error=float(np.max(abs(saved-forward))),slack_max_error=float(np.max(abs(v[:,0]-1))),
      fitting_inputs_SHA256=sha(d/'fitting_inputs.npz'),reference_SHA256=sha(d/'reference_outputs.npz'),
      preprocessing_seconds=time.perf_counter()-clock))
    # Reference-only electrical shape and source-state descriptions, before method output errors.
    shapes=[];conditions=[];refrows=np.flatnonzero(masks['reference'])
    p=inp['wind_group_P_MW'].sum(1);q=inp['wind_group_Q_MVAr'].sum(1)
    connected=inp['connected'].reshape(2044,6).mean(1)
    source=dict(wind_P_MW=p,wind_Q_MVAr=q,load_P_MW=inp['load_P_MW'].sum(1),load_Q_MVAr=inp['load_Q_MVAr'].sum(1),
      connected_unit_fraction=connected,weather_load_multiplier=inp['weather_load_multiplier'],
      minimum_voltage_pu=abs(v[:,1:]).min(1),loss_MW=saved[:,-1])
    for j,o in enumerate(OBJECTS):
        rr=fullref[:,j];rs=np.sort(rr);rq=rs[np.ceil(QUANTILES*len(rr)).astype(int)-1];std=rr.std()
        shapes.append({**o,'scene':scene,'n':len(rr),'reference_mean':rr.mean(),'reference_std':std,
          'skewness':float(skew(rr)),'excess_kurtosis':float(kurtosis(rr)),
          'q01_standardized':float((rq[1]-rr.mean())/std),'q99_standardized':float((rq[6]-rr.mean())/std),
          **{f'q{int(1000*t):03d}':float(x) for t,x in zip(QUANTILES,rq)},
          'process_count':5,'reference_independence':'not IID; five full held-out processes'})
        if o['priority']:
            for side,mask in [('lower',rr<=rq[3]),('upper',rr>=rq[4])]:
                ids=refrows[mask]
                conditions.append({**o,'scene':scene,'side':side,'snapshots':len(ids),
                  'processes':int(len(np.unique(inp['process_id'][ids]))),
                  **{n+'_mean':float(a[ids].mean()) for n,a in source.items()},
                  'source_rows':json.dumps(ids.tolist())})
    csv(d/'reference_distribution_shapes.csv',shapes);csv(d/'AC_tail_source_conditions.csv',conditions)
    mean=fullref.mean(0);std=fullref.std(0);edges=np.array([np.linspace(min(a)-4*s,max(a)+4*s,241) for a,s in zip(fullref.T,std)])
    # Common model-independent histogram definition, save empirical reference counts too.
    hist=np.array([np.histogram(a,e)[0] for a,e in zip(fullref.T,edges)])
    save_npz(d/'reference_distribution_data.npz',edges=edges,counts=hist,values=fullref,mean=mean,std=std)
    event('prepared',scene=scene,dimension=dim)
def fits(scene,k):
    d=O/scene;c=d/'candidates';c.mkdir(exist_ok=True);a=np.load(d/'fitting_inputs.npz');x,z=a['x'],a['validation']
    for p in range(1 if k==1 else 2):
        ini,labels,meta=initialize(x,k,p,a['scale'],population_covariance=True)
        ip=c/f'K{k}_p{p+1:02d}_initial.npz'
        if not ip.exists():save_npz(ip,**ini,labels=labels)
        gp=c/f'K{k}_p{p+1:02d}_GMM'
        if not gp.with_suffix('.json').exists():
            event('fitting',scene=scene,K=k,family='GMM',partition=p+1)
            m,audit=fit(x,ini,actual_covariance_floor=True,relative_tolerance=True)
            save_fit(gp,m,audit,z,dict(scene=scene,K=k,family='GMM',acceleration='EM',initial_df='Infinity',**meta))
        gm=np.load(gp.with_suffix('.npz'))
        for nu in [12,200]:
            tp=c/f'K{k}_p{p+1:02d}_gmm_warm_df{nu}_PX'
            if tp.with_suffix('.json').exists():continue
            event('fitting',scene=scene,K=k,family='TMM',partition=p+1,initial_df=nu)
            start={n:gm[n].copy() for n in ['weights','means','scatter','df']};start['df'][:]=nu;start['scatter']*=(nu-2)/nu
            try:
                m,audit=fit(x,start,'PX',actual_covariance_floor=True,relative_tolerance=True)
                save_fit(tp,m,audit,z,dict(scene=scene,K=k,family='TMM',acceleration='PX',initial_df=nu,**meta))
            except Exception:
                write_json(tp.with_suffix('.json'),dict(scene=scene,K=k,family='TMM',status='exception',error=traceback.format_exc(),**meta))
    attempts=[]
    for p in sorted(c.glob('K*.json')):
        aa=json.loads(p.read_text());aa['candidate']=p.stem;attempts.append(aa)
    csv(d/'all_fit_attempts.csv',attempts)
    selected=[]
    for kk in KS:
        for fam in ['GMM','TMM']:
            eligible=[r for r in attempts if r['K']==kk and r['family']==fam and 'validation_ll' in r]
            if not eligible:continue
            conv=[r for r in eligible if r['status']=='converged'];b=max(conv or eligible,key=lambda r:r['validation_ll'])
            dim=x.shape[1];params=(kk-1)+kk*dim+kk*dim*(dim+1)//2+(kk if fam=='TMM' else 0)
            selected.append(dict(scene=scene,K=kk,family=fam,candidate=b['candidate'],validation_ll=b['validation_ll'],
              train_ll=b['train_ll'],status=b['status'],iterations=b['iterations'],dimension=dim,parameter_count=params,
              df=json.dumps(b['df']),weights=json.dumps(b['weights']),model_SHA256=b['model_sha256']))
    csv(d/'selected_models.csv',selected)
    event('selected',scene=scene,K=k)
def evaluate(scene,k):
    d=O/scene;dest=d/'propagation';dest.mkdir(exist_ok=True);a=np.load(d/'fitting_inputs.npz')
    ref=np.load(d/'reference_outputs.npz')['values'];histref=np.load(d/'reference_distribution_data.npz');edges=histref['edges']
    for r in pd.read_csv(d/'selected_models.csv').query('K==@k').to_dict('records'):
        m=np.load(d/'candidates'/(r['candidate']+'.npz'))
        for rep in range(4):
            stem=f"{r['family']}_K{k}_r{rep}";done=dest/(stem+'_complete.json')
            if done.exists():continue
            seed=2026101922+100000*SCENES.index(scene)+1000*k+rep
            event('propagating',scene=scene,K=k,family=r['family'],replicate=rep,seed=seed)
            clock=time.perf_counter();z,w=propagate(m,N,seed,'sobol',randomizer='MT19937')
            state=(z*a['scale'])@a['basis'].T+a['center'];v=voltage(state);out=outputs(v)
            propseconds=time.perf_counter()-clock;assert np.isfinite(out).all();assert abs(w.sum()-1)<1e-12
            clock=time.perf_counter();gg=[];qq=[];curves=[];hh=[];masses=[]
            for j,o in enumerate(OBJECTS):
                meta=dict(scene=scene,K=k,family=r['family'],propagation_replicate=rep,seed=seed,n=N,
                  candidate=r['candidate'],dimension=r['dimension'],parameter_count=r['parameter_count'],**o)
                g,q,curve=distribution_metrics(ref[:,j],out[:,j],w)
                rr=np.sort(ref[:,j]);ix=np.argsort(out[:,j]);ps=out[ix,j];cw=np.cumsum(w[ix]);cw/=cw[-1]
                f=np.r_[0,cw][np.searchsorted(ps,rr,side='right')];rf=np.arange(1,len(rr)+1)/len(rr)
                for side,lo,hi in [('lower',.005,.05),('upper',.95,.995)]:
                    delta=f[(rf>=lo)&(rf<=hi)]-rf[(rf>=lo)&(rf<=hi)]
                    g[side+'_tail_cdf_sup_abs']=float(np.max(abs(delta)))
                    g[side+'_tail_cdf_mean_abs']=float(np.mean(abs(delta)))
                    g[side+'_tail_cdf_signed_mean']=float(np.mean(delta))
                g['mean_absolute_error']=abs(g['signed_mean_error']);g['variance_signed_error']=g['model_variance']-g['reference_variance']
                g['variance_absolute_error']=abs(g['variance_signed_error'])
                gg.append({**meta,**g})
                for row in q:
                    # Original binomial intervals presume IID; not valid for this process-held-out reference.
                    row={n:x for n,x in row.items() if not n.startswith('reference_quantile_ci') and n!='reference_cdf_pointwise_se'}
                    qq.append({**meta,**row})
                curves.append(np.array(curve))
                counts=np.histogram(out[:,j],edges[j],weights=w)[0];hh.append(counts/np.diff(edges[j]))
                masses.append([w[out[:,j]<edges[j,0]].sum(),w[out[:,j]>edges[j,-1]].sum(),w[out[:,j]==0].sum()])
            csv(dest/(stem+'_global.csv'),gg);csv(dest/(stem+'_quantiles.csv'),qq)
            save_npz(dest/(stem+'_curves.npz'),cdf=np.array(curves),histogram_density=np.array(hh),outside_and_zero_mass=np.array(masses))
            desired=N*m['weights'];counts=np.maximum(1,np.floor(desired).astype(int))
            while counts.sum()<N:counts[np.argmax(desired-counts)]+=1
            while counts.sum()>N:
                ids=np.flatnonzero(counts>1);counts[ids[np.argmin((desired-counts)[ids])]]-=1
            write_json(done,dict(scene=scene,K=k,family=r['family'],replicate=rep,seed=seed,
              candidate=r['candidate'],model_SHA256=r['model_SHA256'],component_counts=counts.tolist(),
              weights_sum=float(w.sum()),ESS=float(1/np.dot(w,w)),
              propagation_seconds=propseconds,processing_seconds=time.perf_counter()-clock,
              all_outputs_finite=True,global_rows=97,quantile_rows=776))
            event('propagation_complete',scene=scene,K=k,family=r['family'],replicate=rep)
def summarize():
    gg=pd.concat([pd.read_csv(p) for p in O.glob('W*/propagation/*_global.csv')],ignore_index=True)
    qq=pd.concat([pd.read_csv(p) for p in O.glob('W*/propagation/*_quantiles.csv')],ignore_index=True)
    gg.to_csv(O/'global_metrics.csv',index=False);qq.to_csv(O/'quantile_metrics.csv',index=False)
    keys=['scene','K','family','output','node','branch','unit','priority','dimension','parameter_count']
    gm=['ks','w1','model_mean','reference_mean','signed_mean_error','mean_absolute_error','model_variance','reference_variance','variance_signed_error','variance_absolute_error','lower_tail_cdf_sup_abs','lower_tail_cdf_mean_abs','lower_tail_cdf_signed_mean','upper_tail_cdf_sup_abs','upper_tail_cdf_mean_abs','upper_tail_cdf_signed_mean']
    qm=['reference_quantile','model_quantile','signed_error','absolute_error','reference_cdf_error']
    def aggregate(df,keys,metrics,name):
        a=df.groupby(keys,dropna=False)[metrics].agg(['mean','min','max']);a.columns=['_'.join(c) for c in a.columns]
        a=a.reset_index();a.to_csv(O/name,index=False);return a
    gs=aggregate(gg,keys,gm,'global_repeat_summary.csv');qs=aggregate(qq,keys+['q'],qm,'quantile_repeat_summary.csv')
    comp=[];same=keys.copy();same.remove('family');same.remove('parameter_count')
    for kind,table,metrics,kk in [('global',gs,['ks','w1','lower_tail_cdf_sup_abs','upper_tail_cdf_sup_abs'],same),
        ('quantile',qs,['absolute_error'],same+['q'])]:
        for identity,group in table.groupby(kk,dropna=False):
            if set(group.family)!=set(['GMM','TMM']):continue
            g=group[group.family=='GMM'].iloc[0];t=group[group.family=='TMM'].iloc[0]
            for metric in metrics:
                gmean=g[metric+'_mean'];tmean=t[metric+'_mean'];delta=gmean-tmean
                state='TMM better beyond propagation range' if t[metric+'_max']<g[metric+'_min'] else 'GMM better beyond propagation range' if g[metric+'_max']<t[metric+'_min'] else 'propagation ranges overlap'
                comp.append(dict(zip(kk,identity))|dict(kind=kind,metric=metric,GMM_error_mean=gmean,TMM_error_mean=tmean,
                  GMM_minus_TMM_error=delta,GMM_min=g[metric+'_min'],GMM_max=g[metric+'_max'],TMM_min=t[metric+'_min'],TMM_max=t[metric+'_max'],comparison=state,
                  reference_value=g.get('reference_quantile_mean',np.nan),GMM_signed_error=g.get('signed_error_mean',np.nan),TMM_signed_error=t.get('signed_error_mean',np.nan)))
    ct=pd.DataFrame(comp);ct.to_csv(O/'same_K_object_comparison.csv',index=False)
    # Symmetric achieved-error references, inspect all measured K including K1.
    matched=[]
    for kind,table,metrics,kk in [('global',gs,['ks','w1','lower_tail_cdf_sup_abs','upper_tail_cdf_sup_abs'],['scene','output','node','branch','unit','priority','dimension']),
        ('quantile',qs,['absolute_error'],['scene','output','node','branch','unit','priority','dimension','q'])]:
        for identity,group in table.groupby(kk,dropna=False):
            for metric in metrics:
                for _,target in group.iterrows():
                    threshold=target[metric+'_mean'];row=dict(zip(kk,identity))|dict(kind=kind,metric=metric,reference_family=target.family,reference_K=int(target.K),actual_error_reference=threshold)
                    for fam in ['GMM','TMM']:
                        ok=group[(group.family==fam)&(group[metric+'_mean']<=threshold)].sort_values('K')
                        row[fam+'_minimum_tested_K']=int(ok.iloc[0].K) if len(ok) else np.nan
                        row[fam+'_parameters']=int(ok.iloc[0].parameter_count) if len(ok) else np.nan
                    matched.append(row)
    csv(O/'matched_accuracy_K_parameters.csv',matched)
    for name in ['reference_distribution_shapes.csv','AC_tail_source_conditions.csv','selected_models.csv','all_fit_attempts.csv']:
        tables=[pd.read_csv(p) for p in O.glob('W*/'+name)]
        if tables:pd.concat(tables,ignore_index=True).to_csv(O/name,index=False)
    priority=ct[ct.priority==True];priority.to_csv(O/'priority_same_K_comparison.csv',index=False)
    wins=priority[(priority.comparison=='TMM better beyond propagation range')&((priority.kind=='global')|(priority.q.isin([.01,.05,.95,.99])))].sort_values('GMM_minus_TMM_error',ascending=False)
    wins.head(30).to_csv(O/'priority_local_advantages.csv',index=False)
    print('PRIORITY_LOCAL_ROWS',wins.head(8).to_csv(index=False),flush=True)
def plots():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=O/'figures';dest.mkdir(exist_ok=True);pdest=O/'plot_data';pdest.mkdir(exist_ok=True)
    table=pd.read_csv(O/'same_K_object_comparison.csv')
    priority=table[table.priority==True].copy()
    chosen=[]
    # One fixed接入 endpoint voltage + collection P first, then largest observed local advantage.
    for scene,k,output,node,branch in [('W2',1,'V',18,0),('W2',1,'P',0,17)]:chosen.append((scene,k,output,node,branch))
    ranked=priority[priority.comparison=='TMM better beyond propagation range'].sort_values('GMM_minus_TMM_error',ascending=False)
    for r in ranked.to_dict('records'):
        item=(r['scene'],int(r['K']),r['output'],int(r['node']),int(r['branch']))
        if item not in chosen:chosen.append(item)
        if len(chosen)>=8:break
    index=[]
    for scene,k,output,node,branch in chosen:
        j=next(j for j,o in enumerate(OBJECTS) if o['output']==output and o['node']==node and o['branch']==branch)
        o=OBJECTS[j];d=O/scene;rr=np.load(d/'reference_outputs.npz')['values'][:,j];h=np.load(d/'reference_distribution_data.npz');edges=h['edges'][j]
        rs=np.sort(rr);cuts=rs[np.ceil(np.array([.005,.05,.95,.995])*len(rs)).astype(int)-1]
        lo,hi=rr.min(),rr.max();margin=(hi-lo)*.06;limits=[(lo-margin,hi+margin),(cuts[0],cuts[1]),(cuts[2],cuts[3])]
        fig,axes=plt.subplots(3,2,figsize=(10,9),layout='constrained');data=[]
        for row,lim in enumerate(limits):
            ax,ac=axes[row];bins=np.linspace(lo-margin,hi+margin,31);cnt,be=np.histogram(rr,bins)
            ax.stairs(cnt/len(rr)/np.diff(be),be,color='black',label='AC reference (365)')
            ac.step(rs,np.arange(1,len(rs)+1)/len(rs),where='post',color='black',label='AC')
            for family,color,style in [('GMM','blue','--'),('TMM','red','-')]:
                for rep in range(4):
                    a=np.load(d/'propagation'/f'{family}_K{k}_r{rep}_curves.npz');g=a['cdf'][j]
                    ax.stairs(a['histogram_density'][j],edges,color=color,linestyle=style,alpha=1 if rep==0 else .2,label=family if rep==0 else None)
                    ac.plot(g[0],g[1],color=color,linestyle=style,alpha=1 if rep==0 else .2,label=family if rep==0 else None)
                    if row==0:
                        data.extend(dict(curve='CDF',family=family,replicate=rep,x=float(x),y=float(y)) for x,y in zip(g[0],g[1]))
                        data.extend(dict(curve='histogram_density',family=family,replicate=replicate,bin_left=float(l),bin_right=float(r),y=float(y)) for replicate in [rep] for l,r,y in zip(edges[:-1],edges[1:],a['histogram_density'][j]))
            for aa in [ax,ac]:aa.set_xlim(*lim);aa.grid(alpha=.16);aa.set_xlabel(o['unit'])
            ax.set_ylim(bottom=0);ax.set_ylabel('Density ('+o['unit']+'$^{-1}$)');ac.set_ylabel('CDF');ac.set_ylim(0,1)
            ax.set_title(['Full distribution','Lower tail','Upper tail'][row]);ax.legend(fontsize=8);ac.legend(fontsize=8)
            if row:
                vals=[]
                for line in ac.lines:
                    xx=np.asarray(line.get_xdata());yy=np.asarray(line.get_ydata());mask=(xx>=lim[0])&(xx<=lim[1]);vals.extend(yy[mask])
                if vals:ac.set_ylim(max(0,min(vals)-.02),min(1,max(vals)+.02))
        label=f'{scene}_K{k}_{output}{node or branch}';fig.suptitle(f'{scene}, K={k}: '+('bus '+str(node)+' |V|' if output=='V' else 'from branch '+str(branch)+' '+output if branch else 'total active loss'))
        fig.supxlabel('Empirical density diagnosis; same model bins for GMM/TMM; faint lines: four propagations. Reference: five held-out processes; sparse extreme order statistics.',fontsize=8)
        for ext in ['png','svg']:fig.savefig(dest/(label+'.'+ext),dpi=140,facecolor='white')
        plt.close(fig);csv(pdest/(label+'.csv'),data);index.append(f'<p>{label}</p><a href="figures/{label}.svg"><img width="900" src="figures/{label}.png"></a>')
    (O/'figure_index.html').write_text('<!doctype html><meta charset="utf-8"><title>Typhoon K1/3/6</title>'+''.join(index))
def finish():
    summarize();plots()
    fitrows=pd.read_csv(O/'all_fit_attempts.csv');chosen=pd.read_csv(O/'selected_models.csv')
    assert len(chosen)==24
    assert len(list(O.glob('W*/propagation/*_complete.json')))==96
    for scene in SCENES:
        for k in KS:
            for family in ['GMM','TMM']:
                for rep in range(4):
                    p=O/scene/'propagation'/f'{family}_K{k}_r{rep}'
                    g=pd.read_csv(str(p)+'_global.csv');q=pd.read_csv(str(p)+'_quantiles.csv')
                    assert len(g)==97 and len(q)==776 and (g.scene==scene).all() and (g.K==k).all() and (g.family==family).all()
                    assert g[['output','node','branch']].drop_duplicates().shape[0]==97
    completions=[json.loads(p.read_text()) for p in O.glob('W*/propagation/*_complete.json')]
    cost=dict(input_generation_seconds=0,new_AC_seconds=0,PtoV_seconds=0,AC_reuse=True,
      all_fit_attempt_seconds=float(fitrows.seconds.sum()),propagation_seconds=sum(r['propagation_seconds'] for r in completions),
      error_processing_seconds=sum(r['processing_seconds'] for r in completions),
      current_invocation_wall_seconds=time.perf_counter()-START,current_invocation_CPU_seconds=time.process_time()-CPU,
      money='unknown; billing not exposed',previous_input_AC_cost='existing input package, not counted as new speedup',
      unknown='host scheduling; prior interrupted invocation overhead if any')
    write_json(O/'cost_ledger.json',cost)
    shapes=pd.read_csv(O/'reference_distribution_shapes.csv');advantages=pd.read_csv(O/'priority_local_advantages.csv')
    report='''# 台风构造输入：33节点K1/3/6概率潮流比较\n\n本批只复用既有四组2044快照及标准AC，未新增源荷或AC。训练1314、验证365、参考365，按原完整过程划分。K为全网联合电压模型分量数。\n\n参考365点来自5个保留过程，具有时间关联；.005/.995仅约2个顺序观测。四次65536模型传播只描述模型数值传播波动，不增加实测或独立AC参考，也不是训练重复。\n\n候选按验证电压联合似然选择；常规两组PC balanced，TMM初始df12/200、PX-ECM、max300/tol1e-6；失败和未收敛状态保留。协方差下限1e-7，t scatter与实际协方差区分。没有专项算法核查或选中模型收紧。\n\n'''
    report+='## 实际所选模型\n\n'+chosen[['scene','K','family','dimension','candidate','status','validation_ll','df']].to_csv(index=False)+'\n\n'
    report+='## 优先对象的局部差异\n\n'+advantages.head(16)[['scene','K','output','node','branch','kind','metric','q','GMM_error_mean','TMM_error_mean','GMM_min','GMM_max','TMM_min','TMM_max']].to_csv(index=False)+'\n\n'
    report+='优先对象事先由接入拓扑确定，全部97对象均保存；“超出传播范围”仅指这4次传播的范围不重叠，不代表参考不确定性或统计显著性。单个分位改善不能外推到整个尾段；同K表与双尾CDF分别列出。不利与接近结果均保留，不按全网平均或双尾共同达标否决局部收益。\n\n'
    report+='## 物理解释与限制\n\nW2/W3相同风电，切出/恢复状态混合改变源荷组合；W3仅预设15%负荷下降。源荷尾部条件见AC_tail_source_conditions.csv。风机吸收无功，因此反送/受电变化与Q、电压、损耗不能只用有功符号解释。具体输出非高斯形态需读本对象参考分布和同均值/方差参照，不能由风P零质量或高风速直接推断输出重尾。\n\n电压|V|、线路P/Q和网损各自比较，单节点幅值准确不说明两端联合关系准确。未建立新气象模型或假称Kelmarsh观测台风。共同PCA维数及重构差逐场景保存，维数变化不作为TMM收益。\n\n'
    report+='费用见cost_ledger.json；货币费用unknown。图为Python云端补充诊断，沿现有Matplotlib，AC黑/GMM蓝虚/TMM红实；不冒充MATLAB正式论文图。\n'
    (O/'report_zh.md').write_text(report)
    files=[p for p in O.rglob('*') if p.is_file() and not p.name.endswith('.tmp') and p.name not in ['artifact_manifest.json','progress.json','progress_history.jsonl','run_manifest.json']]
    write_json(O/'artifact_manifest.json',dict(files=[dict(path=str(p.relative_to(O)),bytes=p.stat().st_size,SHA256=sha(p)) for p in files]))
    m=json.loads((O/'run_manifest.json').read_text());m.update(status='complete',selected_models=24,propagations=96,
      fit_attempts=len(fitrows),selected_status_counts=chosen.status.value_counts().to_dict(),finished_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      artifact_manifest_SHA256=sha(O/'artifact_manifest.json'))
    write_json(O/'run_manifest.json',m);event('complete',no_additional_K_or_network=True)
def main():
    protocol()
    for scene in SCENES:prepare(scene)
    for k in KS:
        for scene in SCENES:
            fits(scene,k);evaluate(scene,k)
        summarize();event('K_stage_complete',K=k)
    finish()
if __name__=='__main__':
    try:main()
    except Exception:
        event('failed',error=traceback.format_exc());raise
