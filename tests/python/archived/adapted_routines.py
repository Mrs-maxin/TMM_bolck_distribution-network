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
            seed=2026101928+100000*SCENES.index(scene)+1000*k+rep
            event('propagating',scene=scene,K=k,family=r['family'],replicate=rep,seed=seed)
            clock=time.perf_counter();z,w=propagate(m,N,seed,'sobol',randomizer='MT19937')
            state=(z*a['scale'])@a['basis'].T+a['center'];v=voltage(state);out=outputs(v)
            if rep==0:save_npz(dest/(stem+'_saved_outputs.npz'),values=out[:,PRIMARY],weights=w,object_indices=PRIMARY)
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
            bin_counts=np.array([np.histogram(out[:,j],edges[j])[0] for j in range(len(OBJECTS))])
            bin_weight2=np.array([np.histogram(out[:,j],edges[j],weights=w*w)[0] for j in range(len(OBJECTS))])
            save_npz(dest/(stem+'_curves.npz'),bin_counts=bin_counts,bin_weight2=bin_weight2,cdf=np.array(curves),histogram_density=np.array(hh),outside_and_zero_mass=np.array(masses))
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
    gg=pd.concat([pd.read_csv(p) for p in O.glob('*/propagation/*_global.csv')],ignore_index=True)
    qq=pd.concat([pd.read_csv(p) for p in O.glob('*/propagation/*_quantiles.csv')],ignore_index=True)
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
        tables=[pd.read_csv(p) for p in O.glob('*/'+name)]
        if tables:pd.concat(tables,ignore_index=True).to_csv(O/name,index=False)
    priority=ct[ct.priority==True];priority.to_csv(O/'priority_same_K_comparison.csv',index=False)
    wins=priority[(priority.comparison=='TMM better beyond propagation range')&((priority.kind=='global')|(priority.q.isin([.01,.05,.95,.99])))].sort_values('GMM_minus_TMM_error',ascending=False)
    wins.head(30).to_csv(O/'priority_local_advantages.csv',index=False)
    print('PRIORITY_LOCAL_ROWS',wins.head(8).to_csv(index=False),flush=True)