"""Tail-quantile-interval W1 and shared-bin PDF from saved analytic CDFs."""
import recompute as r
import numpy as np
import pandas as pd
import json,time,argparse
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
ROOT=r.ROOT;OUT=ROOT/'tail_distribution';CACHE=OUT/'computation_records'

def quantile(kernel,p,j,known):
    def F(x):return kernel.cdf([max(x,0)**2 if j<32 else x])[0]
    q0=known[0];q1=known[-1]
    if p in [.001,.999]:return q0 if p==.001 else q1
    if j<32:q0=max(0.,q0)
    return brentq(lambda x:F(x)-p,q0,q1,xtol=max(abs(q1-q0)*1e-10,1e-13),rtol=1e-12)

def interval_w1(kernel,j,sorted_mc,a,b,known,iqr):
    qa=quantile(kernel,a,j,known);qb=quantile(kernel,b,j,known)
    ra,rb=np.quantile(sorted_mc,[a,b]);edges=np.unique([qa,qb,ra,rb]);n=len(sorted_mc)
    previous=None;error=np.inf;value=None;levels=0
    for nodes in [24,48,96,192,384,768]:
        x,w=leggauss(nodes);points=[];weights=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            if hi<=lo:continue
            points.extend(((hi-lo)*(x+1)/2+lo).tolist());weights.extend(((hi-lo)*w/2).tolist())
        points=np.array(points);weights=np.array(weights)
        F=kernel.cdf(points**2 if j<32 else points)
        MC=np.searchsorted(sorted_mc,points,side='right')/n
        value=float(weights@abs(np.clip(F,a,b)-np.clip(MC,a,b))/(b-a));levels=nodes
        if previous is not None:
            error=abs(value-previous)
            if error<=1e-4*iqr+1e-3*value:break
        previous=value
    return dict(W1=value,quad_difference=error,quad_nodes=levels,
        numerical_pass=error<=1e-4*iqr+1e-3*value,model_a=qa,model_b=qb,MC_a=ra,MC_b=rb)

def model_job(scene,k,name):
    start=time.perf_counter();CACHE.mkdir(exist_ok=True)
    out=CACHE/f'{scene}_K{k}_{name}_W1.json'
    if out.exists():return scene,k,name,'cached'
    d=r.OLD/scene;ns=r.numerical_functions();co=dict(np.load(d/'voltage_coordinates.npz'))
    forms=ns['quadratic_forms'](co);model=r.read_models(d/f'models_K{k}.pkl')[name]
    terms=ns['terms'](model,r.MODELS[name],forms)
    mc=np.load(d/'mc_outputs.npy',mmap_mode='r')
    selected=pd.read_csv(OUT/'focus_quantities_mc_selection.csv');selected=selected[selected.情形==r.CN[scene]].set_index('电气量')
    moments=json.loads((OUT/f'{scene}_MC.json').read_text())['moments']
    records={}
    for line in (ROOT/'computation_records/analytic_quantile_log.jsonl').read_text().splitlines():
        item=json.loads(line)
        if item['scene']==scene and item['K']==k and item['model']==name:records[item['j']]=item
    assert len(records)==97
    rows=[];pdf=[]
    for j in range(97):
        obj,kind,unit,scale=r.object_info(j);raw=np.sort(np.array(mc[:,j]));known=records[j]['quantiles']
        kernel=r.CDFKernel(terms[j],cutoff_multiplier=records[j].get('cutoff_multiplier',1))
        iqr=float(moments[j]['iqr'])
        for side,a,b in [('下尾',.001,.05),('上尾',.95,.999)]:
            value=interval_w1(kernel,j,raw,a,b,known,iqr)
            rows.append(dict(情形=r.CN[scene],方法=r.CN[name],K=k,电气量=obj,类型=kind,尾侧=side,
                概率区间下限=a,概率区间上限=b,尾部分位区间W1=value['W1']*scale,
                W1除以MC四分位距=value['W1']/iqr,单位=unit,
                W1积分加密差=value['quad_difference']*scale,W1积分阶数=value['quad_nodes'],W1数值收敛=value['numerical_pass'],
                方法区间左端=value['model_a']*scale,方法区间右端=value['model_b']*scale,
                MC区间左端=value['MC_a']*scale,MC区间右端=value['MC_b']*scale,模型状态=r.status(scene,k,name)))
        if obj in selected.index:
            side=selected.loc[obj,'尾侧']
            fullp=np.unique(np.r_[np.linspace(.0001,.9999,201),.0005,.001,.005,.01,.025,.05,.95,.975,.99,.995,.999,.9995])
            tail=np.geomspace(.0001,.05,61);tailp=tail if side=='下尾' else np.sort(1-tail)
            for zoom,pp in [('全分布PDF',fullp),('尾部PDF放大',tailp)]:
                edges=np.unique(np.quantile(raw,pp));F=kernel.cdf(edges**2 if j<32 else edges)
                Fmc=np.searchsorted(raw,edges,side='right')/len(raw);delta=np.diff(F)
                tolerance=1e-7
                pdf.append(dict(scene=scene,model=name,K=k,j=j,object=obj,kind=kind,side=side,unit=unit,view=zoom,
                    edges=(edges*scale).tolist(),MC_density=(np.diff(Fmc)/np.diff(edges)/scale).tolist(),
                    model_density=(delta/np.diff(edges)/scale).tolist(),
                    model_negative_mass_intervals=int((delta < -tolerance).sum()),minimum_model_interval_mass=float(delta.min()),
                    MC_probability_in_view=float(Fmc[-1]-Fmc[0]),model_probability_in_view=float(F[-1]-F[0]),
                    normalization='full-distribution mass divided by bin width; tail is not renormalized'))
        if j%24==0:print('W1',scene,k,name,j+1,'/97',flush=True)
    r.write_json(out,dict(rows=rows,pdf=pdf,seconds=time.perf_counter()-start))
    return scene,k,name,time.perf_counter()-start

def collect():
    items=[json.loads((CACHE/f'{s}_K{k}_{name}_W1.json').read_text()) for s in r.SCENES for k in [1,2,4,6] for name in r.MODELS]
    frame=pd.DataFrame([row for item in items for row in item['rows']]);assert len(frame)==9312
    frame.to_csv(ROOT/'results/tail_quantile_w1_all_quantities.csv',index=False,encoding='utf-8-sig')
    keys=['情形','K','电气量','类型','尾侧','概率区间下限','概率区间上限','单位']
    wide=frame.pivot(index=keys,columns='方法',values='尾部分位区间W1').reset_index()
    wide.columns.name=None
    wide.to_csv(ROOT/'results/tail_quantile_w1_method_compare.csv',index=False,encoding='utf-8-sig')
    selected=pd.read_csv(OUT/'focus_quantities_mc_selection.csv')[['情形','电气量','尾侧']]
    wide.merge(selected,on=['情形','电气量','尾侧'],validate='m:1').to_csv(OUT/'focus_quantities_tail_quantile_w1_method_compare.csv',index=False,encoding='utf-8-sig')
    print('W1 collected',len(frame),'notconverged',int((~frame.W1数值收敛).sum()),flush=True)

def refine_failed():
    for path in sorted(CACHE.glob('*_W1.json')):
        item=json.loads(path.read_text());failed=[x for x in item['rows'] if not x['W1数值收敛']]
        if not failed:continue
        prefix=path.stem.split('_K');scene=prefix[0];k=int(prefix[1].split('_')[0]);name=prefix[1].split('_',1)[1].removesuffix('_W1')
        d=r.OLD/scene;ns=r.numerical_functions();co=dict(np.load(d/'voltage_coordinates.npz'));forms=ns['quadratic_forms'](co)
        model=r.read_models(d/f'models_K{k}.pkl')[name];records={}
        for line in (ROOT/'computation_records/analytic_quantile_log.jsonl').read_text().splitlines():
            record=json.loads(line)
            if (record['scene'],record['K'],record['model'])==(scene,k,name):records[record['j']]=record
        mc=np.load(d/'mc_outputs.npy',mmap_mode='r')
        for row in failed:
            j=next(j for j in range(97) if r.object_info(j)[0]==row['电气量']);scale=r.object_info(j)[3]
            tm=ns['terms'](model,r.MODELS[name],{key:value[j:j+1] for key,value in forms.items() if key in ['A','b','c']})[0]
            kernel=r.CDFKernel(tm,cutoff_multiplier=records[j].get('cutoff_multiplier',1));raw=np.sort(np.array(mc[:,j]));iqr=float(np.quantile(raw,.75)-np.quantile(raw,.25))
            value=interval_w1(kernel,j,raw,row['概率区间下限'],row['概率区间上限'],records[j]['quantiles'],iqr)
            row.update(尾部分位区间W1=value['W1']*scale,W1除以MC四分位距=value['W1']/iqr,W1积分加密差=value['quad_difference']*scale,W1积分阶数=value['quad_nodes'],W1数值收敛=value['numerical_pass'])
            print('refined W1',scene,k,name,row['电气量'],row['尾侧'],value['quad_nodes'],value['numerical_pass'],flush=True)
        r.write_json(path,item)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--refine-only',action='store_true');a=p.parse_args();CACHE.mkdir(exist_ok=True)
    if a.refine_only:refine_failed();collect();raise SystemExit
    tasks=[(s,k,name) for s in r.SCENES for k in [1,2,4,6] for name in r.MODELS if not (CACHE/f'{s}_K{k}_{name}_W1.json').exists()]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(model_job,*task) for task in tasks]):print(future.result(),flush=True)
    collect()
