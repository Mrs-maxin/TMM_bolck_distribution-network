"""MC-only tail screening across all voltage/branch-power quantities."""
import recompute as r
import numpy as np
import pandas as pd
from scipy import stats
import json,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
OUT=r.ROOT/'tail_distribution'
P=np.unique(np.r_[0.,1.,np.linspace(.0001,.9999,201),.0001,.0005,.001,.005,.01,.025,.05,.25,.5,.75,.95,.975,.99,.995,.999,.9995,.9999])

def pwm_shape(excess):
    e=np.asarray(excess);n=len(e);b0=float(e.mean());b1=float(np.arange(n)@e/(n*(n-1)));l2=2*b1-b0
    return 2-b0/l2,b0*(b0/l2-1)

def fixed_threshold_interval(excess,rng,repeats=199):
    n=len(excess);count=rng.multinomial(n,np.full(n,1/n),size=repeats)
    before=np.cumsum(count,axis=1)-count
    b0=count@excess/n;b1=(count*(2*before+count-1))@excess/(2*n*(n-1))
    l2=2*b1-b0;xi=2-b0/l2
    return np.quantile(xi,[.025,.975])

def scene_features(scene):
    mc=np.load(r.OLD/scene/'mc_outputs.npy',mmap_mode='r');rows=[];grid=[];mom=[]
    for j in range(97):
        x=np.sort(np.array(mc[:,j]));q=np.quantile(x,P);obj,kind,unit,scale=r.object_info(j)
        lookup=lambda p:float(np.interp(p,P,q))
        mean=float(np.mean(x));var=float(np.var(x));med=lookup(.5);iqr=lookup(.75)-lookup(.25)
        mom.append(dict(scene=scene,j=j,object=obj,kind=kind,unit=unit,scale=scale,mean=mean,variance=var,iqr=iqr,
            skew=float(stats.skew(x)),excess_kurtosis=float(stats.kurtosis(x))))
        grid.append(dict(j=j,probabilities=P.tolist(),quantiles=q.tolist()))
        for sign,side in [(1,'上尾'),(-1,'下尾')]:
            z=x if sign==1 else -x[::-1]
            central=(lookup(.75)-med) if sign==1 else (med-lookup(.25))
            ratio=[]
            for tail in [.01,.005,.001]:
                distance=(lookup(1-tail)-med) if sign==1 else (med-lookup(tail))
                ratio.append(distance/central/(stats.norm.ppf(1-tail)/stats.norm.ppf(.75)))
            xis=[];thresholds=[];sizes=[]
            for prob in [.95,.975,.99,.995]:
                threshold=float(np.quantile(z,prob));e=z[z>threshold]-threshold
                xi,beta=pwm_shape(e);xis.append(float(xi));thresholds.append(threshold);sizes.append(len(e))
            lo,hi=fixed_threshold_interval(e,np.random.default_rng(2026102600+100*j+(1 if sign==1 else 2)))
            strong=ratio[-1]>=1.25 and min(xis[1:])>.05 and lo>0
            if strong:interpretation='当前尾段重尾特征较强'
            elif ratio[-1]>=1.25:interpretation='尾部延伸明显，重尾形状证据不足'
            else:interpretation='当前尾段延伸较弱'
            row=dict(情形=r.CN[scene],电气量=obj,类型=kind,尾侧=side,MC均值=mean*scale,MC方差=var*scale**2,
                MC中位值=med*scale,MC四分位距=iqr*scale,单位=unit,
                MC偏度=mom[-1]['skew'],MC超额峰度=mom[-1]['excess_kurtosis'],
                高斯标准化尾部延伸比_尾概率1百分比=ratio[0],高斯标准化尾部延伸比_尾概率0_5百分比=ratio[1],高斯标准化尾部延伸比_尾概率0_1百分比=ratio[2],
                GPD形状_尾概率5百分比阈值=xis[0],GPD形状_尾概率2_5百分比阈值=xis[1],GPD形状_尾概率1百分比阈值=xis[2],GPD形状_尾概率0_5百分比阈值=xis[3],
                GPD形状_尾概率0_5百分比阈值区间下限=lo,GPD形状_尾概率0_5百分比阈值区间上限=hi,
                GPD形状_四阈值最大差=max(xis)-min(xis),尾部超额场景数_尾概率0_5百分比阈值=sizes[-1],尾部特征解释=interpretation,
                判别范围='MC有限尾段；不等于渐近重尾证明')
            rows.append(row)
        if j%16==0:print(scene,'MC quantities',j+1,'/97',flush=True)
    r.write_json(OUT/f'{scene}_MC.json',dict(tails=rows,quantiles=grid,moments=mom))
    return scene

def collect():
    items=[json.loads((OUT/f'{s}_MC.json').read_text()) for s in r.SCENES]
    frame=pd.DataFrame([row for x in items for row in x['tails']]);assert len(frame)==776
    quantiles={}
    for scene,item in zip(r.SCENES,items):
        for entry,mom in zip(item['quantiles'],item['moments']):
            quantiles[(r.CN[scene],mom['object'])]=(entry,mom['scale'])
    for prob,label in [(.001,'MC_0_1百分位值'),(.05,'MC_5百分位值'),(.95,'MC_95百分位值'),(.999,'MC_99_9百分位值')]:
        frame[label]=[float(np.interp(prob,quantiles[(row.情形,row.电气量)][0]['probabilities'],quantiles[(row.情形,row.电气量)][0]['quantiles']))*quantiles[(row.情形,row.电气量)][1] for row in frame.itertuples()]
    frame['MC所选尾段跨度_尾概率0_1至5百分比']=np.where(frame.尾侧=='上尾',frame.MC_99_9百分位值-frame.MC_95百分位值,frame.MC_5百分位值-frame.MC_0_1百分位值)
    descriptions=pd.read_csv(r.OUT/'quantity_ids_units.csv')[['电气量','说明']].rename(columns={'说明':'电气量说明'})
    frame=frame.merge(descriptions,on='电气量',validate='m:1')
    frame.to_csv(OUT/'mc_tail_features_all_quantities.csv',index=False,encoding='utf-8-sig')
    main=frame[frame.类型.isin(['V','P','Q'])].copy();assert len(main)==768
    chosen=[];refs=[]
    for (scene,kind),part in main.groupby(['情形','类型'],sort=False):
        best=part.sort_values(['高斯标准化尾部延伸比_尾概率0_1百分比','电气量','尾侧'],ascending=[False,True,True]).drop_duplicates('电气量')
        top=best.head(3).copy();top['图中身份']='按MC尾部延伸比选出的重点电气量';chosen.append(top)
        # The comparison is closest to Gaussian tail extension, never chosen by model error.
        ref=best.assign(distance=abs(best['高斯标准化尾部延伸比_尾概率0_1百分比']-1)).sort_values(['distance','电气量']).head(1).drop(columns='distance')
        ref['图中身份']='尾部延伸较弱的比较电气量';refs.append(ref)
    selected=pd.concat(chosen,ignore_index=True);references=pd.concat(refs,ignore_index=True)
    assert len(selected)==36 and len(references)==12
    selected.to_csv(OUT/'focus_quantities_mc_selection.csv',index=False,encoding='utf-8-sig')
    references.to_csv(OUT/'weak_tail_reference_quantities.csv',index=False,encoding='utf-8-sig')
    raw_moments=pd.DataFrame([row for x in items for row in x['moments']])
    raw_moments['原始电气量单位']=raw_moments['kind'].map({'V':'p.u.','P':'MW','Q':'Mvar','loss':'MW'})
    raw_moments['方差原单位']=raw_moments['原始电气量单位']+'²'
    raw_moments['scene']=raw_moments['scene'].map(r.CN)
    raw_moments=raw_moments.rename(columns={'scene':'情形','j':'电气量列号_从0开始','object':'电气量','kind':'类型',
        'unit':'汇报单位','scale':'原单位至汇报单位系数','mean':'MC均值_原单位','variance':'MC方差_原单位平方',
        'iqr':'MC四分位距_原单位','skew':'MC偏度','excess_kurtosis':'MC超额峰度'})
    raw_moments.to_csv(OUT/'mc_full_mean_variance_raw_units.csv',index=False,encoding='utf-8-sig')
    print(selected[['情形','电气量','尾侧','高斯标准化尾部延伸比_尾概率0_1百分比','GPD形状_尾概率0_5百分比阈值','尾部特征解释']].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    tasks=[s for s in r.SCENES if not (OUT/f'{s}_MC.json').exists()]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=get_context('spawn')) as pool:
        for result in as_completed([pool.submit(scene_features,s) for s in tasks]):print(result.result(),flush=True)
    collect()
