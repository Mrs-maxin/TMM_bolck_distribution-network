"""Clarify saved result files and audit comparisons without new PPF/fitting.

Canonical long-table numbers and frozen archives are preserved. Derived wide
tables report numerical ordering per electrical quantity; parameter validity is
reported separately. No all-network mean is used as an accuracy conclusion.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
if ROOT.name=='程序':ROOT=ROOT.parent
import csv,json,hashlib,re
import numpy as np
import pandas as pd

MOVES={
 '设置.csv':'算例设置.csv','出力.png':'注入功率PDF.png',
 '清单.csv':'计算记录/冻结存档SHA256清单.csv','检查.json':'计算记录/最初归档核验.json',
 '结果/CDF.csv':'结果/尾部CDF_RMSE_全部电气量.csv',
 '结果/分位数.csv':'结果/极端分位数误差_全部电气量.csv',
 '结果/尾段.csv':'结果/尾部CDF取值与概率误差_逐阈值.csv',
 '结果/误差.csv':'结果/尾部两类误差_合并查看表.csv',
 '结果/MC分位.csv':'结果/MC极端分位值及95区间.csv',
 '结果/电气量.csv':'结果/MC电气量分布特征.csv',
 '结果/对象.csv':'结果/电气量编号与单位.csv',
 '结果/CDF对比.csv':'计算记录/整理前尾部CDF_RMSE比较表.csv',
 '结果/分位对比.csv':'计算记录/整理前极端分位数比较表.csv',
 '结果/旧误差.csv':'历史结果/尾部CDF_MAE_全部电气量.csv',
 '结果/旧尾部.csv':'历史结果/尾部CDF_MAE_筛选电气量.csv',
 '结果/求解.csv':'计算记录/解析CDF与分位数求解检查.csv',
 '结果/计算.jsonl':'计算记录/分位数解析计算记录.jsonl',
 '结果/计算备份.jsonl':'计算记录/早期分位数计算检查点.jsonl',
 '结果/积分修正.jsonl':'计算记录/频率积分加密记录.jsonl',
 '结果/积分抽查.json':'计算记录/频率积分抽查汇总.json',
 '结果/核验.json':'计算记录/原指标重算核验.json',
 '重点/对象.csv':'尾部筛选/尾部延伸电气量清单.csv',
 '重点/CDF.csv':'尾部筛选/尾部CDF_RMSE_筛选电气量.csv',
 '重点/分位数.csv':'尾部筛选/尾部两类误差_筛选合并查看表.csv',
 '重点/说明.md':'尾部筛选/说明.md','重点/核验.json':'计算记录/原尾部筛选核验.json',
 '复核/参数.csv':'模型参数核查/电压分布_均值协方差三阶矩.csv',
 '复核/强度.csv':'模型参数核查/分组t_随机强度矩.csv',
 '复核/四阶.csv':'模型参数核查/分组t_电压四阶统计量.csv',
 '复核/标准TMM.csv':'模型参数核查/标准TMM_自由度与四阶统计量.csv',
 '复核/分组.csv':'模型参数核查/电源分组及剩余电压方向.csv',
 '复核/说明.md':'模型参数核查/说明.md',
 '根因/支路无功.csv':'诊断记录/根因检查/支路17无功_物理条件比较.csv',
 '根因/无功尾段.csv':'诊断记录/根因检查/支路17无功_逐阈值CDF比较.csv',
 '根因/电压响应抵消.csv':'诊断记录/根因检查/固定无功_电压响应抵消.csv',
 '根因/自由度来源.csv':'诊断记录/根因检查/自由度_保留电压方向检查.csv',
 '根因/节点18分量.csv':'诊断记录/根因检查/节点18电压_分量偏度与四阶矩.csv',
 '根因/自由度对照.csv':'诊断记录/根因检查/节点18电压_自由度改变.csv',
 '根因/节点电压对照.csv':'诊断记录/根因检查/节点18电压_自由度改变后的误差.csv',
 '根因/说明.md':'诊断记录/根因检查/说明.md',
 '根因/功率平衡.json':'诊断记录/根因检查/MC末端无功平衡核验.json',
 '根因/核验.json':'计算记录/原根因检查核验.json',
 '支路/基础.csv':'诊断记录/支路检查/三相单位换算与功率公式.csv',
 '支路/MC核查.csv':'诊断记录/支路检查/独立交流潮流与原MC比较.csv',
 '支路/解析核查.csv':'诊断记录/支路检查/同一电压分布_解析CDF与直接计算比较.csv',
 '支路/基础核验.json':'诊断记录/支路检查/基础功率计算核验.json',
 '支路/设置.json':'诊断记录/支路检查/核查设置.json',
 '重算.py':'程序/重算.py','积分检查.py':'程序/积分检查.py','查看.py':'程序/查看.py',
 '根因.py':'程序/根因.py','支路核查.py':'程序/支路核查.py',
 '支路修正.py':'程序/支路修正_未完成.py','整理结果.py':'程序/整理结果.py',
}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()

def move_files():
    ledger=[]
    for p in (ROOT/'支路').glob('*.json'):
        if p.name not in ['基础核验.json','设置.json']:
            MOVES[str(p.relative_to(ROOT))]='诊断记录/支路检查/未完成解析检查/'+p.name
    for old,new in MOVES.items():
        src=ROOT/old;dst=ROOT/new
        if not src.exists():continue
        assert not dst.exists(),(old,new,'destination already exists')
        before=sha(src);dst.parent.mkdir(parents=True,exist_ok=True);src.rename(dst)
        assert sha(dst)==before
        ledger.append(dict(旧路径=old,新路径=new,移动前SHA256=before,移动后SHA256=before,移动时内容不变=True))
    # Retain earlier interpreter cache away from the reading entry points.
    if (ROOT/'__pycache__').exists():
        (ROOT/'__pycache__').rename(ROOT/'计算记录/旧程序缓存')
    for folder in ['重点','复核','根因','支路']:
        p=ROOT/folder
        if p.exists() and not any(p.iterdir()):p.rmdir()
    target=ROOT/'计算记录/文件改名对照.csv'
    if ledger:pd.DataFrame(ledger).to_csv(target,index=False,encoding='utf-8-sig')

def update_program_paths():
    filename_map={Path(k).name:Path(v).name for k,v in MOVES.items() if k.endswith('.csv') and v.startswith('结果/')}
    filename_map.update({'CDF对比.csv':'尾部CDF_RMSE_三方法对比.csv','分位对比.csv':'极端分位数误差_三方法对比.csv'})
    causal_map={Path(k).name:Path(v).name for k,v in MOVES.items() if k.startswith('根因/') and k.endswith('.csv')}
    branch_map={Path(k).name:Path(v).name for k,v in MOVES.items() if k.startswith('支路/') and k.endswith('.csv')}
    logmap={Path(k).name:str(Path(v).relative_to('计算记录')) for k,v in MOVES.items() if k.startswith('结果/') and v.startswith('计算记录/') and Path(k).suffix in ['.json','.jsonl','.csv']}
    for p in (ROOT/'程序').glob('*.py'):
        if p.name in ['整理结果.py','生成说明.py']:continue
        s=p.read_text()
        s=s.replace('ROOT=Path(__file__).resolve().parent\n','ROOT=Path(__file__).resolve().parents[1]\n')
        s=s.replace("OUT=r.ROOT/'根因'","OUT=r.ROOT/'诊断记录/根因检查'")
        s=s.replace("OUT=r.ROOT/'支路'","OUT=r.ROOT/'诊断记录/支路检查'")
        for old,new in filename_map.items():s=s.replace("/'"+old+"'","/'"+new+"'")
        if p.name=='根因.py':
            for old,new in causal_map.items():s=s.replace("/'"+old+"'","/'"+new+"'")
            s=s.replace("r.ROOT/'清单.csv'","r.ROOT/'计算记录/冻结存档SHA256清单.csv'")
            s=s.replace("r.ROOT/'根因.py'","r.ROOT/'程序/根因.py'")
        if p.name.startswith('支路'):
            for old,new in branch_map.items():s=s.replace("/'"+old+"'","/'"+new+"'")
        for old,new in logmap.items():
            s=s.replace("r.OUT/'"+old+"'","r.ROOT/'计算记录/"+new+"'")
            s=s.replace("OUT/'"+old+"'","ROOT/'计算记录/"+new+"'")
        # The blocked trial remains separate from complete numerical checks.
        if p.name=='支路修正_未完成.py':
            s=s.replace("OUT=r.ROOT/'诊断记录/支路检查'","OUT=r.ROOT/'诊断记录/支路检查/未完成解析检查'")
        p.write_text(s)

def order_text(row,columns):
    order=sorted(columns,key=lambda x:row[x])
    return ' < '.join(order)

def read_metrics():
    d=ROOT/'结果'
    return pd.read_csv(d/'尾部CDF_RMSE_全部电气量.csv'),pd.read_csv(d/'极端分位数误差_全部电气量.csv')

def compare_methods():
    c,q=read_metrics();idx=['情形','K','电气量','类型','尾侧'];models=['AQGMM','标准TMM','分组t']
    focus=pd.read_csv(ROOT/'尾部筛选/尾部延伸电气量清单.csv')
    features=pd.read_csv(ROOT/'结果/电气量编号与单位.csv').rename(columns={'说明':'电气量说明'})
    w=c.pivot(index=idx,columns='方法',values='CDF_RMSE_百分点').reset_index()
    flags=c[c.方法=='AQGMM'][idx+['MC尾部延伸']]
    w=w.merge(flags,on=idx,validate='1:1').merge(features,on=['电气量','类型'],validate='m:1')
    w=w.drop(columns=['单位'])
    w['误差指标']='尾部CDF-RMSE';w['误差单位']='百分点';w['尾概率范围']='0.1%—5%';w['每侧阈值数']=50
    w['误差数值由小到大']=w.apply(lambda row:order_text(row,models),axis=1)
    w['分组t减AQGMM_百分点']=w['分组t']-w['AQGMM'];w['分组t减标准TMM_百分点']=w['分组t']-w['标准TMM']
    w['分组t误差小于AQGMM']=w['分组t']<w['AQGMM'];w['分组t误差小于标准TMM']=w['分组t']<w['标准TMM']
    w['分组t误差三者最小']=w['分组t误差小于AQGMM']&w['分组t误差小于标准TMM']
    w=w.rename(columns={m:m+'_尾部CDF_RMSE_百分点' for m in models})
    w.to_csv(ROOT/'结果/尾部CDF_RMSE_三方法对比.csv',index=False,encoding='utf-8-sig')
    w[w.MC尾部延伸].to_csv(ROOT/'尾部筛选/尾部CDF_RMSE_三方法对比.csv',index=False,encoding='utf-8-sig')
    qi=['情形','K','电气量','类型','尾侧','尾概率','分位水平','MC分位值','单位']
    qw=q.pivot(index=qi,columns='方法',values=['方法分位值','分位误差','绝对分位误差']).reset_index()
    qw.columns=['_'.join(str(v) for v in col if v!='') if isinstance(col,tuple) else col for col in qw.columns]
    qw=qw.merge(flags,on=idx,validate='m:1').merge(features.drop(columns=['单位']),on=['电气量','类型'],validate='m:1')
    qw['误差指标']='极端分位数绝对误差'
    qw['绝对误差数值由小到大']=qw.apply(lambda row:' < '.join(sorted(models,key=lambda m:row['绝对分位误差_'+m])),axis=1)
    qw['分组t绝对误差小于AQGMM']=qw['绝对分位误差_分组t']<qw['绝对分位误差_AQGMM']
    qw['分组t绝对误差小于标准TMM']=qw['绝对分位误差_分组t']<qw['绝对分位误差_标准TMM']
    qw['分组t绝对误差三者最小']=qw['分组t绝对误差小于AQGMM']&qw['分组t绝对误差小于标准TMM']
    qw.to_csv(ROOT/'结果/极端分位数误差_三方法对比.csv',index=False,encoding='utf-8-sig')
    qw[qw.MC尾部延伸].to_csv(ROOT/'尾部筛选/极端分位数误差_三方法对比.csv',index=False,encoding='utf-8-sig')
    qfocus=q.merge(focus,on=['情形','电气量','类型','尾侧'],validate='m:1')
    qfocus.to_csv(ROOT/'尾部筛选/极端分位数误差_筛选电气量.csv',index=False,encoding='utf-8-sig')
    counts=[]
    for k in [1,2,4,6]:
        z=w[(w.K==k)&w.MC尾部延伸]
        counts.append(dict(指标='尾部CDF-RMSE',K=k,比较项数=len(z),
            分组t误差小于AQGMM项数=int(z['分组t误差小于AQGMM'].sum()),
            分组t误差小于标准TMM项数=int(z['分组t误差小于标准TMM'].sum()),
            分组t误差三者最小项数=int(z['分组t误差三者最小'].sum()),
            每项含义='注入情形×电气量×尾侧；只含MC尾部延伸标记'))
    pd.DataFrame(counts).to_csv(ROOT/'尾部筛选/尾部CDF_RMSE比较项数_按K.csv',index=False,encoding='utf-8-sig')
    # Quantiles are counted separately by level; physical error units are never averaged.
    qc=[]
    for (k,p),z in qw[qw.MC尾部延伸].groupby(['K','分位水平']):
        qc.append(dict(K=k,分位水平=p,比较项数=len(z),
            分组t绝对误差小于AQGMM项数=int(z['分组t绝对误差小于AQGMM'].sum()),
            分组t绝对误差小于标准TMM项数=int(z['分组t绝对误差小于标准TMM'].sum()),
            分组t绝对误差三者最小项数=int(z['分组t绝对误差三者最小'].sum())))
    pd.DataFrame(qc).to_csv(ROOT/'尾部筛选/极端分位误差比较项数_按K及分位水平.csv',index=False,encoding='utf-8-sig')
    return c,q,w,qw

def audit(c,q):
    keys=['情形','方法','K','电气量','尾侧'];points=pd.read_csv(ROOT/'结果/尾部CDF取值与概率误差_逐阈值.csv')
    assert len(c)==9312 and len(q)==27936 and len(points)==465600
    assert not c.duplicated(keys).any() and not q.duplicated(keys+['分位水平']).any()
    points['squared']=(points['方法CDF']-points['MC_CDF'])**2
    grouped=points.groupby(keys,sort=False).agg(mse=('squared','mean'),n=('squared','size')).reset_index()
    check=c.merge(grouped,on=keys,validate='1:1');assert (check.n==50).all()
    max_rmse=float(abs(np.sqrt(check.mse)-check.CDF_RMSE).max())
    max_pp=float(abs(100*check.CDF_RMSE-check.CDF_RMSE_百分点).max())
    errors=q['方法分位值']-q['MC分位值'];max_signed=float(abs(errors-q['分位误差']).max());max_absolute=float(abs(abs(errors)-q['绝对分位误差']).max())
    assert max_rmse<1e-12 and max_pp<1e-10 and max_signed<1e-8 and max_absolute<1e-8
    signed=np.where(points.尾侧=='下尾',points.方法CDF-points.MC_CDF,points.MC_CDF-points.方法CDF)
    max_tail=float(abs(signed*100-points['尾概率偏差_百分点']).max());assert max_tail<1e-10
    focus=pd.read_csv(ROOT/'尾部筛选/尾部延伸电气量清单.csv')
    fc=pd.read_csv(ROOT/'尾部筛选/尾部CDF_RMSE_筛选电气量.csv')
    joined=fc.merge(c,on=keys,suffixes=('_focus','_all'),validate='1:1')
    assert len(focus)==373 and len(fc)==4476 and len(joined)==4476
    assert np.max(abs(joined.CDF_RMSE_百分点_focus-joined.CDF_RMSE_百分点_all))<1e-10
    merged=pd.read_csv(ROOT/'结果/尾部两类误差_合并查看表.csv')
    assert len(merged)==27936
    mixed=merged.merge(q,on=keys+['分位水平'],suffixes=('_merged','_q'),validate='1:1')
    assert np.max(abs(mixed.绝对分位误差_merged-mixed.绝对分位误差_q))<1e-8
    # Frozen archive verification is separate from transformed comparison tables.
    manifest=pd.read_csv(ROOT/'计算记录/冻结存档SHA256清单.csv');bad=[]
    for _,x in manifest.iterrows():
        if sha(ROOT/x['path'])!=x.SHA256:bad.append(x['path'])
    assert not bad,bad
    result=dict(CDF_table_is='tail_CDF_RMSE_not_global_CDF_or_CDF_curve',tail_probability_range=[.001,.05],thresholds_per_side=50,
        tail_CDF_RMSE_rows=len(c),extreme_quantile_rows=len(q),tail_curve_rows=len(points),
        max_recomputed_RMSE_probability_difference=max_rmse,max_probability_to_percentage_point_difference=max_pp,
        max_quantile_signed_error_identity_difference=max_signed,max_quantile_absolute_error_identity_difference=max_absolute,
        max_tail_probability_error_identity_difference_pp=max_tail,
        MC_tail_extension_objects=373,MC_tail_extension_CDF_rows=4476,
        frozen_archive_files=len(manifest),frozen_archive_hash_mismatches=bad,
        new_injection_or_AC_or_model_fit=False,new_error_metrics=False,
        note='File organization and per-object comparison only; unfinished branch-Q integral trials are excluded.')
    (ROOT/'计算记录/结果整理核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    move_files();update_program_paths();c,q,w,qw=compare_methods();audit(c,q)
