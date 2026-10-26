"""Tables and diagnostic figures from the independent IEEE33 heavy-tail study."""
from pathlib import Path
import sys,json,hashlib,zipfile
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[2]
import easyplot_py as ep
import matplotlib.pyplot as plt
import run_direct_heavytail_case33 as run
OUT=run.OUT
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
LABEL={'pareto':'Pareto','lognormal':'对数正态','loglogistic':'对数逻辑斯蒂','mixed':'不同节点采用不同分布'}
MLABEL={'AQGMM':'AQGMM','standard_TMM':'标准TMM','grouped_t_shift':'当前分组t'}
COLORS={'AQGMM':'#568FC3','standard_TMM':'#76AAA0','grouped_t_shift':'#D8898D',
        'pareto':'#568FC3','lognormal':'#76AAA0','loglogistic':'#D8898D'}
STYLES={'AQGMM':'--','standard_TMM':'-.','grouped_t_shift':'-'}

def export(fig,name,data=None):
    ep.export_figure(fig,FIG/name,formats=('png','pdf'),dpi=220,overwrite=True,
        provenance=dict(purpose='研究讨论用诊断图，正式论文图另由MATLAB导出',data=data,
            palette=COLORS,font='Times New Roman + PingFang SC',source='protocol.json',
            confidence='MC尾概率95%逐点Wilson区间；不是同时置信带'))
    plt.close(fig)

def input_plots():
    x=np.linspace(.001,1.,1200);rows=[]
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout='constrained')
    for name,rv in run.LAWS.items():
        axes[0].plot(x,rv.pdf(x),color=COLORS[name],label=LABEL[name],lw=1.5)
        axes[1].semilogy(x,rv.sf(x),color=COLORS[name],label=LABEL[name],lw=1.5)
        rows.append(pd.DataFrame(dict(distribution=name,power_MW=x,PDF_per_MW=rv.pdf(x),upper_probability=rv.sf(x))))
    axes[0].set(xlabel='电源有功注入 / MW',ylabel='概率密度 / (1/MW)',xlim=(0,.7),ylim=(0,None))
    axes[1].set(xlabel='电源有功注入 / MW',ylabel='高于该出力的概率',xlim=(.2,1),ylim=(1e-5,1))
    for ax in axes:ep.format_axes(ax)
    axes[0].legend(frameon=False,fontsize=8)
    pd.concat(rows).to_csv(OUT/'input_distribution_curves.csv',index=False)
    export(fig,'input_PDF_and_upper_probability','input_distribution_curves.csv')

def wilson(p,n=1000000):
    z=norm.ppf(.975);den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return mid-half,mid+half

def comparison_plots(scene,representatives):
    d=OUT/scene
    k=2
    for obj in representatives:
        j=next(i for i in range(97) if run.object_name(i)==obj)
        feature=pd.read_csv(d/'MC_electrical_features.csv').iloc[j]
        side='upper' if feature.extension_upper>=feature.extension_lower else 'lower'
        fig,axes=plt.subplots(1,2,figsize=(7.1,2.65),layout='constrained');curverows=[]
        for model in MLABEL:
            v=np.load(d/f'CDF_{model}_K{k}.npz');x=v['thresholds'][j];cdf=v['CDF'][j];mc=v['MC_CDF'][j]
            order=np.argsort(x,kind='stable')
            if model=='AQGMM':
                axes[0].plot(x[order],mc[order],color='#444444',lw=2,label='MC方法')
            axes[0].plot(x[order],cdf[order],color=COLORS[model],ls=STYLES[model],lw=1.3,label=MLABEL[model])
            sl=slice(0,50) if side=='lower' else slice(50,100)
            xx=x[sl];yy=cdf[sl] if side=='lower' else 1-cdf[sl]
            empirical=mc[sl] if side=='lower' else 1-mc[sl];o=np.argsort(xx)
            if model=='AQGMM':
                lo,hi=wilson(empirical)
                axes[1].plot(xx[o],empirical[o],color='#444444',lw=2,label='MC方法')
                axes[1].fill_between(xx[o],lo[o],hi[o],color='#999999',alpha=.25,lw=0)
            axes[1].plot(xx[o],yy[o],color=COLORS[model],ls=STYLES[model],lw=1.3,label=MLABEL[model])
            curverows.append(pd.DataFrame(dict(model=model,threshold=x,CDF=cdf,MC_CDF=mc)))
        unit='p.u.' if j<32 else 'MW' if j<64 or j==96 else 'Mvar'
        name=f'节点{j+2}电压' if j<32 else f'支路{j-31}首端有功' if j<64 else f'支路{j-63}首端无功' if j<96 else '系统有功网损'
        axes[0].set(xlabel=f'{name} / {unit}',ylabel='累积概率',ylim=(0,1))
        axes[1].set(xlabel=f'{name} / {unit}',ylabel='低于该值的概率' if side=='lower' else '高于该值的概率',yscale='log',ylim=(.0005,.08))
        for ax in axes:ep.format_axes(ax)
        axes[0].legend(frameon=False,fontsize=7)
        stem=f'{scene}_{obj}_K{k}'
        pd.concat(curverows).to_csv(FIG/f'{stem}.csv',index=False)
        export(fig,stem,f'{stem}.csv')

def summarize():
    tables=[];reps=[];features=[]
    for scene in run.SCENES:
        d=OUT/scene
        if not (d/'objects_grouped_t_shift_K2.csv').exists():continue
        if (d/'comparison_complete.json').exists():
            per=pd.read_csv(d/'all_object_errors.csv')
        else:
            parts=[]
            for model in MLABEL:
                for k in run.KS:
                    p=d/f'objects_{model}_K{k}.csv'
                    if p.exists():parts.append(pd.read_csv(p).assign(scene=scene,K=k,model=model))
            per=pd.concat(parts,ignore_index=True)
        tables.append(per)
        f=pd.read_csv(d/'MC_electrical_features.csv');features.append(f.assign(scene=scene))
        ids=json.loads((d/'representatives_locked_before_fit.json').read_text())['objects']
        for obj in ids:
            ff=f[f.object==obj].iloc[0];side='upper' if ff.extension_upper>=ff.extension_lower else 'lower'
            rows=per[(per.object==obj)&(per.K==2)].set_index('model')
            reps.append(dict(scene=scene,object=obj,side=side,extension=float(ff['extension_'+side]),
                **{name:float(rows.loc[name,side+'_tail_pp']) for name in MLABEL},
                GMM_full_CDF_pp=float(rows.loc['AQGMM','full_CDF_max_pp']),
                grouped_full_CDF_pp=float(rows.loc['grouped_t_shift','full_CDF_max_pp'])))
        # Three strongest electrical-object types are fixed by MC, independently of method errors.
        chosen=[]
        for kind in ['V','P','Q']:
            ff=f[f.kind==kind].copy();score=ff[['extension_lower','extension_upper']].max(axis=1)
            chosen.append(ff.loc[score.idxmax(),'object'])
        comparison_plots(scene,chosen)
    if not tables:return
    alltab=pd.concat(tables,ignore_index=True);alltab.to_csv(OUT/'all_cases_object_errors.csv',index=False)
    feat=pd.concat(features,ignore_index=True);feat.to_csv(OUT/'all_cases_MC_features.csv',index=False)
    rp=pd.DataFrame(reps);rp.to_csv(OUT/'representative_K2_table.csv',index=False)
    side_rows=[]
    for side in ['lower','upper']:
        sub=alltab[alltab['extended_'+side]].copy()
        sub['side']=side;sub['tail_error_pp']=sub[side+'_tail_pp'];side_rows.append(sub)
    pd.concat(side_rows).to_csv(OUT/'MC_screened_object_side_errors.csv',index=False)
    # Per-object smaller-K comparisons, never replacing them with all-network medians.
    kvals=[]
    for (scene,obj),block in alltab.groupby(['scene','object']):
        ff=feat[(feat.scene==scene)&(feat.object==obj)].iloc[0]
        for side in ['lower','upper']:
            if not ff['extended_'+side]:continue
            for model in ['standard_TMM','grouped_t_shift']:
                for k in [1,2]:
                    t=block[(block.model==model)&(block.K==k)].iloc[0]
                    g=block[(block.model=='AQGMM')&(block[side+'_tail_pp']<=t[side+'_tail_pp']+.05)&
                            (block.full_CDF_max_pp<=t.full_CDF_max_pp+.5)]
                    kvals.append(dict(scene=scene,object=obj,side=side,model=model,TMM_K=k,
                                      first_matching_GMM_K=int(g.K.min()) if len(g) else '>6'))
    pd.DataFrame(kvals).to_csv(OUT/'per_object_component_comparison.csv',index=False)
    lines=['# IEEE 33节点直接给定重尾注入功率：首轮检查','',
        '节点电压和支路有功的结果随注入分布及位置变化。当前分组t在对数逻辑斯蒂情形的若干电气量上更准，在Pareto情形的若干电气量上更差。该测试尚不支持当前分组t在不同重尾输入下稳定优于AQGMM。',
        '本轮是理论场景数值检查，结果不构成方法在任意重尾输入下均优的理论证明。','',
        '本测试采用原始负荷3.715 MW、2.300 Mvar，保持恒定。8个电源节点为18、17、16、15、14、13、33、22。',
        '每站有功注入完整分布的均值为0.25 MW，标准差为0.0645497 MW；电源无功注入为零。输入无截断、无均值加偏差构造。',
        '各站有功出力的Pearson相关系数设为0.3，通过高斯Copula及Nataf校正给定。相关关系是理论场景假设。',
        'Pareto形状参数5、最小出力0.2 MW；对数正态对数标准差0.254044329、尺度0.242061459 MW；对数逻辑斯蒂形状7.300642012、尺度0.242355580 MW。三者均有有限四阶矩。',
        '混用情形按上述节点顺序依次采用：Pareto、对数正态、对数逻辑斯蒂、Pareto、对数正态、对数逻辑斯蒂、Pareto、对数正态。',
        '', '## 电气量比较', '',
        '每种情形逐场景交流潮流计算100万个注入场景，得到32个节点电压、32条支路首端有功、32条支路首端无功和总有功网损。',
        '8192个独立注入场景用于估计电压分布参数，另外4096个用于选择参数初值。两组电压均由MATLAB快速映射计算。',
        'AQGMM、按矩计算的标准TMM和当前含均值移动的分组t采用相同场景，分量数取1、2、4、6。电气量CDF均由电压二次型解析计算。',
        '下表为K=2时的部分电气量结果。节点和支路先由MC结果确定；没有根据方法优劣选择展示对象。误差单位为百分点。',
        '', '| 分布情形 | 电气量 | 侧别 | 尾部延伸比 | AQGMM | 标准TMM | 当前分组t |',
        '|---|---|---|---:|---:|---:|---:|']
    for _,r in rp.iterrows():
        lines.append(f'| {LABEL[r.scene]} | {r.object} | {"上尾" if r.side=="upper" else "下尾"} | {r.extension:.2f} | {r.AQGMM:.4f} | {r.standard_TMM:.4f} | {r.grouped_t_shift:.4f} |')
    lines+=['','## 误差及结果边界','',
        '尾部概率误差：分别在MC的0.1%至5%下尾和上尾阈值处，计算方法给出的尾概率与MC尾概率之差，再对50个阈值取绝对误差平均值。乘100后以百分点报告。',
        '尾部延伸比：MC的0.5%或99.5%分位值到中位数的距离，除以MC标准差及标准正态99.5%分位值。大于1.15时记为该概率范围内尾部明显延伸，不能据此证明电压或支路功率具有严格渐近重尾。',
        '完整CDF网格最大误差及上下尾误差均对97个电气量逐个保留。all_cases_object_errors.csv包含完整结果，MC_screened_object_side_errors.csv包含MC先行筛选的对象和侧别。CDF检查采用0.1%至99.9%的251个MC分位阈值。',
        '当前分组t内部使用由参数估计及参数选择场景计算的有限强度近似。该近似没有改变输入功率分布，但可能限制无界重尾情形下电气量尾部的表示能力。',
        '本次每种分布仅完成一组独立场景，尚不能判断不同组场景下的排序稳定性。不同方法的差异需要结合MC区间、参数估计和解析积分误差共同判断。',
        '解析概率已与近似电压分布直接计算的电气量频率核对。K=2关键电气量的强度积分从48点加密至96点，最大CDF变化小于0.005个百分点。较大的支路无功误差超过这一数值积分差异。',
        '固定无功的物理检查发现：Pareto情形中，节点18无功注入应为-0.040 Mvar。AQGMM、标准TMM、当前分组t近似电压分布对应的无功注入均方根偏差分别约为1.11、1.32、3.27 kvar，而MC支路17无功标准差约为0.144 kvar。该检查支持固定无功约束在电压联合分布近似中未被严格保持，是该支路概率误差的一个来源。详见fixed_reactive_injection_check_K2.csv。',
        '输入分布为理论假设，不代表具备容量上限的实际光伏出力规律；不把理论测试结果直接解释为实际光伏运行结论。',
        'per_object_component_comparison.csv是逐电气量的分量数探索比较，以尾部误差相差0.05个百分点、CDF网格最大误差相差0.5个百分点作为描述性接近标准，不将其作为预先注册的论文通过条件。',
        '原项目未写入。本独立目录保留源程序快照、参数、场景、MC结果、逐对象误差和诊断图。Python图用于讨论，正式论文图需按项目MATLAB要求导出。','']
    (OUT/'report_zh.md').write_text('\n'.join(lines))
    print(rp.to_string(index=False))

def main():
    with ep.publication_context(font_family='Times New Roman',cjk_family='PingFang SC',base_size=9,
            glyphs='电源有功注入概率密度高于该出力的概率对数正态逻辑斯蒂节点电压支路首端无功累积低值当前分组') as fonts:
        run.dump(OUT/'font_check.json',fonts)
        input_plots();summarize()

if __name__=='__main__':main()
