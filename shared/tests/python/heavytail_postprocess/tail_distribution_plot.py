"""MC tail comparisons and shared-bin analytic PDF figures, all K retained."""
import recompute as r
import sys,json,argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
sys.path.insert(0,str(Path.home()/'.codex/skills/easyplot/scripts'))
from easyplot_py import publication_context,make_scientific_plate,format_axes,export_figure

OUT=r.ROOT/'tail_distribution';FIG=OUT/'figures';CACHE=OUT/'computation_records'
PALETTE={'MC':'#707983','AQGMM':'#568FC3','标准TMM':'#BB9491','分组t':'#F59092'}
STYLE={'AQGMM':'-','标准TMM':'--','分组t':'-.'}
records=[]

def export(fig,stem,metadata):
    export_figure(fig,stem,formats=('png','pdf'),dpi=300,overwrite=True,
        provenance=dict(palette=PALETTE,source='frozen MC and saved analytic voltage laws',purpose='independent-copy research figures',**metadata))
    plt.close(fig)
    records.append(dict(**metadata,PNG=str(stem.with_suffix('.png').relative_to(OUT)),PDF=str(stem.with_suffix('.pdf').relative_to(OUT))))

def overview():
    full=pd.read_csv(OUT/'mc_tail_features_all_quantities.csv')
    selected=pd.read_csv(OUT/'focus_quantities_mc_selection.csv')
    refs=pd.read_csv(OUT/'weak_tail_reference_quantities.csv')
    for scene in r.SCENES:
        cn=r.CN[scene];raw=np.load(r.OLD/scene/'mc_outputs.npy',mmap_mode='r')
        for kind,label in [('V','节点电压'),('P','支路有功'),('Q','支路无功')]:
            z=full[(full.情形==cn)&(full.类型==kind)].copy()
            z['number']=z.电气量.str[1:].astype(int)
            fig,axes=make_scientific_plate(2,1,180,125,dpi=300)
            for side,color,marker in [('下尾','#568FC3','o'),('上尾','#BB9491','s')]:
                a=z[z.尾侧==side].sort_values('number')
                axes[0,0].plot(a.number,a.高斯标准化尾部延伸比_尾概率0_1百分比,color=color,marker=marker,markersize=2.2,linewidth=.9,label=side)
                axes[1,0].errorbar(a.number,a.GPD形状_尾概率0_5百分比阈值,
                    yerr=np.array([a.GPD形状_尾概率0_5百分比阈值-a.GPD形状_尾概率0_5百分比阈值区间下限,a.GPD形状_尾概率0_5百分比阈值区间上限-a.GPD形状_尾概率0_5百分比阈值]),
                    color=color,marker=marker,markersize=2,linewidth=.6,capsize=1.3,label=side)
            axes[0,0].axhline(1,color='#b7bdc3',linestyle=':',linewidth=.8)
            axes[1,0].axhline(0,color='#b7bdc3',linestyle=':',linewidth=.8)
            axes[0,0].set_ylabel('0.1%尾部延伸比\n（高斯同尺度＝1）')
            axes[1,0].set_ylabel('GPD形状参数\n（尾概率0.5%阈值）')
            for ax in axes.ravel():
                ax.set_xlabel('节点编号' if kind=='V' else '支路编号');format_axes(ax);ax.legend(loc='best',frameon=False)
                ax.set_xlim(z.number.min()-.5,z.number.max()+.5)
            export(fig,FIG/scene/f'{kind}_all_quantities_tail_compare',dict(情形=cn,电气量=label,图式='全部32项尾部特征',范围='上下尾',单位='无量纲'))
            chosen=pd.concat([selected[(selected.情形==cn)&(selected.类型==kind)],refs[(refs.情形==cn)&(refs.类型==kind)]],ignore_index=True).drop_duplicates('电气量')
            curves=[]
            for row in chosen.itertuples():
                j=next(j for j in range(97) if r.object_info(j)[0]==row.电气量)
                x=np.sort(np.array(raw[:,j]));q25,q50,q75=np.quantile(x,[.25,.5,.75]);sgn=1 if row.尾侧=='上尾' else -1
                scale=((q75-q50) if sgn==1 else (q50-q25))/stats.norm.ppf(.75)
                zvalues=sgn*(x-q50)/scale
                if sgn==-1:zvalues=zvalues[::-1]
                curves.append((row.电气量,row.图中身份,zvalues))
            colors=['#568FC3','#BB9491','#F59092','#707983'];styles=['-','--','-.',':']
            for tail in [False,True]:
                fig,axes=make_scientific_plate(1,1,180,98,dpi=300);ax=axes[0,0]
                for index,(obj,role,values) in enumerate(curves):
                    probabilities=np.sort(1-np.geomspace(.0001,.05,80)) if tail else np.linspace(.0001,.9999,251)
                    edges=np.unique(np.quantile(values,probabilities));F=np.searchsorted(values,edges,side='right')/len(values)
                    density=np.diff(F)/np.diff(edges);mid=(edges[:-1]+edges[1:])/2
                    ax.plot(mid,density,color=colors[index],linestyle=styles[index],linewidth=1.15,
                        label=obj+('（比较）' if '较弱' in role else ''))
                if tail:
                    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(left=1)
                    ax.set_xlabel('沿所选尾侧的偏移 / 稳健标准差（对数轴）');ax.set_ylabel('标准化PDF（对数轴）')
                else:
                    xx=np.linspace(-4,8,300);ax.plot(xx,stats.norm.pdf(xx),color='#c5cbd0',linestyle=':',linewidth=1,label='同尺度高斯')
                    ax.set_xlim(-4,8);ax.set_ylim(bottom=0);ax.set_xlabel('沿所选尾侧的偏移 / 稳健标准差');ax.set_ylabel('标准化PDF')
                format_axes(ax);ax.legend(loc='best',frameon=False)
                view='MC标准化尾部PDF比较' if tail else 'MC标准化PDF比较'
                export(fig,FIG/scene/f'{kind}_{view}',dict(情形=cn,电气量=label,图式=view,范围='MC先行前三项及较弱比较项',单位='标准化'))
        print('MC figures',scene,flush=True)

def model_pdf_figures():
    selected=pd.read_csv(OUT/'focus_quantities_mc_selection.csv');density_rows=[];checks=[]
    for scene in r.SCENES:
        data={}
        for k in [1,2,4,6]:
            for model in r.MODELS:
                path=CACHE/f'{scene}_K{k}_{model}_W1.json';assert path.exists(),path
                payload=json.loads(path.read_text())
                for row in payload['pdf']:data[(row['object'],k,model,row['view'])]=row
        for selection in selected[selected.情形==r.CN[scene]].itertuples():
            obj=selection.电气量;kind=selection.类型;side=selection.尾侧
            j=next(j for j in range(97) if r.object_info(j)[0]==obj)
            number=int(obj[1:]);unit=r.object_info(j)[2]
            name=('节点'+str(number)+'电压') if kind=='V' else ('支路'+str(number)+('有功' if kind=='P' else '无功'))
            for view in ['全分布PDF','尾部PDF放大']:
                fig,axes=make_scientific_plate(2,2,180,125,dpi=300);max_density=0;MC_positive=[]
                for ax,k in zip(axes.ravel(),[1,2,4,6]):
                    ref=data[(obj,k,'AQGMM',view)];edges=np.array(ref['edges']);mid=(edges[:-1]+edges[1:])/2
                    MC=np.array(ref['MC_density']);max_density=max(max_density,float(MC.max()));MC_positive.extend(MC[MC>0])
                    ax.stairs(MC,edges,color=PALETTE['MC'],fill=True,alpha=.18,linewidth=.7,label='MC')
                    for model,cn in r.CN.items():
                        if model not in r.MODELS:continue
                        row=data[(obj,k,model,view)];assert np.allclose(row['edges'],edges)
                        assert row['model_negative_mass_intervals']==0,(scene,obj,k,model,view,row['minimum_model_interval_mass'])
                        y=np.array(row['model_density']);max_density=max(max_density,float(y.max()))
                        display=y if view=='全分布PDF' else np.where(y>0,y,np.nan)
                        ax.plot(mid,display,color=PALETTE[cn],linestyle=STYLE[cn],linewidth=1,label=cn)
                        checks.append(dict(情形=r.CN[scene],电气量=obj,K=k,方法=cn,图式=view,
                            MC显示范围内概率=row['MC_probability_in_view'],方法显示范围内概率=row['model_probability_in_view'],
                            非正密度区间数=int((y<=0).sum()),最小区间概率差=row['minimum_model_interval_mass']))
                        for i in range(len(mid)):
                            density_rows.append(dict(情形=r.CN[scene],电气量=obj,K=k,方法=cn,图式=view,尾侧=side,
                                区间左端=edges[i],区间右端=edges[i+1],MC区间平均PDF=MC[i],方法区间平均PDF=y[i],横轴单位=unit,PDF单位=unit+'⁻¹'))
                    ax.set_title(f'K = {k}',loc='left',fontsize=8,pad=4)
                    ax.set_xlabel(name+' / '+unit);ax.set_ylabel('PDF / '+unit+'⁻¹')
                    ax.set_xlim(edges[0],edges[-1]);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
                    format_axes(ax);ax.legend(loc='best',frameon=False)
                for ax in axes.ravel():
                    if view=='尾部PDF放大':
                        ax.set_yscale('log');ax.set_ylim(max(max_density*1e-8,min(MC_positive)/100),max_density*1.5)
                    else:ax.set_ylim(0,max_density*1.1)
                export(fig,FIG/scene/f'{obj}_{view}',dict(情形=r.CN[scene],电气量=obj,图式=view,范围=side if view=='尾部PDF放大' else 'MC0.01%—99.99%显示范围',单位=unit))
            print('PDF figures',scene,obj,flush=True)
    pd.DataFrame(density_rows).to_csv(OUT/'pdf_bin_mean_density.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(OUT/'pdf_display_range_check.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--overview-only',action='store_true');p.add_argument('--pdf-only',action='store_true');a=p.parse_args();FIG.mkdir(exist_ok=True)
    glyphs='节点编号支路电压有功无功上下尾标准化概率分布比较稳健同尺度高斯形状参数阈值'+'K=PDFμσ⁻¹²0.1%'
    with publication_context(base_size=8,font_family='DejaVu Sans',cjk_family='Arial Unicode MS',glyphs=glyphs) as font:
        assert not font['glyph_coverage'].get('missing_glyphs'),font
        if not a.pdf_only:overview()
        if not a.overview_only:model_pdf_figures()
    # Preserve an earlier overview index if this run intentionally writes only one part.
    path=OUT/'figure_index.csv';new=pd.DataFrame(records)
    if path.exists():new=pd.concat([pd.read_csv(path),new],ignore_index=True).drop_duplicates('PNG',keep='last')
    new.to_csv(path,index=False,encoding='utf-8-sig')
    r.write_json(OUT/'plot_font_check.json',font)
    print('figures indexed',len(new),flush=True)
