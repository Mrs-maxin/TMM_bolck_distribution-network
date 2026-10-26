"""Matched diagnostic PDF plots; fixed objects, units and original method colours."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from easyplot_py import publication_context,save_figure
from independent_block_t_fit import OUT,SRC,geometry
from independent_block_t_numeric import specs


def run():
 g=geometry();ss=specs(g);ref=np.load(SRC/'ac_results.npz')['reference'];focus=['V18','V33','P1','P16','P32','Q17','loss'];figures=OUT/'figures';figures.mkdir(exist_ok=True);audit=[];styles={'AQGMM':('#356D9A','--'),'Standard_TMM':('#98671F','-.'),'Block_TMM':('#AB4B53','-')}
 for name in focus:
  j=next(i for i,s in enumerate(ss) if s['name']==name);s=ss[j];values=ref[:,j];density,edges=np.histogram(values,bins='fd',density=True);pd.DataFrame(dict(left=edges[:-1],right=edges[1:],pdf=density)).to_csv(figures/f'{name}_AC_histogram.csv',index=False)
  unit='p.u.' if s['kind']=='V' else 'MVAr' if s['kind']=='Q' else 'MW';physical=f'Bus {name[1:]} voltage magnitude' if s['kind']=='V' else 'System active loss' if name=='loss' else f"Branch {name[1:]} ({s['from_bus']} to {s['to_bus']})\nsending-end {'active' if s['kind']=='P' else 'reactive'} power"
  for k in [1,2,4]:
   curves={m:pd.read_csv(OUT/'distributions'/f'K{k}'/m/f'{name}.csv') for m in styles};union=np.unique(np.concatenate([t.value.to_numpy() for t in curves.values()]));table=pd.DataFrame(dict(value=union))
   for model,t in curves.items():
    for column in ['pdf','cdf']:table[model+'_'+column]=np.interp(union,t.value,t[column],left=np.nan,right=np.nan)
   csv=figures/f'{name}_K{k}_curves.csv';table.to_csv(csv,index=False)
   for extent in ['full','AC_operating_range','lower','upper']:
    with publication_context(font_family='DejaVu Sans',base_size=10):
     f,ax=plt.subplots(figsize=(100/25.4,76/25.4));ax.stairs(density,edges,fill=True,color='#D8DDE2',edgecolor='#566169',linewidth=.6,label='Independent AC')
     for model,t in curves.items():color,ls=styles[model];ax.plot(t.value,t.pdf,color=color,ls=ls,lw=1.1,label=model.replace('_',' '))
     if extent=='full':limits=(union.min(),union.max())
     elif extent=='AC_operating_range':span=values.max()-values.min();limits=(values.min()-.05*span,values.max()+.05*span)
     else:limits=np.quantile(values,[.01,.05] if extent=='lower' else [.95,.99])
     ax.set_xlim(*limits)
     if extent!='full':
      peaks=[density[(edges[:-1]<=limits[1])&(edges[1:]>=limits[0])].max()];troughs=[0.]
      for t in curves.values():
       yy=t.pdf[(t.value>=limits[0])&(t.value<=limits[1])]
       if len(yy):peaks.append(yy.max());troughs.append(yy.min())
      ax.set_ylim(1.1*min(troughs),1.12*max(peaks))
     ax.set_xlabel(physical+f' ({unit})',fontsize=8 if s['kind'] in ['P','Q'] else 10);ax.set_ylabel(f'PDF (1/{unit})');ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False));ax.legend(fontsize=6.5,frameon=False);f.tight_layout();f.canvas.draw();renderer=f.canvas.get_renderer();box=f.bbox
     texts=[ax.xaxis.label,ax.yaxis.label,ax.get_legend()]+[t for v,t in zip(ax.get_xticks(),ax.get_xticklabels()) if limits[0]<=v<=limits[1]]+[t for v,t in zip(ax.get_yticks(),ax.get_yticklabels()) if ax.get_ylim()[0]<=v<=ax.get_ylim()[1]]
     for artist in texts:
      bb=artist.get_window_extent(renderer);assert bb.x0>=box.x0-1 and bb.x1<=box.x1+1 and bb.y0>=box.y0-1 and bb.y1<=box.y1+1,(name,k,extent,'text outside canvas')
     stem=figures/f'{name}_K{k}_{extent}'
     for ext in ['png','svg']:save_figure(f,stem.with_suffix('.'+ext),dpi=200,overwrite=True)
     audit.append(dict(object=name,K=k,extent=extent,xlim=ax.get_xlim(),ylim=ax.get_ylim(),unit=unit,legend=[t.get_text() for t in ax.get_legend().get_texts()],data=csv.name,data_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),negative_values_clipped=False,smoothing=False,text_within_canvas=True));plt.close(f)
 (OUT/'figure_checks.json').write_text(json.dumps(dict(status='passed',figures=len(audit),checks=audit),indent=2)+'\n');print('PLOTS_COMPLETE',len(audit))

if __name__=='__main__':run()
