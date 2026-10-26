"""Render single-model analytic curves and check visible physical labels."""
from pathlib import Path
import json
import os
os.environ['MPLBACKEND']='Agg'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easyplot_py import publication_context,save_figure
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'result/case33bw/direct_pv8_python_cloud_20261026/analytic_only'


def run():
    records=[]
    for info in sorted(OUT.glob('K*/*/*.json')):
        r=json.loads(info.read_text())
        if r.get('status')!='passed':continue
        t=pd.read_csv(info.with_suffix('.csv'))
        physical=r['physical_quantity'].replace(' sending-end','\nsending-end')
        with publication_context(font_family='DejaVu Sans',base_size=10):
            for quantity,column in [('PDF','pdf'),('CDF','cdf')]:
                fig,ax=plt.subplots(figsize=(100/25.4,76/25.4))
                ax.plot(t.value,t[column],color='#356D9A',lw=1.1,label=f"{r['model']}, K={r['K']}")
                ax.set_xlabel(physical+f" ({r['unit']})",fontsize=8 if r['kind'] in ('P','Q') else 10)
                ax.set_ylabel(f"PDF (1/{r['unit']})" if quantity=='PDF' else 'Cumulative probability')
                ax.legend(fontsize=7,frameon=False)
                fig.tight_layout();fig.canvas.draw();renderer=fig.canvas.get_renderer()
                canvas=fig.bbox
                checks=[]
                xticks=[t for v,t in zip(ax.get_xticks(),ax.get_xticklabels()) if ax.get_xlim()[0]<=v<=ax.get_xlim()[1]]
                yticks=[t for v,t in zip(ax.get_yticks(),ax.get_yticklabels()) if ax.get_ylim()[0]<=v<=ax.get_ylim()[1]]
                for artist in [ax.xaxis.label,ax.yaxis.label,ax.get_legend(),*xticks,*yticks]:
                    if not artist.get_visible():continue
                    box=artist.get_window_extent(renderer)
                    if hasattr(artist,'get_text') and not artist.get_text():continue
                    checks.append(box.x0>=canvas.x0-1 and box.x1<=canvas.x1+1 and box.y0>=canvas.y0-1 and box.y1<=canvas.y1+1)
                assert all(checks),(r['object'],r['model'],quantity,'text outside canvas')
                name=f"{r['object']}_K{r['K']}_{r['model'].replace(' ','_')}_{quantity}"
                for ext in ('png','svg'):save_figure(fig,OUT/'figures'/f'{name}.{ext}',dpi=180,overwrite=True)
                records.append(dict(object=r['object'],model=r['model'],K=r['K'],quantity=quantity,
                    xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),xlim=ax.get_xlim(),ylim=ax.get_ylim(),
                    legend=ax.get_legend().get_texts()[0].get_text(),text_within_canvas=all(checks),
                    curve_points=len(t),data_sha256=__import__('hashlib').sha256(info.with_suffix('.csv').read_bytes()).hexdigest(),
                    smoothing=False,negative_values_clipped=False))
                plt.close(fig)
    (OUT/'figure_check_report.json').write_text(json.dumps(dict(status='passed',single_model_only=True,figures=records),indent=2)+'\n')
    print('Checked',len(records),'single-model figures, PNG and SVG each')


if __name__=='__main__':run()
