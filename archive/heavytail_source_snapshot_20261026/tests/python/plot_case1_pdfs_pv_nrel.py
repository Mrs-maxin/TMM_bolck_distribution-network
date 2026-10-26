"""Case 1 diagnostic PDFs: full, lower-tail and upper-tail windows for every method.

AC: kernel density of the 32768 AC load-flow solutions with width h. Proposed method and prior
GMM: characteristic-function inversion of the voltage quadratics with the same smoothing
(tests/plot_analytic_pdfs_pv_fixed_forecast.py). Gao2024: Gaussian mixture of the linearized
quantity, each component widened by h. Williams2012 and Mohammadi2017: derivative of their CDF,
smoothed by h. Not formal paper figures.
"""
import os, sys, json
from pathlib import Path
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault('PV_RISK_DIR', str(REPO/'result/case33bw/pv_nrel_case33_20261022'))
os.environ.setdefault('PV_RISK_SCENE', 'nrel_ha4_40_60')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import analytic_limit_risk_pv_fixed_forecast as risk
import importlib.util
spec = importlib.util.spec_from_file_location('pdfs', REPO/'tests/python/plot_analytic_pdfs_pv_fixed_forecast.py')
src = open(spec.origin).read(); pdf_ns = {}
exec(compile(src[src.index('def pdf_on_grid'):src.index('rows = []')], spec.origin, 'exec'),
     {'np': np, 'leggauss': np.polynomial.legendre.leggauss}, pdf_ns)
pdf_on_grid = pdf_ns['pdf_on_grid']

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']; plt.rcParams['axes.unicode_minus'] = False
D = risk.D; TB = D/'tail_baselines'; OUT = D/'case1_summary'/'diagnostic_pdfs'; OUT.mkdir(parents=True, exist_ok=True)
best = json.loads((D/'compare_summary.json').read_text())['best_K_by_validation']
gao_k = json.loads((TB/'gao_protocol.json').read_text())['best_K_by_validation']
selected = pd.concat([pd.read_csv(D/f) for f in ['selected_models.csv', 'selected_models_shared.csv']])
METHODS = [('SBTMM', f"本文方法 (K={best['SBTMM']})", '-', 'tab:red', 2.0),
           ('GMM', f"前作GMM (K={best['GMM']})", '--', 'tab:blue', 1.5),
           ('Gao2024', f'Gao2024 (K={gao_k})', '-.', 'tab:green', 1.5),
           ('Williams2012', 'Williams2012 最大熵', ':', 'tab:purple', 1.6),
           ('Mohammadi2017', 'Mohammadi2017 鞍点近似', (0, (5, 1, 1, 1)), 'tab:orange', 1.5)]
LABEL = {'undervoltage_side': '下尾：低电压', 'overvoltage_side': '上尾：过电压',
         'reverse_flow_increase': '下尾：反向功率增大', 'reversal_reverse_to_import': '上尾：由反送转为受电'}
ref = np.load(D/'reference_outputs.npz')['values']; ref_q = risk.ac_quantities(np.load(D/'reference_AC.npz')['voltage'])
cdfs = np.load(TB/'analytic_baseline_cdfs.npz'); g = np.load(TB/f'gao_K{gao_k}.npz')
events = json.loads((D/'limit_events_locked.json').read_text())['events']


def main():
    for _, t in pd.read_csv(D/'targets.csv').iterrows():
        j = int(t.object_index); voltage = t.output == 'V'
        name = f'V{int(t.node)}' if voltage else f'P{int(t.branch)}'
        xlabel = f'节点{int(t.node)}电压幅值 (p.u.)' if voltage else f'支路{int(t.branch)}首端有功功率 (MW)'
        r = ref[:, j]; n = len(r); iqr = np.subtract(*np.quantile(r, [.75, .25])); h = .9*min(r.std(), iqr/1.349)*n**-.2
        lo, hi = r.min()-4*h, r.max()+4*h; grid = np.linspace(lo, hi, 2401); dx = grid[1]-grid[0]
        hist, _ = np.histogram(r, np.r_[grid-dx/2, grid[-1]+dx/2]); ac = gaussian_filter1d(hist/(n*dx), h/dx, mode='constant', truncate=5)
        curves = {}
        for fam in ['SBTMM', 'GMM']:
            cache = OUT/f'{name}_{fam}.npy'
            if cache.exists():
                curves[fam] = np.load(cache); continue
            model = dict(np.load(D/'candidates'/(selected[(selected.family == fam) & (selected.K == best[fam])].candidate.iloc[0]+'.npz')))
            terms = risk.component_terms(model, 'BTMM' if fam == 'SBTMM' else 'GMM', j)
            dens = pdf_on_grid(terms, grid**2, 2*np.median(r)*h)*2*grid if voltage else pdf_on_grid(terms, grid, h)
            np.save(cache, dens); curves[fam] = dens
        s = np.sqrt(g['sds'][:, j]**2+h**2)
        curves['Gao2024'] = (g['weights'][:, None]*stats.norm.pdf((grid[None]-g['means'][:, j, None])/s[:, None])/s[:, None]).sum(0)
        rs = np.sort(r); own = np.linspace(rs[0]-3*r.std(), rs[-1]+3*r.std(), 1601)
        for m in ['Williams2012', 'Mohammadi2017']:
            cdf = np.interp(grid, own, cdfs[f'{m}_{j}']); curves[m] = gaussian_filter1d(np.gradient(cdf, dx), h/dx, mode='constant', truncate=5)
        windows = [('全分布', lo, hi), ('下尾放大（交流潮流0.1%—5%分位）', *np.quantile(r, [.001, .05])),
                   ('上尾放大（交流潮流95%—99.9%分位）', *np.quantile(r, [.95, .999]))]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        for ax, (title, a, b) in zip(axes, windows):
            m = (grid >= a) & (grid <= b)
            ax.plot(grid[m], ac[m], 'k-', lw=2.4, alpha=.55, label='交流潮流（32768场景）')
            for key, label, style, color, lw in METHODS:
                ax.plot(grid[m], curves[key][m], linestyle=style, color=color, lw=lw, label=label)
            for e in [e for e in events if e['object_index'] == j]:
                if a <= e['value'] <= b:
                    ax.axvline(e['value'], color='gray', lw=.8, ls=':')
            top = max(ac[m].max(), *(c[m].max() for c in curves.values())); ax.set_ylim(0, 1.08*top); ax.set_xlim(a, b)
            ax.set_title(title, fontsize=10); ax.grid(alpha=.3); ax.set_xlabel(xlabel)
        axes[0].set_ylabel('概率密度'); axes[0].legend(fontsize=8)
        fig.suptitle(f'{xlabel}；关注：{LABEL.get(t.operating_impact, t.side)}；灰色虚线为锁定越限限值', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(OUT/f'{name}_case1_pdf.png', dpi=130); plt.close(fig)
        print('done', name, flush=True)


if __name__ == '__main__':
    main()
