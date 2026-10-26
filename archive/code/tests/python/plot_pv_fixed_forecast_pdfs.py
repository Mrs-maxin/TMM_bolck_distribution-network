"""Diagnostic full-PDF views for the fixed-forecast PV case (not formal paper figures).

Model curves come from the first propagation repeat of each selected model, regenerated
with the original seed; the regenerated outputs are checked against the saved samples.
AC, GMM and TMM share the AC-determined bandwidth and grid for each object.
"""
import importlib.util, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('run', REPO/'tests/python/run_pv_fixed_forecast_case33.py')
run = importlib.util.module_from_spec(spec); sys.dont_write_bytecode = True; spec.loader.exec_module(run)
O = REPO/'result/case33bw/pv_fixed_forecast_case33_20261021'
SCENE = 'common_strength'; D = O/SCENE; OUT = D/'diagnostic_pdfs'; OUT.mkdir(exist_ok=True)
KS = [1, 3, 6, 12]; N = run.CONFIG['propagation_points']


def object_index(output, node=0, branch=0):
    return next(j for j, o in enumerate(run.OBJECTS) if o['output'] == output and o['node'] == node and o['branch'] == branch)


targets = pd.read_csv(D/'targets.csv')
objects = [(int(r.object_index), r.side, r.operating_impact) for _, r in targets.iterrows()]
objects += [(object_index('V', 18), 'upper', 'overvoltage_side_not_selected'), (run.LOSS, 'both', 'system_loss')]
indices = sorted({j for j, _, _ in objects})

fit = np.load(D/'fitting_inputs.npz')
selected = pd.concat([pd.read_csv(D/'selected_models.csv'), pd.read_csv(D/'selected_models_block.csv')], ignore_index=True)
FAMILIES = [('GMM', '--', 'tab:blue'), ('TMM', '-', 'tab:red'), ('BTMM', '-.', 'tab:green')]
model_outputs, checks = {}, []
for k in KS:
    for family, _, _ in FAMILIES:
        row = selected[(selected.K == k) & (selected.family == family)].iloc[0]
        m = np.load(D/'candidates'/(row.candidate+'.npz'))
        seed = run.CONFIG['propagation_seed_base']+100000*run.SCENES.index(SCENE)+1000*k
        z, w = run.propagate_any(m, N, seed)
        out = run.outputs(run.voltage((z*fit['scale'])@fit['basis'].T+fit['center']))
        saved = np.load(D/'propagation'/f'{family}_K{k}_r0_saved_outputs.npz')
        cols = list(saved['object_indices'])
        diff = float(np.max(abs(out[:, cols]-saved['values']))) if cols else np.nan
        assert diff < 1e-10 and np.allclose(w, saved['weights']), (family, k, diff)
        checks.append(dict(K=k, family=family, max_difference_to_saved=diff))
        model_outputs[(family, k)] = (out[:, indices], w)
pd.DataFrame(checks).to_csv(OUT/'regeneration_check.csv', index=False)
ref = np.load(D/'reference_outputs.npz')['values'][:, indices]


def density(x, w, grid, h):
    dx = grid[1]-grid[0]
    hist, _ = np.histogram(x, np.r_[grid-dx/2, grid[-1]+dx/2], weights=w)
    return gaussian_filter1d(hist/(w.sum()*dx), h/dx, mode='constant', truncate=5)


def label(j):
    o = run.OBJECTS[j]
    return {'V': f"Node {o['node']} voltage magnitude (p.u.)", 'P': f"Branch {o['branch']} sending-end P (MW)",
            'loss': 'System active power loss (MW)'}[o['output']]


rows = []
for j, side, impact in objects:
    c = indices.index(j); r = ref[:, c]; n = len(r)
    iqr = np.subtract(*np.quantile(r, [.75, .25])); h = .9*min(r.std(), iqr/1.349)*n**-.2
    lo, hi = r.min()-4*h, r.max()+4*h; grid = np.linspace(lo, hi, 32768)
    ac = density(r, np.ones(n), grid, h)
    fig, axes = plt.subplots(2, len(KS), figsize=(4*len(KS), 6.4), sharex=True)
    for col, k in enumerate(KS):
        for row, scale in enumerate(['linear', 'log']):
            ax = axes[row, col]; ax.plot(grid, ac, 'k-', lw=1.8, label='AC reference')
            for family, style, color in FAMILIES:
                x, w = model_outputs[(family, k)]; x = x[:, c]
                ax.plot(grid, density(x, w, grid, h), style, color=color, lw=1.3,
                        label={'BTMM': 'Block TMM'}.get(family, family))
                rows.append(dict(object=label(j), K=k, family=family, outside_display_probability=float(w[(x < lo) | (x > hi)].sum())))
            for q in [.005, .995]:
                ax.axvline(np.quantile(r, q), color='gray', lw=.6, ls=':')
            if scale == 'log':
                ax.set_yscale('log'); ax.set_ylim(ac.max()*1e-4, ac.max()*2)
            if row == 0:
                ax.set_title(f'K = {k}')
            if row == 1:
                ax.set_xlabel(label(j))
            if col == 0:
                ax.set_ylabel('Probability density' + (' (log scale)' if scale == 'log' else ''))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{label(j)}; tail of interest: {side} ({impact}); dotted lines: AC 0.5% and 99.5% quantiles; diagnostic only", fontsize=9)
    fig.tight_layout(); name = run.OBJECTS[j]['output']+str(run.OBJECTS[j]['node'] or run.OBJECTS[j]['branch'])
    fig.savefig(OUT/f'{name}_PDF_K1_3_6_12.png', dpi=130); plt.close(fig)
pd.DataFrame(rows).to_csv(OUT/'display_range_outside_probability.csv', index=False)

# Full / lower-tail / upper-tail layout of the earlier cloud diagnostics: PDF row and CDF row,
# tail windows are the AC 0.5%-5% and 95%-99.5% quantile ranges.
TAIL = OUT/'full_and_tails'; TAIL.mkdir(exist_ok=True)
for j, side, impact in objects:
    c = indices.index(j); r = ref[:, c]; n = len(r); rs = np.sort(r)
    iqr = np.subtract(*np.quantile(r, [.75, .25])); h = .9*min(r.std(), iqr/1.349)*n**-.2
    lo, hi = r.min()-4*h, r.max()+4*h; grid = np.linspace(lo, hi, 32768); ac = density(r, np.ones(n), grid, h)
    windows = {'Full': (lo, hi), 'Lower tail (AC 0.5%-5%)': tuple(np.quantile(r, [.005, .05])),
               'Upper tail (AC 95%-99.5%)': tuple(np.quantile(r, [.95, .995]))}
    name = run.OBJECTS[j]['output']+str(run.OBJECTS[j]['node'] or run.OBJECTS[j]['branch'])
    for k in KS:
        curves = {f: model_outputs[(f, k)] for f, _, _ in FAMILIES}
        pdfs = {f: density(x[:, c], w, grid, h) for f, (x, w) in curves.items()}
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for col, (title, (a, b)) in enumerate(windows.items()):
            m = (grid >= a) & (grid <= b)
            ax = axes[0, col]; ax.plot(grid[m], ac[m], 'k-', lw=1.8, label='AC')
            for family, style, color in FAMILIES:
                ax.plot(grid[m], pdfs[family][m], style, color=color, lw=1.4, label={'BTMM': 'Block TMM'}.get(family, family))
            top = max(ac[m].max(), *(p[m].max() for p in pdfs.values()))
            ax.set_ylim(0, 1.08*top); ax.set_xlim(a, b); ax.set_title(title); ax.grid(alpha=.3)
            ax = axes[1, col]; ax.step(rs, np.arange(1, n+1)/n, 'k-', lw=1.6, where='post', label='AC')
            for family, style, color in FAMILIES:
                x, w = curves[family]; o = np.argsort(x[:, c]); xs = x[o, c]; cw = np.cumsum(w[o])
                ax.step(xs, cw, style, color=color, lw=1.3, where='post')
            ax.set_xlim(a, b)
            if col == 1:
                ax.set_ylim(0, .06)
            elif col == 2:
                ax.set_ylim(.94, 1.0)
            else:
                ax.set_ylim(0, 1)
            ax.grid(alpha=.3); ax.set_xlabel(label(j))
        axes[0, 0].set_ylabel('PDF'); axes[1, 0].set_ylabel('CDF'); axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{label(j)}, K = {k}; tail of interest: {side} ({impact})", fontsize=11)
        fig.text(.5, .01, 'rep0; same AC-determined bandwidth for all curves; diagnostic only, not a formal paper figure',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .03, 1, .96)); fig.savefig(TAIL/f'{name}_K{k}_full_lower_upper.png', dpi=130); plt.close(fig)
print(json.dumps(dict(figures=sorted(p.name for p in OUT.glob('*.png')), regeneration=checks), indent=1))
