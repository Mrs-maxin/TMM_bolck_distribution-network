"""Diagnostic full-PDF and tail views for any case of run_pv_fixed_forecast.py (not formal paper figures).

Model curves come from the first repeat of the voltage-to-quantity distribution of each selected model,
regenerated with the original seed and checked against the saved outputs. AC, GMM, TMM and block TMM
share the AC-determined bandwidth and grid for each object; windows are the AC 0.5%-5% and 95%-99.5%
quantile ranges of the exact-comparison scenarios.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import importlib.util, sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('run', REPO/'tests/python/run_pv_fixed_forecast.py')
run = importlib.util.module_from_spec(spec); sys.dont_write_bytecode = True; spec.loader.exec_module(run)
FAMILIES = [('GMM', '--', 'tab:blue'), ('TMM', '-', 'tab:red'), ('BTMM', '-.', 'tab:green')]


def density(x, w, grid, h):
    dx = grid[1]-grid[0]
    hist, _ = np.histogram(x, np.r_[grid-dx/2, grid[-1]+dx/2], weights=w)
    return gaussian_filter1d(hist/(w.sum()*dx), h/dx, mode='constant', truncate=5)


def label(j):
    o = run.OBJECTS[j]
    return {'V': f"Node {o['node']} voltage magnitude (p.u.)", 'P': f"Branch {o['branch']} sending-end P (MW)",
            'Q': f"Branch {o['branch']} sending-end Q (MVAr)", 'loss': 'System active power loss (MW)'}[o['output']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=sorted(run.CASES), default='case141')
    parser.add_argument('--scene', default='common_strength')
    parser.add_argument('--out')
    parser.add_argument('--with-K12', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run.setup_case(args.case); run.apply_mode(args.smoke, args.with_K12)
    O = Path(args.out or REPO/run.CONFIG['default_out'].format(scene=args.scene)); D = O/args.scene
    OUT = D/'diagnostic_pdfs'; OUT.mkdir(exist_ok=True); KS = run.KS; N = run.CONFIG['propagation_points']
    targets = pd.read_csv(D/'targets.csv')
    objects = [(int(r.object_index), r.side, r.operating_impact) for _, r in targets.iterrows()] if 'object_index' in targets else []
    events = pd.read_csv(D/'limit_risk_events.csv')
    for _, e in events[events.source.str.contains('median')].drop_duplicates('object_index').iterrows():
        side = 'lower' if 'lowest' in e.source else 'upper'
        objects.append((int(e.object_index), side, e.source.replace(' ', '_')))
    objects.append((run.LOSS, 'both', 'system_loss'))
    seen = set(); objects = [o for o in objects if not (o[0] in seen or seen.add(o[0]))]
    indices = sorted({j for j, _, _ in objects})
    fit = np.load(D/'fitting_inputs.npz')
    selected = pd.concat([pd.read_csv(D/'selected_models.csv'), pd.read_csv(D/'selected_models_block.csv')], ignore_index=True)
    model_outputs, checks = {}, []
    for k in KS:
        for family, _, _ in FAMILIES:
            row = selected[(selected.K == k) & (selected.family == family)].iloc[0]
            m = np.load(D/'candidates'/(row.candidate+'.npz'))
            seed = run.CONFIG['propagation_seed_base']+100000*run.SCENES.index(args.scene)+1000*k
            z, w = run.propagate_any(m, N, seed)
            out = np.concatenate([run.outputs(run.voltage((zz*fit['scale'])@fit['basis'].T+fit['center']))[:, indices]
                                  for zz in np.array_split(z, 8)])
            saved = np.load(D/'propagation'/f'{family}_K{k}_r0_saved_outputs.npz')
            cols = [indices.index(c) for c in saved['object_indices'] if c in indices]
            keep = [i for i, c in enumerate(saved['object_indices']) if c in indices]
            diff = float(np.max(abs(out[:, cols]-saved['values'][:, keep])))
            assert diff < 1e-10 and np.allclose(w, saved['weights']), (family, k, diff)
            checks.append(dict(K=k, family=family, max_difference_to_saved=diff))
            model_outputs[(family, k)] = (out, w)
    pd.DataFrame(checks).to_csv(OUT/'regeneration_check.csv', index=False)
    ref = np.load(D/'reference_outputs.npz')['values'][:, indices]
    rows = []
    for j, side, impact in objects:
        c = indices.index(j); r = ref[:, c]; n = len(r); rs = np.sort(r)
        iqr = np.subtract(*np.quantile(r, [.75, .25])); h = .9*min(r.std(), iqr/1.349)*n**-.2
        lo, hi = r.min()-4*h, r.max()+4*h; grid = np.linspace(lo, hi, 32768); ac = density(r, np.ones(n), grid, h)
        name = run.OBJECTS[j]['output']+str(run.OBJECTS[j]['node'] or run.OBJECTS[j]['branch'] or '')
        fig, axes = plt.subplots(2, len(KS), figsize=(4*len(KS), 6.4), sharex=True, squeeze=False)
        for col, k in enumerate(KS):
            for rr, scale in enumerate(['linear', 'log']):
                ax = axes[rr, col]; ax.plot(grid, ac, 'k-', lw=1.8, label='AC power flow')
                for family, style, color in FAMILIES:
                    x, w = model_outputs[(family, k)]; x = x[:, c]
                    ax.plot(grid, density(x, w, grid, h), style, color=color, lw=1.3, label={'BTMM': 'Block TMM'}.get(family, family))
                    if rr == 0:
                        rows.append(dict(object=label(j), K=k, family=family, outside_display_probability=float(w[(x < lo) | (x > hi)].sum())))
                for q in [.005, .995]:
                    ax.axvline(np.quantile(r, q), color='gray', lw=.6, ls=':')
                if scale == 'log':
                    ax.set_yscale('log'); ax.set_ylim(ac.max()*1e-4, ac.max()*2)
                if rr == 0:
                    ax.set_title(f'K = {k}')
                else:
                    ax.set_xlabel(label(j))
                if col == 0:
                    ax.set_ylabel('Probability density'+(' (log scale)' if scale == 'log' else ''))
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{args.case} {args.scene}: {label(j)}; side: {side} ({impact}); dotted: AC 0.5% and 99.5% quantiles; diagnostic only", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT/f'{name}_PDF_all_K.png', dpi=110); plt.close(fig)
        windows = {'Full': (lo, hi), 'Lower tail (AC 0.5%-5%)': tuple(np.quantile(r, [.005, .05])),
                   'Upper tail (AC 95%-99.5%)': tuple(np.quantile(r, [.95, .995]))}
        for k in KS:
            curves = {f: model_outputs[(f, k)] for f, _, _ in FAMILIES}
            pdfs = {f: density(x[:, c], w, grid, h) for f, (x, w) in curves.items()}
            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            for col, (title, (a, b)) in enumerate(windows.items()):
                m = (grid >= a) & (grid <= b)
                ax = axes[0, col]; ax.plot(grid[m], ac[m], 'k-', lw=1.8, label='AC power flow')
                for family, style, color in FAMILIES:
                    ax.plot(grid[m], pdfs[family][m], style, color=color, lw=1.4, label={'BTMM': 'Block TMM'}.get(family, family))
                top = max(ac[m].max(), *(p[m].max() for p in pdfs.values()))
                ax.set_ylim(0, 1.08*top); ax.set_xlim(a, b); ax.set_title(title); ax.grid(alpha=.3)
                ax = axes[1, col]; ax.step(rs, np.arange(1, n+1)/n, 'k-', lw=1.6, where='post')
                for family, style, color in FAMILIES:
                    x, w = curves[family]; o = np.argsort(x[:, c]); ax.step(x[o, c], np.cumsum(w[o]), style, color=color, lw=1.3, where='post')
                ax.set_xlim(a, b); ax.set_ylim(*{0: (0, 1), 1: (0, .06), 2: (.94, 1.0)}[col]); ax.grid(alpha=.3); ax.set_xlabel(label(j))
            axes[0, 0].set_ylabel('PDF'); axes[1, 0].set_ylabel('CDF'); axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"{args.case} {args.scene}: {label(j)}, K = {k}; side: {side} ({impact})", fontsize=11)
            fig.text(.5, .01, 'first repeat; same AC-determined bandwidth for all curves; diagnostic only, not a formal paper figure',
                     ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .03, 1, .96)); fig.savefig(OUT/f'{name}_K{k}_full_lower_upper.png', dpi=110); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'display_range_outside_probability.csv', index=False)
    print(json.dumps(dict(figures=len(list(OUT.glob('*.png'))), regeneration=checks), indent=1))


if __name__ == '__main__':
    main()
