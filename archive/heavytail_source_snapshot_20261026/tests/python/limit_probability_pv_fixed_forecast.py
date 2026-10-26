"""Descriptive voltage-limit probabilities for the fixed-forecast PV case (not a pre-registered criterion).

Model outputs are regenerated from the first propagation repeat with the original seed and
checked against the saved samples. Thresholds are the project research limits 0.95/1.05 p.u.
plus deeper levels inside the AC reference support; AC probabilities carry their event counts.
"""
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('run', REPO/'tests/python/run_pv_fixed_forecast_case33.py')
run = importlib.util.module_from_spec(spec); sys.dont_write_bytecode = True; spec.loader.exec_module(run)
O = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO/'result/case33bw/pv_fixed_forecast_case33_20261021'
SCENE = sys.argv[2] if len(sys.argv) > 2 else 'common_strength'
D = O/SCENE; N = run.CONFIG['propagation_points']; KS = [1, 3, 6, 12]
selected = pd.concat([pd.read_csv(D/'selected_models.csv'), pd.read_csv(D/'selected_models_block.csv')], ignore_index=True)
fit = np.load(D/'fitting_inputs.npz'); ref = np.load(D/'reference_outputs.npz')['values']
V = lambda node: node-2
EVENTS = {
    'V29 < 0.960': lambda y: y[:, V(29)] < .960,
    'V31 < 0.955': lambda y: y[:, V(31)] < .955,
    'V31 < 0.950': lambda y: y[:, V(31)] < .950,
    'any node < 0.950': lambda y: np.any(y[:, :32] < .950, axis=1),
    'V18 > 1.070': lambda y: y[:, V(18)] > 1.070,
    'V18 > 1.080': lambda y: y[:, V(18)] > 1.080,
    'V18 > 1.090': lambda y: y[:, V(18)] > 1.090,
    'nodes 13-18 all > 1.070': lambda y: np.all(y[:, [V(b) for b in range(13, 19)]] > 1.070, axis=1),
}
rows = [dict(K='AC', family='reference', **{e: f(ref).mean() for e, f in EVENTS.items()},
             **{e+' count': int(f(ref).sum()) for e, f in EVENTS.items()})]
for k in KS:
    for family in ['GMM', 'TMM', 'BTMM']:
        r = selected[(selected.K == k) & (selected.family == family)].iloc[0]
        m = np.load(D/'candidates'/(r.candidate+'.npz'))
        seed = run.CONFIG['propagation_seed_base']+100000*run.SCENES.index(SCENE)+1000*k
        z, w = run.propagate_any(m, N, seed)
        y = run.outputs(run.voltage((z*fit['scale'])@fit['basis'].T+fit['center']))
        saved = np.load(D/'propagation'/f'{family}_K{k}_r0_saved_outputs.npz')
        assert np.max(abs(y[:, list(saved['object_indices'])]-saved['values'])) < 1e-10
        rows.append(dict(K=k, family=family, **{e: float(w[f(y)].sum()) for e, f in EVENTS.items()}))
table = pd.DataFrame(rows); table.to_csv(D/'limit_probabilities_descriptive.csv', index=False)
pd.set_option('display.width', 250)
print(table[['K', 'family']+list(EVENTS)].to_string(index=False, float_format=lambda v: f'{100*v:.3f}%'))
print(table.loc[0, [e+' count' for e in EVENTS]].to_string())
