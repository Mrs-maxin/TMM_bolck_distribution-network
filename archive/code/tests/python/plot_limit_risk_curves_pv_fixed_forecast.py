"""Diagnostic curves: limit-violation probability versus limit value, computed analytically.

Model curves use the characteristic-function inversion of tests/analytic_limit_risk_pv_fixed_forecast.py;
the AC curve is the violation frequency over the AC power-flow solutions of every source-load scenario,
shown with its 95% Wilson interval. Not formal paper figures.
"""
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('risk', REPO/'tests/python/analytic_limit_risk_pv_fixed_forecast.py')
risk = importlib.util.module_from_spec(spec); spec.loader.exec_module(risk)
D = risk.D; OUT = D/'diagnostic_limit_risk'; OUT.mkdir(exist_ok=True)
KS = [1, 6, 12]
FAMILIES = [('GMM', '--', 'tab:blue', 'GMM'), ('TMM', '-', 'tab:red', 'TMM'), ('BTMM', '-.', 'tab:green', 'Block TMM')]
CURVES = [('V18', 16, np.linspace(1.04, 1.10, 61), 'above', 'Node 18 voltage magnitude limit (p.u.)', 'P(V18 > limit)', lambda v: v**2),
          ('V31', 29, np.linspace(.945, .975, 61), 'below', 'Node 31 voltage magnitude limit (p.u.)', 'P(V31 < limit)', lambda v: v**2),
          ('V33', 31, np.linspace(.945, .975, 61), 'below', 'Node 33 voltage magnitude limit (p.u.)', 'P(V33 < limit)', lambda v: v**2),
          ('P19', 50, np.linspace(-.05, .25, 61), 'above', 'Branch 19 sending-end active power limit (MW)', 'P(P19 > limit)', lambda v: v)]
ref = np.load(D/'reference_AC.npz')['voltage']
ref_q = np.c_[abs(ref[:, 1:])**2, risk.run.evaluate_outputs(ref, risk.run.BASE, risk.run.BUS, risk.run.BRANCH)[:, :32]]
n = len(ref_q); z = 1.959964
rows = []
for name, q, limits, side, xlabel, ylabel, transform in CURVES:
    th = transform(limits)
    ac = np.array([(ref_q[:, q] > t).mean() if side == 'above' else (ref_q[:, q] < t).mean() for t in th])
    centre = (ac+z*z/(2*n))/(1+z*z/n); half = z*np.sqrt(ac*(1-ac)/n+z*z/(4*n*n))/(1+z*z/n)
    fig, axes = plt.subplots(1, len(KS), figsize=(4.6*len(KS), 4.2), sharey=True)
    for ax, k in zip(axes, KS):
        ax.fill_between(limits, np.maximum(centre-half, 1e-7), centre+half, color='0.8', label='AC 95% interval')
        ax.semilogy(limits, np.maximum(ac, 1e-7), 'k-', lw=1.8, label='AC power flow')
        for family, style, color, label in FAMILIES:
            r = risk.selected[(risk.selected.K == k) & (risk.selected.family == family)].iloc[0]
            model = dict(np.load(D/'candidates'/(r.candidate+'.npz')))
            F, _ = risk.cdf_at(risk.component_terms(model, family, q), th)
            p = 1-F if side == 'above' else F
            ax.semilogy(limits, np.clip(p, 1e-7, 1), style, color=color, lw=1.4, label=label)
            rows += [dict(quantity=name, K=k, family=family, limit=float(l), probability=float(v), AC=float(a))
                     for l, v, a in zip(limits, p, ac)]
        ax.set_title(f'K = {k}'); ax.set_xlabel(xlabel); ax.grid(alpha=.3, which='both'); ax.set_ylim(1e-5, 1)
    axes[0].set_ylabel(ylabel); axes[0].legend(fontsize=8, loc='lower left' if side == 'above' else 'lower right')
    fig.suptitle('Limit-violation probability: analytic model values and AC power-flow frequency (diagnostic)', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT/f'{name}_limit_risk_K1_6_12.png', dpi=130); plt.close(fig)
    print('done', name, flush=True)
pd.DataFrame(rows).to_csv(OUT/'limit_risk_curves.csv', index=False)
