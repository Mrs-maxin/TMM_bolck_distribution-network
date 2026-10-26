"""Analytical limit-violation probabilities for case 1 on the locked limit events.

Reuses the quadratic forms, strength quadrature and Gil-Pelaez inversion of
tests/analytic_limit_risk_pv_fixed_forecast.py (plan section 14 of the fixed-forecast PV study).
Families: GMM, TMM, BTMM, SBTMM, SGMM at every fitted K. Each analytic value is checked against the
weighted propagation of the same model.
"""
import os, sys, json
from pathlib import Path
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.setdefault('PV_RISK_DIR', str(REPO/'result/case33bw/pv_nrel_case33_20261022')))
os.environ.setdefault('PV_RISK_SCENE', 'nrel_ha4_40_60')
import numpy as np
import pandas as pd
import analytic_limit_risk_pv_fixed_forecast as risk

D = risk.D
FAMILIES = ['GMM', 'TMM', 'BTMM', 'SBTMM', 'SGMM']
PROPAGATION_POINTS = 65536
PROPAGATION_SEED = 2026102241


def main():
    events = json.loads((D/'limit_events_locked.json').read_text())['events']
    selected = pd.concat([pd.read_csv(D/f) for f in ['selected_models.csv', 'selected_models_block.csv', 'selected_models_shared.csv']],
                         ignore_index=True)
    ref = risk.ac_quantities(np.load(D/'reference_AC.npz')['voltage'])
    def threshold(e):
        return e['value']**2 if e['object_index'] < 32 else e['value']
    rows = [dict(family='exact', K='AC', **{e['name']: float(np.mean(ref[:, e['object_index']] < threshold(e)) if e['op'] == '<'
                                                          else np.mean(ref[:, e['object_index']] > threshold(e))) for e in events})]
    checks = []
    for _, r in selected[selected.family.isin(FAMILIES)].sort_values(['family', 'K']).iterrows():
        cache = D/'analytic_cache'/f'{r.family}_K{r.K}.json'
        if cache.exists():
            saved = json.loads(cache.read_text()); rows.append(saved['row']); checks += saved['checks']; continue
        family = 'BTMM' if r.family in ['SBTMM', 'SGMM'] else r.family
        model = dict(np.load(D/'candidates'/(r.candidate+'.npz')))
        z, wz = risk.run.propagate_any(np.load(D/'candidates'/(r.candidate+'.npz')), PROPAGATION_POINTS, PROPAGATION_SEED+int(r.K))
        xq = risk.quantities(risk.center+z@risk.T.T)
        row = dict(family=r.family, K=int(r.K)); new = []
        for q in sorted({e['object_index'] for e in events}):
            items = [e for e in events if e['object_index'] == q]
            F, info = risk.cdf_at(risk.component_terms(model, family, q), [threshold(e) for e in items])
            for e, f in zip(items, F):
                row[e['name']] = float(f if e['op'] == '<' else 1-f)
                numeric = float(wz[xq[:, q] < threshold(e)].sum() if e['op'] == '<' else wz[xq[:, q] > threshold(e)].sum())
                new.append(dict(family=r.family, K=int(r.K), event=e['name'], analytic=row[e['name']], numerical_integration=numeric,
                                difference=row[e['name']]-numeric, tail_cf=info['tail_cf']))
        rows.append(row); checks += new; print('done', r.family, r.K, flush=True)
        cache.parent.mkdir(exist_ok=True); cache.write_text(json.dumps(dict(row=row, checks=new)))
    pd.DataFrame(rows).to_csv(D/'analytic_limit_risk.csv', index=False)
    check = pd.DataFrame(checks); check.to_csv(D/'analytic_limit_risk_check.csv', index=False)
    json.dump(dict(quadratic_reconstruction_max_error=risk.quadratic_check, strength_nodes=risk.STRENGTH_NODES,
                   max_abs_difference_analytic_vs_numerical=float(check.difference.abs().max()),
                   max_tail_cf=float(check.tail_cf.max()), events=[e['name'] for e in events]),
              open(D/'analytic_limit_risk_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
