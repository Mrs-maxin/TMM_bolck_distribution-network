"""Main-line comparison with analytic electrical-quantity distributions (plans/mainline_comparison_20261024.md section 4.1).

AQGMM, standard TMM and the grouped t mixture with mean shift; distributions of the 97 electrical quantities from the
voltage quadratic forms (conditional characteristic function, strength integration, Gil-Pelaez inversion), no voltage
drawing. The grouping rule uses the Jacobian at the fast-mapping anchor (estimation scenario nearest to the mean
injection), so the whole method needs that single AC power flow.
Usage: python mainline_analytic_case33.py S1|S2E1|S2E2|S3
"""
import os, sys, json, time
if len(sys.argv) >= 2:
    os.environ['MAIN_SCENE'] = sys.argv[1]
os.environ['MAIN_SHIFT'] = '1'
sys.argv = sys.argv[:1]
import numpy as np
import pandas as pd
import mainline_comparison_case33 as mc
import analytic_grouped_t_case33 as an

SCENE = mc.SCENE
OUT = mc.REPO/'result/case33bw/mainline_comparison_analytic_20261026'/SCENE
GRID = np.linspace(0, 1, 401)
if os.environ.get('MAIN_SMOKE') == '1':
    OUT = OUT.parent/('smoke_'+SCENE)


def evaluate_analytic(tm, ref):
    rows = []
    for j, c in enumerate(ref):
        r = c['r']; n = len(r); L = len(mc.LEVELS)
        pts = np.r_[r[(mc.LEVELS*n).astype(int)], r[((1-mc.LEVELS)*n).astype(int)], r[np.minimum((GRID*(n-1)).astype(int), n-1)]]
        F, _ = an.cdf_at(tm[j], pts**2 if j < 32 else pts)
        lower = F[:L]-mc.LEVELS; upper = (1-F[L:2*L])-mc.LEVELS
        empirical = np.searchsorted(r, pts[2*L:], side='right')/n
        err = dict(lower=lower, upper=upper); light = 'upper' if c['heavy'] == 'lower' else 'lower'
        rows.append(dict(object=mc.object_name(j), kind=mc.base.OBJECTS[j]['output'], heavy_side=c['heavy'],
                         heavy_tail=float(np.mean(abs(err[c['heavy']]))), heavy_signed=float(np.mean(err[c['heavy']])),
                         light_tail=float(np.mean(abs(err[light]))), light_signed=float(np.mean(err[light])),
                         ks=float(np.max(abs(F[2*L:]-empirical)))))
    return pd.DataFrame(rows)


def main():
    t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); data = mc.load(); rng = np.random.default_rng(2026102450)
    est = np.load(mc.SRC/'estimation_scenarios.npz'); target = est['target']
    anchor = int(np.argmin(((target-target.mean(0))**2).sum(1))); v0 = est['voltage'][anchor]
    split, Q1, Q2 = mc.chk.split(data['train'], data['x'], data['basis'], data['scale'], rng, SCENE, group_target=data['every'], return_q=True, v0=v0)
    caps = mc.ts.strength_cap(data['every'], split['groups']); merged = [np.array(g) for g in split['group_cols']]
    (OUT/'split.json').write_text(json.dumps(dict(split, anchor_scenario=anchor, strength_caps=caps), ensure_ascii=False, indent=1, default=float))
    print(json.dumps(dict(stage='split', groups=split['groups'], r_first_order=split['r_first_order'], leak=split.get('leak_directions')), ensure_ascii=False), flush=True)
    forms = an.quadratic_forms(data); ref = mc.reference(); rows, fits, timing = [], {}, []
    for k in mc.KS_GMM:
        gmm, info = mc.fit_aqgmm(data['x'], data['xv'], k, data['scale']); fits[k] = info
        models = [('AQGMM', 'GMM', gmm)]
        if k in mc.KS_TMM:
            std, _, _ = mc.mm.moment_models(data['x'], gmm, Q1, Q2)
            models += [('标准TMM', 'TMM', std), (mc.PROPOSED, 'SHIFT', mc.sh.shift_grouped_model(data['x'], gmm, Q1, Q2, merged, caps))]
        for name, kind, m in models:
            t1 = time.perf_counter(); per = evaluate_analytic(an.terms(m, kind, forms), ref); sec = time.perf_counter()-t1
            per.to_csv(OUT/f'objects_{name}_K{k}.csv', index=False); timing.append(dict(model=name, K=k, analytic_seconds=sec))
            extra = dict(df=np.round(np.asarray(m['df'], float), 2).tolist()) if kind == 'TMM' else \
                dict(df=[[round(float(v), 2) for v in c['nu']] for c in m['shifted']], shift_scale=[round(float(c['shift_scale']), 3) for c in m['shifted']]) if kind == 'SHIFT' else {}
            rows.append(dict(model=name, K=k, **mc.aggregate(per), analytic_seconds=sec, **extra))
            print(json.dumps(dict(stage='evaluated', model=name, K=k, V_heavy=round(rows[-1]['V_heavy'], 4), P_heavy=round(rows[-1]['P_heavy'], 4),
                                  loss_heavy=round(rows[-1]['loss_heavy'], 4), seconds=round(sec, 1)), ensure_ascii=False), flush=True)
    summary = pd.DataFrame(rows); summary.to_csv(OUT/'summary.csv', index=False)
    (OUT/'claims_check.json').write_text(json.dumps(mc.claims(summary), ensure_ascii=False, indent=1, default=str))
    (OUT/'run_info.json').write_text(json.dumps(dict(scene=SCENE, method='analytic', starts=mc.STARTS, quadratic_check=forms['check'], anchor_scenario=anchor,
        aqgmm_fits=fits, timing=timing, seconds=time.perf_counter()-t0, script_SHA256=mc.base.sha(mc.Path(__file__))), ensure_ascii=False, indent=1, default=float))
    print(json.dumps(dict(stage='DONE', scene=SCENE, seconds=round(time.perf_counter()-t0)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
