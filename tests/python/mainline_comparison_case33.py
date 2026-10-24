"""Main-line comparison of AQGMM, standard TMM and the grouped t mixture (plans/mainline_comparison_20261024.md).

Usage: python mainline_comparison_case33.py S1|S2E1|S2E2|S3
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import pandas as pd

if len(sys.argv) >= 2:
    os.environ['MAIN_SCENE'] = sys.argv[1]
SCENE = os.environ['MAIN_SCENE']
sys.argv = sys.argv[:1]
import grouped_strength_tmm_case33 as gs
import truncated_strength_case33 as ts
from cloud_numeric_core import initialize, fit, log_density
base, chk, mm = gs.base, gs.chk, gs.mm

REPO = base.REPO
SRC = REPO/'result/case33bw/record_scenarios_20261024'/SCENE
MC = REPO/'cache/record_scenarios_20261024'/f'{SCENE}_mc_outputs.npy'
OUT = REPO/'result/case33bw/mainline_comparison_20261024'/SCENE
KS_GMM, KS_TMM = [1, 2, 3, 4, 6, 8, 12, 16], [1, 2, 3, 4, 6, 8]
STARTS, DRAWS = 6, 262144
LEVELS = np.linspace(.001, .05, 50)
MARGIN = 0.0005
if os.environ.get('MAIN_SMOKE') == '1':
    KS_GMM, KS_TMM, STARTS = [1, 2], [1, 2], 2
    gs.DRAWS = 16384; OUT = OUT.parent/('smoke_'+SCENE)


def coordinates(volt):
    st = np.c_[volt[:, 1:].real, volt[:, 1:].imag]
    return st


def load():
    est, sel = np.load(SRC/'estimation_scenarios.npz'), np.load(SRC/'selection_scenarios.npz')
    st, sv = coordinates(est['voltage']), coordinates(sel['voltage'])
    center = st.mean(0); _, s, v = np.linalg.svd(st-center, full_matrices=False); scale = s/np.sqrt(len(st)-1); basis = v.T
    head = np.load(SRC/'mc_inputs_head.npz')['target']
    return dict(train=est['target'], every=np.r_[est['target'], sel['target'], head], x=(st-center)@basis/scale, xv=(sv-center)@basis/scale,
                basis=basis, scale=scale, center=center)


def fit_aqgmm(x, xv, k, scale):
    best = None; tried = []
    for p in range(1 if k == 1 else STARTS):
        ini, _, info = initialize(x, k, p, scale, population_covariance=True)
        g, audit = fit(x, ini, actual_covariance_floor=True, relative_tolerance=True)
        vl = float(np.mean(log_density(xv, g)[0])); tried.append(dict(start=p, rule=info['rule'], selection_ll=vl, status=audit['status']))
        if best is None or vl > best[0]:
            best = (vl, g, p)
    return best[1], dict(chosen_start=best[2], selection_ll=best[0], starts=tried)


def reference():
    ref = np.load(MC, mmap_mode='r'); cols = []
    for j in range(ref.shape[1]):
        r = np.sort(np.asarray(ref[:, j], float)); n = len(r); med = r[n//2]; sd = r.std()
        lo, hi = (med-r[int(.001*n)])/sd, (r[int(.999*n)]-med)/sd
        cols.append(dict(r=r, heavy='lower' if lo >= hi else 'upper', extension_lower=lo, extension_upper=hi))
    return cols


def object_name(j):
    o = base.OBJECTS[j]
    return 'loss' if o['output'] == 'loss' else f"{o['output']}{o['node'] or o['branch']}"


def evaluate(y, w, ref):
    rows = []
    for j, c in enumerate(ref):
        r = c['r']; n = len(r); o = np.argsort(y[:, j]); ys, cw = y[o, j], np.cumsum(w[o])/w.sum()
        F = lambda v: np.interp(v, ys, cw, left=0., right=1.)
        lower = F(r[(LEVELS*n).astype(int)])-LEVELS; upper = (1-F(r[((1-LEVELS)*n).astype(int)]))-LEVELS
        grid = r[np.linspace(0, n-1, 2001).astype(int)]
        err = dict(lower=lower, upper=upper); light = 'upper' if c['heavy'] == 'lower' else 'lower'
        rows.append(dict(object=object_name(j), kind=base.OBJECTS[j]['output'], heavy_side=c['heavy'],
                         heavy_tail=float(np.mean(abs(err[c['heavy']]))), heavy_signed=float(np.mean(err[c['heavy']])),
                         light_tail=float(np.mean(abs(err[light]))), light_signed=float(np.mean(err[light])),
                         ks=float(np.max(abs(F(grid)-np.linspace(0, 1, 2001))))))
    return pd.DataFrame(rows)


def outputs(model, data, seed):
    y, w = gs.model_outputs(model, data, seed); return y, w


def aggregate(per):
    V, P, L = per[per.kind == 'V'], per[per.kind == 'P'], per[per.kind == 'loss']
    return dict(V_heavy=V.heavy_tail.median(), V_light=V.light_tail.median(), V_signed=V.heavy_signed.median(), V_ks=V.ks.median(),
                P_heavy=P.heavy_tail.median(), P_light=P.light_tail.median(), P_signed=P.heavy_signed.median(), P_ks=P.ks.median(),
                loss_heavy=float(L.heavy_tail.iloc[0]), loss_light=float(L.light_tail.iloc[0]), loss_ks=float(L.ks.iloc[0]))


def claims(summary):
    s = summary.set_index(['model', 'K']); out = dict(C1={}, C2={})
    for k in KS_TMM:
        row = {}
        for q in ['V', 'P', 'loss']:
            d = s.loc[('AQGMM', k), f'{q}_heavy']-s.loc[('分组t', k), f'{q}_heavy']
            row[q] = '更准' if d > MARGIN else '更差' if d < -MARGIN else '相当'
            row[q+'_KS变差'] = bool(s.loc[('分组t', k), f'{q}_ks']-s.loc[('AQGMM', k), f'{q}_ks'] > 0.005)
        out['C1'][f'K={k}'] = row
    for kt in [1, 2]:
        row = {}
        for q in ['V', 'P', 'loss']:
            target = s.loc[('分组t', kt), f'{q}_heavy']+MARGIN
            ok = [k for k in KS_GMM if s.loc[('AQGMM', k), f'{q}_heavy'] <= target]
            row[q] = min(ok) if ok else '>16'
        out['C2'][f'分组t K={kt}'] = row
    return out


def main():
    t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); data = load(); rng = np.random.default_rng(2026102450)
    split, Q1, Q2 = chk.split(data['train'], data['x'], data['basis'], data['scale'], rng, SCENE, group_target=data['every'], return_q=True)
    r1 = split['r_first_order']; caps = ts.strength_cap(data['every'], split['groups']); groups, smax = [], []
    for g, cap in zip(split['group_cols'], caps):
        groups.append(np.array([c for c in g if c < r1])); smax.append(cap)
        for c in g:
            if c >= r1:
                groups.append(np.array([c])); smax.append(cap)
    (OUT/'split.json').write_text(json.dumps(dict(split, strength_caps=caps, model_groups=[g.tolist() for g in groups]), ensure_ascii=False, indent=1, default=float))
    print(json.dumps(dict(stage='split', groups=split['groups'], r_first_order=r1, leak=split.get('leak_directions'), caps=[round(c, 2) for c in caps]), ensure_ascii=False), flush=True)
    ref = reference(); rows = []; fits = {}
    for k in KS_GMM:
        gmm, info = fit_aqgmm(data['x'], data['xv'], k, data['scale']); fits[k] = info
        models = [('AQGMM', gmm)]
        if k in KS_TMM:
            std, _, _ = mm.moment_models(data['x'], gmm, Q1, Q2) if len(groups) else (None, None, None)
            models += [('标准TMM', std if std is not None else gmm)]
            models += [('分组t', gs.grouped_model(data['x'], gmm, Q1, Q2, groups, 'sync', rng, s_max=smax) if len(groups) else gmm)]
        for name, m in models:
            y, w = outputs(m, data, 2026102460+k); per = evaluate(y, w, ref)
            per.to_csv(OUT/f'objects_{name}_K{k}.csv', index=False)
            extra = dict(df=[[round(float(v), 2) for v in c['nu']] for c in m['grouped']]) if isinstance(m, dict) and 'grouped' in m else \
                dict(df=np.round(np.asarray(m['df'], float), 2).tolist()) if name == '标准TMM' else {}
            rows.append(dict(model=name, K=k, **aggregate(per), **extra))
            print(json.dumps(dict(stage='evaluated', model=name, K=k, V_heavy=round(rows[-1]['V_heavy'], 4), P_heavy=round(rows[-1]['P_heavy'], 4),
                                  loss_heavy=round(rows[-1]['loss_heavy'], 4), seconds=round(time.perf_counter()-t0)), ensure_ascii=False), flush=True)
    summary = pd.DataFrame(rows); summary.to_csv(OUT/'summary.csv', index=False)
    (OUT/'claims_check.json').write_text(json.dumps(claims(summary), ensure_ascii=False, indent=1, default=str))
    (OUT/'run_info.json').write_text(json.dumps(dict(scene=SCENE, draws=DRAWS, starts=STARTS, aqgmm_fits=fits, heavy_sides=dict(
        V_lower=int(sum(c['heavy'] == 'lower' for c in ref[:32])), P_lower=int(sum(c['heavy'] == 'lower' for c in ref[32:64])), loss=ref[96]['heavy']),
        seconds=time.perf_counter()-t0, script_SHA256=base.sha(Path(__file__))), ensure_ascii=False, indent=1, default=float))
    print(json.dumps(dict(stage='DONE', scene=SCENE, seconds=round(time.perf_counter()-t0)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
