"""Extreme-weather tests on the IEEE 33-bus network (case_study_plan_20261022.md section 5.2).

Normal intervals follow the case-1 NREL law. With probability p an extreme interval (fast convective
cloud passage) occurs: the same law with the strength multiplied by c, c calibrated so that the median
8-site RMS error of extreme intervals equals that of the largest 1% NREL intervals (forecast 30%-70%).
E2 adds, in extreme intervals only, a load rise at all 32 buses (mean +5%, sd 5% x sqrt(S), P and Q
together, bus correlation 0.8). Models: AQGMM, standard TMM, block TMM and block TMM with shared
residual (split by derivation.md section J), K in KS, compared at equal K against the MC method
(10^6 AC power flows).
Usage: python extreme_weather_case33.py E1|E2 p
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

if len(sys.argv) >= 3:
    os.environ['EXW_SCENE'], os.environ['EXW_P'] = sys.argv[1], sys.argv[2]
SCENE, P = os.environ['EXW_SCENE'], float(os.environ['EXW_P'])
SMOKE = os.environ.get('EXW_SMOKE') == '1'
sys.argv = sys.argv[:1]
import run_pv_nrel_case33 as case
base = case.base
import shared_residual_mixture as shared
import block_split_rule_check as chk
import fit_nrel_pv_forecast_error_case33 as nrel
from cloud_numeric_core import initialize, fit, log_density

REPO = base.REPO; TAG = f'{SCENE}_p{P:g}'
OUT = REPO/"result/case33bw/extreme_weather_case33_20261022"/TAG
CACHE = REPO/'cache/extreme_weather_case33_20261022'; CACHE.mkdir(parents=True, exist_ok=True)
SPLITS = dict(training=8192, validation=4096, mc=1_000_000)
GROUP_SCENARIOS = 45056
KS = [1, 3, 6, 8, 12]
DRAWS = 262144
LEVELS = np.linspace(.001, .05, 50)
LOAD_MEAN, LOAD_SD, LOAD_CORR = 0.05, 0.05, 0.8
SEED = {'E1': 2026102301, 'E2': 2026102302}[SCENE]+int(round(P*1000))
WORKERS = 8
if os.environ.get('EXW_TRAIN'):
    SPLITS['training'] = int(os.environ['EXW_TRAIN']); TAG += f"_train{SPLITS['training']}"; OUT = OUT.parent/TAG
if os.environ.get('EXW_KS'):
    KS = [int(k) for k in os.environ['EXW_KS'].split(',')]
if SMOKE:
    SPLITS = dict(training=1200, validation=600, mc=20000); GROUP_SCENARIOS = 8000; KS = [1, 3]; DRAWS = 16384
    TAG = 'smoke_'+TAG; OUT = OUT.parent/TAG; OUT.mkdir(parents=True, exist_ok=True)


def storm_factor(law, rng):
    nrel.BAND = (0.3, 0.7); _, act, fc = nrel.load_records(); e = act-fc
    rms = np.sqrt((e**2).mean(1)); target = float(np.median(rms[rms >= np.quantile(rms, .99)]))
    z = rng.standard_normal((200000, 8))@np.linalg.cholesky(law['covariance']).T; s0 = rng.standard_normal(200000)
    def med(c):
        S = c*np.exp(law['log_sd']*s0-law['log_sd']**2/2)
        pv = np.clip(.5+law['mean']+np.sqrt(S)[:, None]*z, 0, law['upper'])
        return np.median(np.sqrt(((pv-.5)**2).mean(1)))
    lo, hi = 1., 400.
    for _ in range(60):
        mid = np.sqrt(lo*hi); lo, hi = (mid, hi) if med(mid) < target else (lo, mid)
    return float(np.sqrt(lo*hi)), target


def scenarios(n, rng, law, c):
    sigma = law['log_sd']; storm = rng.random(n) < P
    S = np.exp(sigma*rng.standard_normal(n)-sigma**2/2)*np.where(storm, c, 1.)
    pv_pu = np.clip(base.CONFIG['forecast_fraction']+law['mean']+np.sqrt(S)[:, None]*(rng.standard_normal((n, 8))@np.linalg.cholesky(law['covariance']).T),
                    0, law['upper'])
    fp = 1+0.01*rng.standard_normal((n, 32)); fq = 1+0.01*rng.standard_normal((n, 32))
    if SCENE == 'E2':
        zl = rng.standard_normal((n, 32))@np.linalg.cholesky(LOAD_CORR+(1-LOAD_CORR)*np.eye(32)).T
        rise = LOAD_MEAN+LOAD_SD*np.sqrt(S/c)[:, None]*zl
        fp = np.where(storm[:, None], fp+rise, fp); fq = np.where(storm[:, None], fq+rise, fq)
    lp, lq = 0.8*base.BUS[None, 1:, 2]*fp, 0.8*base.BUS[None, 1:, 3]*fq
    pb = np.zeros((n, 32)); pb[:, base.SITES-2] = base.CAP*pv_pu
    return np.c_[(pb-lp)/base.BASE, -lq/base.BASE], storm


def solve_chunk(target):
    v = np.array([base.polar_ac(t, base.Y, tolerance=1e-11)[0] for t in target]); return v


def ac(target):
    with Pool(WORKERS) as pool:
        return np.concatenate(pool.map(solve_chunk, np.array_split(target, WORKERS*16)))


def to_voltage(z, center, vt, scale):
    s = center+(z*scale)@vt; return np.c_[np.ones(len(s)), s[:, :32]+1j*s[:, 32:]]


def fit_all(xt, xv, scale, Q1, Q2):
    Ut, Wt, Uv, Wv = xt@Q1, xt@Q2, xv@Q1, xv@Q2; models = []
    for k in KS:
        best = {}
        def keep(fam, m, vl, audit):
            if fam not in best or vl > best[fam][1]:
                best[fam] = (m, vl, audit)
        for p in range(1 if k == 1 else 2):
            ini, _, _ = initialize(xt, k, p, scale, population_covariance=True)
            g, a = fit(xt, ini, actual_covariance_floor=True, relative_tolerance=True)
            keep('GMM', g, float(np.mean(log_density(xv, g)[0])), a['status'])
            for nu in [12, 200]:
                m0 = {kk: np.array(g[kk], copy=True) for kk in ['weights', 'means', 'scatter', 'df']}; m0['df'][:] = nu; m0['scatter'] *= (nu-2)/nu
                t, a = fit(xt, m0, actual_covariance_floor=True, relative_tolerance=True)
                keep('TMM', t, float(np.mean(log_density(xv, t)[0])), a['status'])
                if Q2.shape[1] == 0 or Q1.shape[1] == 0:
                    continue
                b, a = base.fit_block(Ut, Wt, base.block_from_gmm(g, Q1, Q2, nu)); b.update(Q1=Q1, Q2=Q2, block=np.array(True))
                keep('BTMM', b, float(base.block_density(Uv, Wv, b)[0].mean()), a['status'])
                sb, a = shared.fit_shared_block(Ut, Wt, shared.shared_from_gmm(g, Q1, Q2, nu, False), False); sb.update(Q1=Q1, Q2=Q2, block=np.array(True))
                keep('SBTMM', sb, float(base.block_density(Uv, Wv, sb)[0].mean()), a['status'])
        for fam, (m, vl, status) in best.items():
            models.append(dict(K=k, family=fam, model=m, validation_ll=vl, status=status,
                               df=np.round(np.asarray(m['df'], float), 1).tolist()))
        print(json.dumps(dict(stage='fitted', K=k, ll={f: round(v[1], 3) for f, v in best.items()})), flush=True)
    return models


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter(); rng = np.random.default_rng(SEED); law = case.nrel_law()
    c, target = storm_factor(law, rng)
    data = {s: scenarios(n, rng, law, c) for s, n in SPLITS.items()}
    group_target = data['mc'][0][:GROUP_SCENARIOS]
    volt = {s: ac(data[s][0]) for s in ['training', 'validation']}
    mc_path = CACHE/f'{TAG}_mc_outputs.npy'
    if mc_path.exists():
        ref = np.load(mc_path)
    else:
        vmc = ac(data['mc'][0]); ref = np.concatenate([base.outputs(v).astype(np.float32) for v in np.array_split(vmc, 64)]); np.save(mc_path, ref); del vmc
    print(json.dumps(dict(stage='AC_done', seconds=time.perf_counter()-t0, storm_factor=c)), flush=True)
    st = {s: np.c_[v[:, 1:].real, v[:, 1:].imag] for s, v in volt.items()}
    center = st['training'].mean(0); _, sv, vt = np.linalg.svd(st['training']-center, full_matrices=False); scale = sv/np.sqrt(len(st['training'])-1)
    xt, xv = (st['training']-center)@vt.T/scale, (st['validation']-center)@vt.T/scale
    split_row, Q1, Q2 = chk.split(data['training'][0], xt, vt.T, scale, rng, TAG, group_target=group_target, return_q=True)
    nobj = 96; srt = [np.sort(ref[:, j]) for j in range(nobj)]
    ext = []
    for j in range(nobj):
        r = srt[j]; sd = r.std(); q = np.quantile(r, [.001, .5, .999])
        ext.append(((q[1]-q[0])/sd, 'lower')); ext.append(((q[2]-q[1])/sd, 'upper'))
    order = np.argsort([-e[0] for e in ext]); targets = []
    for i in order:
        j = i//2
        if base.OBJECTS[j]['output'] in ['V', 'P'] and j not in [t[0] for t in targets] and len(targets) < 8:
            targets.append((j, ext[i][1], float(ext[i][0])))
    (OUT/'targets_locked.json').write_text(json.dumps(dict(rule='largest 0.1%-tail extension over std in the MC result, V and P only, before model errors',
        targets=[dict(object_index=int(j), **base.OBJECTS[j], side=s, extension=e) for j, s, e in targets]), indent=1, default=lambda o: o.item()))
    models = fit_all(xt, xv, scale, Q1, Q2); rows = []
    for m in models:
        z, w = base.propagate_any(m['model'], DRAWS, SEED+7*m['K'])
        y = np.concatenate([base.outputs(to_voltage(zz, center, vt, scale)) for zz in np.array_split(z, 16)])
        per = []
        for j in range(nobj):
            r = srt[j]; o = np.argsort(y[:, j]); ys, cw = y[o, j], np.cumsum(w[o])/w.sum()
            F = lambda x: np.interp(x, ys, cw, left=0., right=1.)
            grid = r[np.linspace(0, len(r)-1, 2001).astype(int)]; Fr = np.linspace(0, 1, 2001)
            ql, qh = r[(LEVELS*len(r)).astype(int)], r[((1-LEVELS)*len(r)).astype(int)]
            dl = F(ql)-LEVELS; dh = (1-F(qh))-LEVELS
            per.append(dict(object_index=j, ks=float(np.max(abs(F(grid)-Fr))), lower=float(np.mean(abs(dl))), upper=float(np.mean(abs(dh))),
                            lower_signed=float(np.mean(dl)), upper_signed=float(np.mean(dh)),
                            beyond_q001=float(F(r[int(.001*len(r))])), beyond_q999=float(1-F(r[int(.999*len(r))]))))
        per = pd.DataFrame(per); V = per.object_index < 32; Pb = (per.object_index >= 32) & (per.object_index < 64)
        tg = [per.iloc[j] for j, s, e in targets]
        rows.append(dict(K=m['K'], family=m['family'], validation_ll=m['validation_ll'], status=m['status'], df=m['df'],
                         V_ks_median=per[V].ks.median(), P_ks_median=per[Pb].ks.median(),
                         target_tail_err_median=float(np.median([t[s] for t, (j, s, e) in zip(tg, targets)])),
                         target_tail_signed_median=float(np.median([t[s+'_signed'] for t, (j, s, e) in zip(tg, targets)])),
                         target_beyond_01_median=float(np.median([t['beyond_q001'] if s == 'lower' else t['beyond_q999'] for t, (j, s, e) in zip(tg, targets)])),
                         **{f"T_{base.OBJECTS[j]['output']}{base.OBJECTS[j]['node'] or base.OBJECTS[j]['branch']}_{s}": float(t[s]) for t, (j, s, e) in zip(tg, targets)}))
        per.to_csv(OUT/f"objects_{m['family']}_K{m['K']}.csv", index=False)
        print(json.dumps(dict(stage='evaluated', K=m['K'], family=m['family'], tail=round(rows[-1]['target_tail_err_median'], 4))), flush=True)
    pd.DataFrame(rows).to_csv(OUT/'summary_same_K.csv', index=False)
    mc_storm = data['mc'][1]
    (OUT/'run_info.json').write_text(json.dumps(dict(scene=SCENE, p=P, storm_factor=c, storm_target_rms=target, realized_storm_share=float(mc_storm.mean()),
        split=split_row, splits=SPLITS, draws=DRAWS, seed=SEED, load_rise=dict(mean=LOAD_MEAN, sd=LOAD_SD, corr=LOAD_CORR) if SCENE == 'E2' else None,
        MC_voltage_excess_kurtosis=[float(x) for x in __import__('scipy.stats', fromlist=['kurtosis']).kurtosis(ref[:, :32], axis=0)[[0, -1]]],
        seconds=time.perf_counter()-t0, script_SHA256=base.sha(Path(__file__))), indent=1, default=str))
    print(json.dumps(dict(stage='DONE', seconds=time.perf_counter()-t0)), flush=True)


if __name__ == '__main__':
    main()
