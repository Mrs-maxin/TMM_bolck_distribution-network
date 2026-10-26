"""Source-load scenarios taken from NREL records and their exact AC results (scenario_lock_20261024.md).

Each scenario takes the 8-site forecast-error vector of one recorded 15-min interval, adds 0.05 * L eps
(L L^T = covariance of the interval pool), adds it to the 50% forecast and clips to [0, observed max output].
Each site has 0.5805 x 50/70 = 0.4146 MW, so the 70% forecast equals the former 50% forecast in MW.
Loads follow the case33 three-level rule (network, region, bus, about 1%) at 0.8 x base load.
  S1    Denver 8 sites, forecast 60%-80%, scenarios at 70% forecast
  S2E1  S1 intervals; with probability 5% an interval from the 1% largest-error intervals of forecast 50%-90%
  S2E2  S2E1 plus, in extreme intervals, a load rise of all buses (mean +5%, sd 5% x interval RMS / median
        extreme RMS, bus correlation 0.8, P and Q together)
  S3    Denver 6 sites + Colorado Springs 2 sites, forecast 60%-80%
Estimation (8192) and selection (4096) scenarios: voltages by the fast mapping. MC method: 10^6 scenarios,
AC power flow for every scenario, 97 electrical quantities saved.
Usage: python generate_record_scenarios_case33.py S1|S2E1|S2E2|S3
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
from multiprocessing import Pool
import numpy as np

if len(sys.argv) >= 2:
    os.environ['REC_SCENE'] = sys.argv[1]
SCENE = os.environ['REC_SCENE']
SMOKE = os.environ.get('REC_SMOKE') == '1'
sys.argv = sys.argv[:1]
import run_pv_nrel_case33 as case
base = case.base
import fit_nrel_pv_forecast_error_case33 as one
import fit_nrel_two_region_case33 as two
from fast_voltage_map import fast_map

REPO = base.REPO
OUT = REPO/'result/case33bw/record_scenarios_20261024'/(('smoke_' if SMOKE else '')+SCENE)
CACHE = REPO/'cache/record_scenarios_20261024'
SPLITS = dict(estimation=8192, selection=4096, mc=1_000_000)
if SMOKE:
    SPLITS = dict(estimation=600, selection=300, mc=4000)
SEEDS = dict(S1=2026102411, S3=2026102421, S2E1=2026102431, S2E2=2026102441)
H, FORECAST, P_EXTREME, EXTREME_SHARE = 0.05, 0.7, 0.05, 0.01
BAND, EXTREME_BAND = (0.6, 0.8), (0.5, 0.9)
CAPACITY = base.CAP*0.5/0.7
LOAD_MEAN, LOAD_SD, LOAD_CORR = 0.05, 0.05, 0.8
WORKERS = 8


def pools():
    if SCENE == 'S3':
        two.BAND = BAND; sites, act, fc = two.load_records()
    else:
        one.BAND = BAND; sites, act, fc = one.load_records()
    normal = dict(e=act-fc, upper=act.max(0), sites=[s['key'] for s in sites])
    if not SCENE.startswith('S2'):
        return normal, None
    one.BAND = EXTREME_BAND; s2, act2, fc2 = one.load_records(); assert [s['key'] for s in s2] == normal['sites']
    e2 = act2-fc2; rms = np.sqrt((e2**2).mean(1)); keep = rms >= np.quantile(rms, 1-EXTREME_SHARE)
    return normal, dict(e=e2[keep], upper=act2.max(0), rms=rms[keep], band_intervals=len(e2))


def draw_pv(pool, idx, rng):
    L = np.linalg.cholesky(np.cov(pool['e'].T))
    dev = pool['e'][idx]+H*(rng.standard_normal((len(idx), pool['e'].shape[1]))@L.T)
    return np.clip(FORECAST+dev, 0, pool['upper'])


def loads(n, rng):
    rho_g, rho_a, rho_n = base.CONFIG['load_PQ_correlation']
    wp, wq = np.array(base.CONFIG['load_P_weights'], float), np.array(base.CONFIG['load_Q_weights'], float)
    wp *= base.CONFIG['load_relative_sd']/np.linalg.norm(wp); wq *= base.CONFIG['load_relative_sd']/np.linalg.norm(wq)
    def paired(width, rho):
        a = rng.standard_normal((n, width)); return a, rho*a+np.sqrt(1-rho*rho)*rng.standard_normal((n, width))
    gp, gq = paired(1, rho_g); ap, aq = paired(4, rho_a); zp, zq = paired(32, rho_n)
    return 1+wp[0]*gp+wp[1]*ap[:, base.REGIONS]+wp[2]*zp, 1+wq[0]*gq+wq[1]*aq[:, base.REGIONS]+wq[2]*zq


def scenarios(n, seed, normal, extreme):
    rng = np.random.default_rng(seed)
    extreme_mask = (rng.random(n) < P_EXTREME) if extreme is not None else np.zeros(n, bool)
    pv = np.empty((n, 8)); record = np.empty(n, int)
    i0 = np.flatnonzero(~extreme_mask); record[i0] = rng.integers(0, len(normal['e']), len(i0)); pv[i0] = draw_pv(normal, record[i0], rng)
    i1 = np.flatnonzero(extreme_mask)
    if len(i1):
        record[i1] = rng.integers(0, len(extreme['e']), len(i1)); pv[i1] = draw_pv(extreme, record[i1], rng)
    fp, fq = loads(n, rng)
    if SCENE == 'S2E2' and len(i1):
        zl = rng.standard_normal((len(i1), 32))@np.linalg.cholesky(LOAD_CORR+(1-LOAD_CORR)*np.eye(32)).T
        ratio = extreme['rms'][record[i1]]/np.median(extreme['rms'])
        rise = LOAD_MEAN+LOAD_SD*ratio[:, None]*zl; fp[i1] += rise; fq[i1] += rise
    lp = 0.8*base.BUS[None, 1:, 2]*fp; lq = 0.8*base.BUS[None, 1:, 3]*fq
    pb = np.zeros((n, 32)); pb[:, base.SITES-2] = CAPACITY*pv
    return np.c_[(pb-lp)/base.BASE, -lq/base.BASE], pv, extreme_mask, record


def solve_chunk(target):
    return np.array([base.polar_ac(t, base.Y, tolerance=1e-11)[0] for t in target])


def main():
    t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    assert base.CONFIG['load_multiplier'] == 0.8
    normal, extreme = pools(); info = dict(scene=SCENE, perturbation=H, forecast=FORECAST, site_capacity_MW=CAPACITY, sites=normal['sites'],
                                           normal_intervals=len(normal['e']), seeds={}, mapping={}, seconds={})
    if extreme is not None:
        info.update(extreme_intervals=len(extreme['e']), extreme_band_intervals=extreme['band_intervals'], extreme_probability=P_EXTREME)
    for k, (split, n) in enumerate(SPLITS.items()):
        seed = SEEDS[SCENE]+k; info['seeds'][split] = seed
        target, pv, mask, record = scenarios(n, seed, normal, extreme)
        if split != 'mc':
            v, m = fast_map(target); info['mapping'][split] = m
            base.save_npz(OUT/f'{split}_scenarios.npz', target=target, PV_pu=pv, extreme=mask, record=record, voltage=v)
        else:
            base.save_npz(OUT/'mc_inputs_head.npz', target=target[:32768], PV_pu=pv[:32768], extreme=mask[:32768])
            path = CACHE/f"{('smoke_' if SMOKE else '')+SCENE}_mc_outputs.npy"
            tm = time.perf_counter()
            with Pool(WORKERS) as pool:
                v = np.concatenate(pool.map(solve_chunk, np.array_split(target, WORKERS*32)))
            ref = np.concatenate([base.outputs(c).astype(np.float32) for c in np.array_split(v, 64)]); np.save(path, ref)
            info['seconds']['mc_ac'] = time.perf_counter()-tm; info['mc_outputs'] = str(path.relative_to(REPO))
            info['extreme_share_mc'] = float(mask.mean())
            dev = pv-FORECAST; rec = np.clip(FORECAST+normal['e'], 0, normal['upper'])-FORECAST
            info['plausibility'] = dict(records=one.describe(rec, None), scenarios=one.describe(dev[~mask], None))
            if extreme is not None:
                rec2 = np.clip(FORECAST+extreme['e'], 0, extreme['upper'])-FORECAST
                info['plausibility'].update(extreme_records=one.describe(rec2, None), extreme_scenarios=one.describe(dev[mask], None))
            info['MC_voltage_minimum_pu'] = float(ref[:, :32].min()); info['MC_voltage_maximum_pu'] = float(ref[:, :32].max())
        print(json.dumps(dict(stage='split_done', split=split, seconds=round(time.perf_counter()-t0, 1))), flush=True)
    info['seconds']['total'] = time.perf_counter()-t0; info['script_SHA256'] = base.sha(Path(__file__))
    (OUT/'run_info.json').write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(dict(stage='DONE', scene=SCENE, seconds=round(info['seconds']['total'], 1))), flush=True)


if __name__ == '__main__':
    main()
