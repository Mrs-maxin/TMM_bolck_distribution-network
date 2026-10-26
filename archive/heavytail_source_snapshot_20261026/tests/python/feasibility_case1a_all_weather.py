"""Feasibility check for case 1-A: all bus injections follow one weather process.

PV at the 8 case-1 sites follows the NREL-fitted law (common lognormal strength, log-sd 1.19, outputs
held at zero or the clear-sky maximum). Every load P and Q deviates from its forecast by LOAD_SD times
the square root of the same strength (bus-to-bus correlation LOAD_CORR, P and Q with the same factor),
plus LOAD_INDEPENDENT independent fluctuation. Standard TMM and GMM are fitted at equal K on the same
scenarios; node-voltage magnitude distributions are compared with the MC method (AC power flow of every
scenario). LOAD_SD is provisional until checked against the load-forecast literature.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import sys, json
sys.dont_write_bytecode = True
import numpy as np
import run_pv_nrel_case33 as case
base = case.base
from cloud_numeric_core import initialize, fit, log_density

STRENGTH_LAW = os.environ.get('CASE1A_STRENGTH', 'lognormal')
INVGAMMA_DF = 5.0
OUT = base.REPO/'result/case33bw'/('feasibility_case1a_all_weather_20261022' if STRENGTH_LAW == 'lognormal'
                                     else f'feasibility_case1a_all_weather_invgamma_df{INVGAMMA_DF:g}_20261022')
OUT.mkdir(parents=True, exist_ok=True)
LOAD_SD, LOAD_CORR, LOAD_INDEPENDENT = 0.05, 0.8, 0.005
SPLITS = {'training': 8192, 'validation': 4096, 'reference': 32768}
KS = [1, 3, 6, 8]
DRAWS = 65536
LEVELS = np.r_[np.linspace(.001, .05, 50)]


def scenarios(n, rng):
    law = case.nrel_law(); sigma = law['log_sd']
    if STRENGTH_LAW == 'lognormal':
        S = np.exp(sigma*rng.standard_normal(n)-sigma**2/2)
    else:
        S = (INVGAMMA_DF-2)/rng.chisquare(INVGAMMA_DF, n)
    pv_pu = np.clip(base.CONFIG['forecast_fraction']+law['mean']+np.sqrt(S)[:, None]*(rng.standard_normal((n, 8))@np.linalg.cholesky(law['covariance']).T),
                    0, law['upper'])
    zl = rng.standard_normal((n, 32))@np.linalg.cholesky(LOAD_CORR+(1-LOAD_CORR)*np.eye(32)).T
    f = 1+LOAD_SD*np.sqrt(S)[:, None]*zl
    fp = f+LOAD_INDEPENDENT*rng.standard_normal((n, 32)); fq = f+LOAD_INDEPENDENT*rng.standard_normal((n, 32))
    lp, lq = 0.8*base.BUS[None, 1:, 2]*fp, 0.8*base.BUS[None, 1:, 3]*fq
    pb = np.zeros((n, 32)); pb[:, base.SITES-2] = base.CAP*pv_pu
    return np.c_[(pb-lp)/base.BASE, -lq/base.BASE]


def ac(target):
    return np.array([base.polar_ac(t, base.Y, tolerance=1e-11)[0] for t in target])


def sample(model, n, rng):
    k = rng.choice(len(model['weights']), size=n, p=model['weights']); d = model['means'].shape[1]; out = np.empty((n, d))
    for j in range(len(model['weights'])):
        m = k == j; L = np.linalg.cholesky(model['scatter'][j]); z = rng.standard_normal((m.sum(), d))@L.T
        nu = model['df'][j]; s = np.ones(m.sum()) if not np.isfinite(nu) else nu/rng.chisquare(nu, m.sum())
        out[m] = model['means'][j]+np.sqrt(s)[:, None]*z
    return out


def main():
    rng = np.random.default_rng(2026102291); data = {}
    for split, n in SPLITS.items():
        t = scenarios(n, rng); v = ac(t); data[split] = (t, v)
        print('AC done', split, flush=True)
    st = {s: np.c_[v[:, 1:].real, v[:, 1:].imag] for s, (_, v) in data.items()}
    center = st['training'].mean(0); _, sv, vt = np.linalg.svd(st['training']-center, full_matrices=False); scale = sv/np.sqrt(len(st['training'])-1)
    xt, xv = (st['training']-center)@vt.T/scale, (st['validation']-center)@vt.T/scale
    vref = np.abs(data['reference'][1][:, 1:])
    rows = []
    for k in KS:
        best = {}
        for p in range(1 if k == 1 else 2):
            ini, _, _ = initialize(xt, k, p, scale, population_covariance=True)
            g, _ = fit(xt, ini, actual_covariance_floor=True, relative_tolerance=True)
            cands = [('GMM', g)]
            for nu in [12, 200]:
                m0 = {kk: np.array(g[kk], copy=True) for kk in ['weights', 'means', 'scatter', 'df']}; m0['df'][:] = nu; m0['scatter'] *= (nu-2)/nu
                cands.append(('TMM', fit(xt, m0, actual_covariance_floor=True, relative_tolerance=True)[0]))
            for fam, m in cands:
                vl = float(np.mean(log_density(xv, m)[0]))
                if fam not in best or vl > best[fam][0]:
                    best[fam] = (vl, m)
        for fam, (vl, m) in best.items():
            z = sample(m, DRAWS, rng); s = center+(z*scale)@vt; vm = np.abs(np.c_[np.ones(DRAWS), s[:, :32]+1j*s[:, 32:]])[:, 1:]
            ks, lo, hi, los, his, deep = [], [], [], [], [], []
            for j in range(32):
                r = np.sort(vref[:, j]); mm = np.sort(vm[:, j])
                grid = np.quantile(r, np.linspace(0, 1, 2001)); Fr = np.searchsorted(r, grid, 'right')/len(r); Fm = np.searchsorted(mm, grid, 'right')/len(mm)
                ks.append(np.max(abs(Fr-Fm)))
                ql, qh = np.quantile(r, LEVELS), np.quantile(r, 1-LEVELS)
                dl = np.searchsorted(mm, ql, 'right')/len(mm)-LEVELS; dh = 1-np.searchsorted(mm, qh, 'right')/len(mm)-LEVELS
                lo.append(np.mean(abs(dl))); hi.append(np.mean(abs(dh))); los.append(np.mean(dl)); his.append(np.mean(dh))
                deep.append([float(np.mean(mm < np.quantile(r, .001))), float(np.mean(mm > np.quantile(r, .999)))])
            rows.append(dict(K=k, family=fam, validation_ll=vl, df=np.round(m['df'], 1).tolist() if fam == 'TMM' else [],
                             voltage_ks_median=float(np.median(ks)), voltage_ks_max=float(np.max(ks)),
                             lower_tail_err_median=float(np.median(lo)), upper_tail_err_median=float(np.median(hi)),
                             lower_tail_err_max=float(np.max(lo)), upper_tail_err_max=float(np.max(hi)),
                             lower_tail_signed_median=float(np.median(los)), upper_tail_signed_median=float(np.median(his)),
                             beyond_MC_q001_median=float(np.median([d[0] for d in deep])), beyond_MC_q999_median=float(np.median([d[1] for d in deep]))))
            print(json.dumps(rows[-1]), flush=True)
    kurt = __import__('scipy.stats', fromlist=['kurtosis']).kurtosis(vref, axis=0)
    (OUT/'feasibility_results.json').write_text(json.dumps(dict(settings=dict(strength_law=STRENGTH_LAW, invgamma_df=INVGAMMA_DF, LOAD_SD=LOAD_SD, LOAD_CORR=LOAD_CORR, LOAD_INDEPENDENT=LOAD_INDEPENDENT,
        splits=SPLITS, draws=DRAWS, tail_levels='0.1%-5% both sides, mean |model CDF - level| at MC quantiles'),
        voltage_excess_kurtosis_range=[float(kurt.min()), float(kurt.max())], rows=rows), indent=1))


if __name__ == '__main__':
    main()
