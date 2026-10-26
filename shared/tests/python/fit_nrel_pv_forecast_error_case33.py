"""Build the case33 PV forecast-error input from NREL Solar Power Data for Integration Studies.

Conditional forecast-error modelling: daytime 15-min intervals are grouped by the
4-hour-ahead forecast level; inside the selected group the 8-site errors (per unit of
capacity) are described as a common weather strength times a correlated deviation,
e = mu + sqrt(S) * L z, E[S] = 1. Two strength laws are fitted by maximum likelihood
(lognormal and inverse-gamma); the fitted laws are compared with the records and with a
correlated Gaussian deviation of the same covariance.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data'/'nrel_sind'/'co'
OUT = ROOT/'result'/'case33bw'/'nrel_pv_input_case33_20261022'
CENTRE = (39.75, -104.95)
N_SITES = 8
BAND = tuple(float(v) for v in os.environ.get('NREL_BAND', '0.4,0.6').split(','))
FORECAST = os.environ.get('NREL_FORECAST', 'HA4')
N_GH = 48


def distance_km(lat1, lon1, lat2, lon2):
    a, b, c, d = map(np.radians, (lat1, lon1, lat2, lon2))
    return 6371*np.arccos(np.clip(np.sin(a)*np.sin(c)+np.cos(a)*np.cos(c)*np.cos(b-d), -1, 1))


def load_records():
    sites = []
    for f in os.listdir(DATA):
        if f.startswith('Actual_') and '_DPV_' in f:
            p = f.split('_')
            sites.append(dict(lat=float(p[1]), lon=float(p[2]), cap=float(p[5][:-2]), key='_'.join(p[1:6])))
    for s in sites:
        s['dist'] = float(distance_km(*CENTRE, s['lat'], s['lon']))
    sites = sorted(sites, key=lambda s: s['dist'])[:N_SITES]
    cap = np.array([s['cap'] for s in sites])
    act = np.stack([np.loadtxt(DATA/f"Actual_{s['key']}_5_Min.csv", delimiter=',', skiprows=1, usecols=1)
                    for s in sites], 1)/cap
    fc = np.stack([np.loadtxt(DATA/f"{FORECAST}_{s['key']}_60_Min.csv", delimiter=',', skiprows=1, usecols=1)
                   for s in sites], 1)/cap
    n15 = act.shape[0]//3
    act15 = act[:n15*3].reshape(n15, 3, -1).mean(1)
    fc15 = np.repeat(fc, 4, 0)[:n15]
    level = fc15.mean(1)
    keep = (level >= BAND[0]) & (level < BAND[1])
    return sites, act15[keep], fc15[keep]


def loglik_lognormal(sigma, d, p, logdet, share=1.0):
    x, w = np.polynomial.hermite.hermgauss(N_GH)
    log_s = np.log(share*np.exp(np.sqrt(2)*sigma*x-sigma**2/2)+1-share)
    ls = log_s[None, :]
    term = -0.5*(p*ls+d[:, None]*np.exp(-ls))
    m = term.max(1, keepdims=True)
    lse = m[:, 0]+np.log((np.exp(term-m)*w[None, :]).sum(1)/np.sqrt(np.pi))
    return float(np.sum(lse-0.5*p*np.log(2*np.pi)-0.5*logdet))


def loglik_t(nu, d, p, logdet):
    scale = (nu-2)/nu
    return float(np.sum(special.gammaln((nu+p)/2)-special.gammaln(nu/2)-0.5*p*np.log(nu*np.pi*scale)
                        -0.5*logdet-0.5*(nu+p)*np.log1p(d/(nu*scale))))


def simulate(mu, chol, law, par, n, rng):
    z = rng.standard_normal((n, len(mu)))@chol.T
    if law == 'lognormal':
        s = np.exp(par*rng.standard_normal(n)-par**2/2)
    elif law == 'lognormal_share':
        s = par[1]*np.exp(par[0]*rng.standard_normal(n)-par[0]**2/2)+1-par[1]
    elif law == 't':
        s = (par-2)/rng.chisquare(par, n)
    else:
        s = np.ones(n)
    return mu+np.sqrt(s)[:, None]*z


def describe(e, fc, upper=1.0):
    act = np.clip(fc+e, 0, upper) if fc is not None else None
    ee = act-fc if fc is not None else e
    c = np.corrcoef(ee.T); iu = np.triu_indices(ee.shape[1], 1)
    dev = ee-ee.mean(0)
    kurt = stats.kurtosis(ee, 0)
    sq = np.corrcoef((dev**2).T)[iu].mean()
    big = (np.abs(dev) > 2*dev.std(0)).sum(1)
    spread = np.sqrt((dev**2).mean(1))
    q = np.quantile(spread, [0.1, 0.9])
    return dict(rms=float(np.sqrt((dev**2).mean())), kurt_median=float(np.median(kurt)), corr=float(c[iu].mean()),
                sq_corr=float(sq), share_half_big=float((big >= ee.shape[1]//2).mean()),
                spread_ratio=float(q[1]/q[0]), q001=float(np.quantile(ee.mean(1), 0.001)),
                q999=float(np.quantile(ee.mean(1), 0.999)))


def main():
    sites, act, fc = load_records()
    e = act-fc
    n, p = e.shape
    mu = e.mean(0); cov = np.cov(e.T)
    chol = np.linalg.cholesky(cov); logdet = 2*np.log(np.diag(chol)).sum()
    d = np.sum(np.linalg.solve(chol, (e-mu).T)**2, 0)
    gauss = float(np.sum(-0.5*d-0.5*p*np.log(2*np.pi)-0.5*logdet))
    ln = optimize.minimize_scalar(lambda s: -loglik_lognormal(s, d, p, logdet), bounds=(0.01, 2.5), method='bounded')
    tt = optimize.minimize_scalar(lambda v: -loglik_t(v, d, p, logdet), bounds=(2.05, 300), method='bounded')
    sh = optimize.minimize(lambda q: -loglik_lognormal(q[0], d, p, logdet, q[1]), x0=[1.2, 0.8],
                           bounds=[(0.01, 3.0), (0.01, 0.999)], method='L-BFGS-B')
    upper = act.max(0)
    print(f'records {n} sites {p} band {BAND} forecast {FORECAST} mean forecast {fc.mean():.3f} '
          f'observed max output {np.round(upper, 3)}')
    print(f'log-likelihood  Gaussian {gauss:.1f} | lognormal strength {-ln.fun:.1f} (log-sd {ln.x:.3f}) '
          f'| inverse-gamma strength {-tt.fun:.1f} (df {tt.x:.2f}) '
          f'| lognormal strength on share {-sh.fun:.1f} (log-sd {sh.x[0]:.3f}, share {sh.x[1]:.3f})')
    rng = np.random.default_rng(20261022)
    m = 200000
    fc_rep = fc[rng.integers(0, n, m)]
    rows = {'records': describe(e, None)}
    for law, par in [('Gaussian', None), ('lognormal', ln.x), ('t', tt.x), ('lognormal_share', sh.x)]:
        rows[law] = describe(simulate(mu, chol, law, par, m, rng), fc_rep, upper)
    keys = list(rows['records'])
    print('quantity'.ljust(16)+''.join(k.rjust(14) for k in rows))
    for k in keys:
        print(k.ljust(16)+''.join(f'{rows[r][k]:14.3f}' for r in rows))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'fit_{FORECAST}_{BAND[0]:.1f}_{BAND[1]:.1f}.json').write_text(json.dumps(dict(
        sites=sites, band=BAND, forecast=FORECAST, records=n, mean_error=mu.tolist(), covariance=cov.tolist(),
        loglik=dict(gaussian=gauss, lognormal=-ln.fun, inverse_gamma=-tt.fun, lognormal_share=-sh.fun),
        lognormal_log_sd=ln.x, inverse_gamma_df=tt.x, share_log_sd=float(sh.x[0]), share=float(sh.x[1]),
        observed_max_output=upper.tolist(), comparison=rows), indent=1))


if __name__ == '__main__':
    main()
