"""Two-cloud-region PV forecast-error input for case33 from NREL records (scenario_lock_20261024.md, S3).

Region A: the 6 Denver distributed-PV sites nearest to 39.75N 104.95W (buses 18, 17, 16, 15, 14, 13).
Region B: the 2 Colorado Springs sites nearest to 38.85N 104.75W (buses 33, 22).
Daytime 15-min intervals whose 8-site mean HA4 forecast lies in 40%-60% of capacity. The error is
e = mu + D(sqrt(S)) L z with one strength per region, (ln S_A, ln S_B) bivariate normal, E[S] = 1.
Fitted by maximum likelihood: correlated strengths, one common strength, independent strengths.
"""
import json
from pathlib import Path

import numpy as np
from scipy import optimize

import fit_nrel_pv_forecast_error_case33 as one

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'result'/'case33bw'/'nrel_pv_two_region_input_case33_20261024'
CENTRES = {'A': ((39.75, -104.95), 6), 'B': ((38.85, -104.75), 2)}
BAND = (0.4, 0.6)
N_GH = 24


def load_records():
    sites = []
    for f in sorted(one.DATA.iterdir()):
        if f.name.startswith('Actual_') and '_DPV_' in f.name:
            p = f.name.split('_')
            sites.append(dict(lat=float(p[1]), lon=float(p[2]), cap=float(p[5][:-2]), key='_'.join(p[1:6])))
    chosen = []
    for region, (centre, count) in CENTRES.items():
        near = sorted(sites, key=lambda s: one.distance_km(*centre, s['lat'], s['lon']))[:count]
        chosen += [dict(s, region=region, dist=float(one.distance_km(*centre, s['lat'], s['lon']))) for s in near]
    cap = np.array([s['cap'] for s in chosen])
    act = np.stack([np.loadtxt(one.DATA/f"Actual_{s['key']}_5_Min.csv", delimiter=',', skiprows=1, usecols=1) for s in chosen], 1)/cap
    fc = np.stack([np.loadtxt(one.DATA/f"HA4_{s['key']}_60_Min.csv", delimiter=',', skiprows=1, usecols=1) for s in chosen], 1)/cap
    n15 = act.shape[0]//3
    act15 = act[:n15*3].reshape(n15, 3, -1).mean(1); fc15 = np.repeat(fc, 4, 0)[:n15]
    level = fc15.mean(1); keep = (level >= BAND[0]) & (level < BAND[1])
    return chosen, act15[keep], fc15[keep]


def pair_mean_sqrt(sa, sb, rho):
    return np.exp(-(sa-sb)**2/8-sa*sb*(1-rho)/4)


def scale_matrix(cov, ia, ib, sa, sb, rho):
    sig = cov.copy(); k = pair_mean_sqrt(sa, sb, rho)
    sig[np.ix_(ia, ib)] /= k; sig[np.ix_(ib, ia)] /= k
    return sig


def loglik(par, y, ia, ib, cov):
    sa, sb, rho = par
    sig = scale_matrix(cov, ia, ib, sa, sb, rho)
    w, V = np.linalg.eigh(sig)
    if w.min() <= 0:
        return -np.inf
    P = V@np.diag(1/w)@V.T; logdet = np.log(w).sum()
    a = np.einsum('ij,jk,ik->i', y[:, ia], P[np.ix_(ia, ia)], y[:, ia])
    b = np.einsum('ij,jk,ik->i', y[:, ib], P[np.ix_(ib, ib)], y[:, ib])
    c = 2*np.einsum('ij,jk,ik->i', y[:, ia], P[np.ix_(ia, ib)], y[:, ib])
    x, wt = np.polynomial.hermite.hermgauss(N_GH)
    z1, z2 = np.meshgrid(np.sqrt(2)*x, np.sqrt(2)*x, indexing='ij'); ww = np.outer(wt, wt)/np.pi
    la = sa*z1-sa**2/2; lb = sb*(rho*z1+np.sqrt(max(1-rho**2, 0))*z2)-sb**2/2
    la, lb, ww = la.ravel(), lb.ravel(), ww.ravel()
    term = (-0.5*(len(ia)*la[None]+len(ib)*lb[None]+a[:, None]*np.exp(-la)[None]+b[:, None]*np.exp(-lb)[None]
                  + c[:, None]*np.exp(-(la+lb)/2)[None]))
    m = term.max(1, keepdims=True)
    lse = m[:, 0]+np.log((np.exp(term-m)*ww[None]).sum(1))
    return float(np.sum(lse-0.5*y.shape[1]*np.log(2*np.pi)-0.5*logdet))


def simulate(mu, cov, ia, ib, par, n, rng):
    sa, sb, rho = par
    L = np.linalg.cholesky(scale_matrix(cov, ia, ib, sa, sb, rho))
    g = rng.standard_normal((n, 2)); la = sa*g[:, 0]-sa**2/2; lb = sb*(rho*g[:, 0]+np.sqrt(max(1-rho**2, 0))*g[:, 1])-sb**2/2
    s = np.empty((n, len(mu))); s[:, ia] = np.exp(la)[:, None]; s[:, ib] = np.exp(lb)[:, None]
    return mu+np.sqrt(s)*(rng.standard_normal((n, len(mu)))@L.T)


def cross_coamplification(e, ia, ib):
    d = e-e.mean(0); r = np.corrcoef(d.T); q = np.corrcoef((d**2).T); c = q-r**2
    within_a = c[np.ix_(ia, ia)][~np.eye(len(ia), dtype=bool)]
    return dict(within_A=float(np.median(within_a)), within_B=float(c[ib[0], ib[1]]), across=float(np.median(c[np.ix_(ia, ib)])),
                corr_across=float(np.median(r[np.ix_(ia, ib)])))


def main():
    sites, act, fc = load_records()
    e = act-fc; n, p = e.shape
    ia = np.array([i for i, s in enumerate(sites) if s['region'] == 'A']); ib = np.array([i for i, s in enumerate(sites) if s['region'] == 'B'])
    mu = e.mean(0); cov = np.cov(e.T); y = e-mu
    fits = {}
    r = optimize.minimize(lambda q: -loglik(q, y, ia, ib, cov), x0=[1.2, 1.2, 0.5], method='Nelder-Mead',
                          options=dict(xatol=1e-4, fatol=1e-3, maxiter=2000))
    fits['correlated'] = dict(par=[float(v) for v in r.x], loglik=-float(r.fun))
    r = optimize.minimize_scalar(lambda s: -loglik([s, s, 1.0], y, ia, ib, cov), bounds=(0.01, 2.5), method='bounded')
    fits['common'] = dict(par=[float(r.x), float(r.x), 1.0], loglik=-float(r.fun))
    r = optimize.minimize(lambda q: -loglik([q[0], q[1], 0.0], y, ia, ib, cov), x0=[1.2, 1.2], method='Nelder-Mead')
    fits['independent'] = dict(par=[float(r.x[0]), float(r.x[1]), 0.0], loglik=-float(r.fun))
    upper = act.max(0)
    rng = np.random.default_rng(20261024); m = 200000; fc_rep = fc[rng.integers(0, n, m)]
    rows = {'records': dict(one.describe(e, None), **cross_coamplification(e, ia, ib))}
    for name, f in fits.items():
        sim = np.clip(fc_rep+simulate(mu, cov, ia, ib, f['par'], m, rng), 0, upper)-fc_rep
        rows[name] = dict(one.describe(sim, None), **cross_coamplification(sim, ia, ib))
    dist_ab = float(one.distance_km(sites[ia[0]]['lat'], sites[ia[0]]['lon'], sites[ib[0]]['lat'], sites[ib[0]]['lon']))
    print(f'records {n}, sites A {[s["key"] for s in sites if s["region"] == "A"]}, B {[s["key"] for s in sites if s["region"] == "B"]}, '
          f'distance between region centres about {dist_ab:.0f} km, observed max output {np.round(upper, 3)}')
    for name, f in fits.items():
        print(f'{name:12s} log-likelihood {f["loglik"]:.1f}  sigma_A {f["par"][0]:.3f} sigma_B {f["par"][1]:.3f} rho {f["par"][2]:.3f}')
    keys = list(rows['records'])
    print('quantity'.ljust(16)+''.join(k.rjust(13) for k in rows))
    for k in keys:
        print(k.ljust(16)+''.join(f'{rows[r_][k]:13.3f}' for r_ in rows))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'fit_HA4_0.4_0.6.json').write_text(json.dumps(dict(
        sites=sites, band=BAND, forecast='HA4', records=n, region_A=ia.tolist(), region_B=ib.tolist(), mean_error=mu.tolist(),
        covariance=cov.tolist(), fits=fits, observed_max_output=upper.tolist(), comparison=rows, distance_km=dist_ab), indent=1))


if __name__ == '__main__':
    main()
