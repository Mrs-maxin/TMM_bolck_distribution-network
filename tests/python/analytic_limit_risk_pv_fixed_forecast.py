"""Analytical limit-violation probability for the fixed-forecast PV case (plan section 14).

Voltage-magnitude squares and sending-end branch active powers are quadratic in the rectangular
voltage. Given a mixture component and a fluctuation strength the voltage is Gaussian, so each
quadratic has a closed-form conditional characteristic function; the strength is integrated by
generalized Gauss-Laguerre quadrature and the CDF follows from Gil-Pelaez inversion.
"""
import importlib.util, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import roots_genlaguerre
from numpy.polynomial.legendre import leggauss

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('run', REPO/'tests/python/run_pv_fixed_forecast_case33.py')
run = importlib.util.module_from_spec(spec); sys.dont_write_bytecode = True; spec.loader.exec_module(run)
O = REPO/'result/case33bw/pv_fixed_forecast_case33_20261021'; SCENE = 'common_strength'; D = O/SCENE
KS = [1, 3, 6, 12]; STRENGTH_NODES = 40
fit = np.load(D/'fitting_inputs.npz'); center, T = fit['center'], fit['basis']*fit['scale']
selected = pd.concat([pd.read_csv(D/'selected_models.csv'), pd.read_csv(D/'selected_models_block.csv')], ignore_index=True)

# Quantity j: 0..31 squared voltage magnitude of buses 2..33; 32..63 sending-end P of branches 1..32 (MW);
# 64 total active power loss (MW).
def ac_quantities(v):
    out = run.evaluate_outputs(v, run.BASE, run.BUS, run.BRANCH)
    return np.c_[abs(v[:, 1:])**2, out[:, :32], out[:, -1]]


def quantities(x):
    return ac_quantities(run.voltage(np.atleast_2d(x)))

d = 64; eye = np.eye(d)
c0 = quantities(np.zeros(d))[0]; plus = quantities(eye); minus = quantities(-eye)
b = (plus-minus)/2; diag = (plus+minus)/2-c0
NQ = 65
A = np.zeros((NQ, d, d)); A[:, np.arange(d), np.arange(d)] = diag.T
for i in range(d):
    pair = quantities(eye[i]+eye[i+1:]) if i+1 < d else np.zeros((0, NQ))
    off = (pair-plus[i]-plus[i+1:]+c0)/2
    A[:, i, i+1:] = off.T; A[:, i+1:, i] = off.T
probe = center+np.random.default_rng(1).normal(scale=.02, size=(50, d))
direct = quantities(probe); rebuilt = np.einsum('ni,qij,nj->nq', probe, A, probe)+probe@b+c0
quadratic_check = float(abs(direct-rebuilt).max())
# Fitting coordinates: x = center + T xi.
At = np.einsum('ia,qij,jb->qab', T, A, T); bt = 2*np.einsum('ia,qij,j->qa', T, A, center)+b.T@T
ct = np.einsum('i,qij,j->q', center, A, center)+b.T@center+c0


def strength_nodes(nu):
    if not np.isfinite(nu) or nu > 1e3:
        return np.array([1.]), np.array([1.])
    if nu > 300:
        # Laguerre weights overflow for very large shape; the strength is then tightly concentrated.
        from scipy.special import gammaincinv
        p, w = leggauss(STRENGTH_NODES); p = (p+1)/2
        return 1/(2*gammaincinv(nu/2, p)/nu), w/w.sum()
    u, w = roots_genlaguerre(STRENGTH_NODES, nu/2-1)
    return nu/(2*u), w/w.sum()


def component_terms(model, family, q):
    """Weighted (m0, lambda, beta) of every component-strength node for quantity q."""
    terms = []
    for k, pi in enumerate(model['weights']):
        if family == 'BTMM':
            Q1, Q2 = model['Q1'], model['Q2']; m, a, G = model['means'][k], model['intercept'][k], model['coefficient'][k]
            mu = Q1@m+Q2@(a+G@m); S, Om = model['scatter'][k], model['residual'][k]
            cov = lambda s: np.c_[Q1, Q2]@np.block([[s*S, s*S@G.T], [s*G@S, s*G@S@G.T+Om]])@np.c_[Q1, Q2].T
        else:
            mu = model['means'][k]; base = model['scatter'][k]; cov = lambda s, base=base: s*base
        nodes, weights = strength_nodes(model['df'][k]) if family != 'GMM' else (np.array([1.]), np.array([1.]))
        lin = 2*At[q]@mu+bt[q]; m0 = mu@At[q]@mu+bt[q]@mu+ct[q]
        for s, w in zip(nodes, weights):
            vals, vecs = np.linalg.eigh(cov(s)); L = vecs*np.sqrt(np.clip(vals, 0, None))
            lam, U = np.linalg.eigh(L.T@At[q]@L); beta = U.T@(L.T@lin)
            terms.append((pi*w, m0, lam, beta))
    return terms


def cdf_at(terms, thresholds):
    w = np.array([t[0] for t in terms]); m0 = np.array([t[1] for t in terms])
    lam = np.array([t[2] for t in terms]); beta = np.array([t[3] for t in terms])
    mean_k = m0+lam.sum(1); var_k = 2*(lam**2).sum(1)+(beta**2).sum(1)
    mean = w@mean_k; sd = np.sqrt(w@(var_k+mean_k**2)-mean**2)
    keep = w > 1e-14; tmax = 12/np.sqrt(var_k[keep]/sd**2).min()
    nodes, weights = leggauss(64); edges = np.linspace(0, tmax, int(np.ceil(tmax*4))+2)
    tau = ((edges[1:]-edges[:-1])[:, None]*(nodes+1)/2+edges[:-1, None]).ravel()
    wt = ((edges[1:]-edges[:-1])[:, None]*weights/2).ravel()
    t = tau/sd; phi = np.zeros(len(t), complex)
    for j in np.flatnonzero(keep):
        z = 1-2j*np.outer(t, lam[j])
        phi += w[j]*np.exp(1j*t*(m0[j]-mean)-.5*np.log(z).sum(1)-(np.outer(t**2, beta[j]**2)/(2*z)).sum(1))
    y = (np.asarray(thresholds)-mean)/sd
    integrand = np.imag(np.exp(-1j*np.outer(y, tau))*phi)/tau
    return .5-(integrand@wt)/np.pi, dict(mean=mean, sd=sd, tmax=float(tmax), tail_cf=float(abs(phi[-1])))


def main():
    EVENTS = [(f'V{n} < 0.95', n-2, .95**2, 'below') for n in range(2, 34)]
    EVENTS += [(f'V{n} > 1.05', n-2, 1.05**2, 'above') for n in range(2, 34)]
    EVENTS += [('V29 < 0.96', 27, .96**2, 'below'), ('V31 < 0.955', 29, .955**2, 'below'),
               ('V18 > 1.07', 16, 1.07**2, 'above'), ('V18 > 1.08', 16, 1.08**2, 'above'), ('V18 > 1.09', 16, 1.09**2, 'above'),
               ('P19 > 0 MW', 32+18, 0., 'above'), ('P31 > 0 MW', 32+30, 0., 'above')]
    ref = np.load(D/'reference_AC.npz')['voltage']; ref_q = ac_quantities(ref)
    rows = [dict(K='AC', family='exact', **{e: float(np.mean(ref_q[:, q] < th) if s == 'below' else np.mean(ref_q[:, q] > th))
                                             for e, q, th, s in EVENTS})]
    checks = []
    for k in KS:
        for family in ['GMM', 'TMM', 'BTMM']:
            cache = D/'analytic_cache'/f'{family}_K{k}.json'
            if cache.exists():
                saved = json.loads(cache.read_text()); rows.append(saved['row']); checks += saved['checks']; continue
            before = len(checks)
            r = selected[(selected.K == k) & (selected.family == family)].iloc[0]
            model = dict(np.load(D/'candidates'/(r.candidate+'.npz')))
            seed = run.CONFIG['propagation_seed_base']+100000*run.SCENES.index(SCENE)+1000*k
            z, wz = run.propagate_any(np.load(D/'candidates'/(r.candidate+'.npz')), run.CONFIG['propagation_points'], seed)
            xq = quantities(center+z@T.T)
            row = dict(K=k, family=family)
            for q in sorted({e[1] for e in EVENTS}):
                items = [e for e in EVENTS if e[1] == q]
                F, info = cdf_at(component_terms(model, family, q), [e[2] for e in items])
                for (name, _, th, side), f in zip(items, F):
                    row[name] = float(f if side == 'below' else 1-f)
                    numeric = float(wz[xq[:, q] < th].sum() if side == 'below' else wz[xq[:, q] > th].sum())
                    checks.append(dict(K=k, family=family, event=name, analytic=row[name], numerical_integration=numeric,
                                       difference=row[name]-numeric, tail_cf=info['tail_cf']))
            rows.append(row); print('done', k, family, flush=True)
            cache.parent.mkdir(exist_ok=True); cache.write_text(json.dumps(dict(row=row, checks=checks[before:])))
    table = pd.DataFrame(rows); table.to_csv(D/'analytic_limit_risk.csv', index=False)
    check = pd.DataFrame(checks); check.to_csv(D/'analytic_limit_risk_check.csv', index=False)
    json.dump(dict(quadratic_reconstruction_max_error=quadratic_check, strength_nodes=STRENGTH_NODES,
                   max_abs_difference_analytic_vs_numerical=float(check.difference.abs().max()),
                   max_tail_cf=float(check.tail_cf.max())), open(D/'analytic_limit_risk_summary.json', 'w'), indent=1)
    print('quadratic check', quadratic_check, 'max |analytic-numerical|', check.difference.abs().max())


if __name__ == '__main__':
    main()
