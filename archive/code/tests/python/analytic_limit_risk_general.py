"""Analytical limit-violation probability for any case of run_pv_fixed_forecast.py (plan sections 14 and 16.1).

Same route as analytic_limit_risk_pv_fixed_forecast.py: squared voltage magnitudes and sending-end
branch active powers are quadratic in the rectangular voltage; given a mixture component and a
fluctuation strength the voltage is Gaussian, so each quadratic has a closed-form conditional
characteristic function; the strength is integrated by generalized Gauss-Laguerre quadrature and the
probability follows from Gil-Pelaez inversion. Only the objects of limit_risk_events.csv are built,
which keeps d = 280 tractable. A squared magnitude has rank 2 and a branch power rank at most 4 in the
voltage coordinates, so eigenvalues below a relative 1e-10 are merged into one Gaussian factor.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import importlib.util, sys, json, time, argparse, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import roots_genlaguerre, gammaincinv
from numpy.polynomial.legendre import leggauss

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('run', REPO/'tests/python/run_pv_fixed_forecast.py')
run = importlib.util.module_from_spec(spec); sys.dont_write_bytecode = True; spec.loader.exec_module(run)
STRENGTH_NODES = 40
RANK_TOLERANCE = 1e-10


def strength_nodes(nu):
    if not np.isfinite(nu) or nu > 1e3:
        return np.array([1.]), np.array([1.])
    if nu > 300:
        # Laguerre weights overflow for very large shape; the strength is then tightly concentrated.
        p, w = leggauss(STRENGTH_NODES); p = (p+1)/2
        return 1/(2*gammaincinv(nu/2, p)/nu), w/w.sum()
    u, w = roots_genlaguerre(STRENGTH_NODES, nu/2-1)
    return nu/(2*u), w/w.sum()


class Quadratics:
    """q(x) = x' A x + b' x + c for the selected quantities, x the rectangular non-slack voltage."""

    def __init__(self, columns):
        self.columns = np.asarray(columns)
        d = run.D; eye = np.eye(d)
        c0 = self.direct(np.zeros(d))[0]; plus = self.direct(eye); minus = self.direct(-eye)
        b = (plus-minus)/2; diag = (plus+minus)/2-c0
        nq = len(columns); A = np.zeros((nq, d, d)); A[:, np.arange(d), np.arange(d)] = diag.T
        for i in range(d-1):
            pair = self.direct(eye[i]+eye[i+1:])
            off = (pair-plus[i]-plus[i+1:]+c0)/2
            A[:, i, i+1:] = off.T; A[:, i+1:, i] = off.T
        self.A, self.b, self.c = A, b.T, c0

    def direct(self, x):
        v = run.voltage(np.atleast_2d(x))
        out = run.evaluate_outputs(v, run.BASE, run.BUS, run.BRANCH)
        full = np.c_[abs(v[:, 1:])**2, out[:, :run.NBR]]
        return full[:, self.columns]

    def check(self, center):
        probe = center+np.random.default_rng(1).normal(scale=.02, size=(50, run.D))
        rebuilt = np.einsum('ni,qij,nj->nq', probe, self.A, probe)+probe@self.b.T+self.c
        return float(abs(self.direct(probe)-rebuilt).max())


def component_terms(model, family, A, b, c):
    """Per quantity, the list of (weight, m0, lam, beta, gaussian_var) over components and strength nodes."""
    nq = len(A); terms = [[] for _ in range(nq)]
    for k, pi in enumerate(model['weights']):
        if family == 'BTMM':
            Q1, Q2 = model['Q1'], model['Q2']; m, a, G = model['means'][k], model['intercept'][k], model['coefficient'][k]
            mu = Q1@m+Q2@(a+G@m); S, Om = model['scatter'][k], model['residual'][k]; Qb = np.c_[Q1, Q2]
            cov = lambda s: Qb@np.block([[s*S, s*S@G.T], [s*G@S, s*G@S@G.T+Om]])@Qb.T
        else:
            mu = model['means'][k]; base = model['scatter'][k]; cov = lambda s, base=base: s*base
        nodes, weights = strength_nodes(model['df'][k]) if family != 'GMM' else (np.array([1.]), np.array([1.]))
        lin = [2*A[q]@mu+b[q] for q in range(nq)]; m0 = [mu@A[q]@mu+b[q]@mu+c[q] for q in range(nq)]
        for s, w in zip(nodes, weights):
            vals, vecs = np.linalg.eigh(cov(s)); L = vecs*np.sqrt(np.clip(vals, 0, None))
            for q in range(nq):
                lam, U = np.linalg.eigh(L.T@A[q]@L); beta = U.T@(L.T@lin[q])
                big = abs(lam) > RANK_TOLERANCE*max(abs(lam).max(), 1e-300)
                terms[q].append((pi*w, m0[q]+lam[~big].sum(), lam[big], beta[big],
                                 float((beta[~big]**2).sum()+2*(lam[~big]**2).sum())))
    return terms


def cdf_at(terms, thresholds):
    w = np.array([t[0] for t in terms]); m0 = np.array([t[1] for t in terms])
    mean_k = np.array([t[1]+t[2].sum() for t in terms])
    var_k = np.array([2*(t[2]**2).sum()+(t[3]**2).sum()+t[4] for t in terms])
    mean = w@mean_k; sd = np.sqrt(w@(var_k+mean_k**2)-mean**2)
    keep = w > 1e-14; tmax = 12/np.sqrt(var_k[keep]/sd**2).min()
    nodes, weights = leggauss(64); edges = np.linspace(0, tmax, int(np.ceil(tmax*4))+2)
    tau = ((edges[1:]-edges[:-1])[:, None]*(nodes+1)/2+edges[:-1, None]).ravel()
    wt = ((edges[1:]-edges[:-1])[:, None]*weights/2).ravel()
    t = tau/sd; phi = np.zeros(len(t), complex)
    for j in np.flatnonzero(keep):
        _, _, lam, beta, gv = terms[j]
        z = 1-2j*np.outer(t, lam)
        phi += w[j]*np.exp(1j*t*(m0[j]-mean)-.5*np.log(z).sum(1)-(np.outer(t**2, beta**2)/(2*z)).sum(1)-.5*gv*t**2)
    y = (np.asarray(thresholds)-mean)/sd
    integrand = np.imag(np.exp(-1j*np.outer(y, tau))*phi)/tau
    return .5-(integrand@wt)/np.pi, dict(mean=mean, sd=sd, tmax=float(tmax), tail_cf=float(abs(phi[-1])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=sorted(run.CASES), default='case141')
    parser.add_argument('--scene', default='common_strength')
    parser.add_argument('--out', help='result directory; default from the case table')
    parser.add_argument('--with-K12', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run.setup_case(args.case); run.apply_mode(args.smoke, args.with_K12)
    O = Path(args.out or REPO/run.CONFIG['default_out'].format(scene=args.scene)); D = O/args.scene
    start = time.perf_counter()
    events = pd.read_csv(D/'limit_risk_events.csv')
    fit = np.load(D/'fitting_inputs.npz'); center, T = fit['center'], fit['basis']*fit['scale']
    selected = pd.concat([pd.read_csv(D/'selected_models.csv'), pd.read_csv(D/'selected_models_block.csv')], ignore_index=True)
    objects = sorted(set(events.object_index.astype(int)))
    def column(j):
        o = run.OBJECTS[j]
        return o['node']-2 if o['output'] == 'V' else run.NV+int(np.flatnonzero(run.ACTIVE == o['branch']-1)[0])
    quad = Quadratics([column(j) for j in objects]); quadratic_check = quad.check(center)
    # Fitting coordinates: x = center + T xi.
    A, b, c = quad.A, quad.b, quad.c
    At = np.einsum('ia,qij,jb->qab', T, A, T); bt = 2*np.einsum('ia,qij,j->qa', T, A, center)+b@T
    ct = np.einsum('i,qij,j->q', center, A, center)+b@center+c
    def level(e):
        return e.threshold**2 if e.output == 'V' else e.threshold
    ref = np.load(D/'reference_AC.npz')['voltage']; ref_q = quad.direct(np.c_[ref[:, 1:].real, ref[:, 1:].imag])
    rows = [dict(K='AC', family='exact', **{e.event: float(np.mean(ref_q[:, objects.index(e.object_index)] < level(e)) if e.side == 'below'
                                                   else np.mean(ref_q[:, objects.index(e.object_index)] > level(e)))
                                             for e in events.itertuples()})]
    checks, timing = [], []
    for k in run.KS:
        for family in ['GMM', 'TMM', 'BTMM']:
            pick = selected[(selected.K == k) & (selected.family == family)]
            if pick.empty:
                continue
            clock = time.perf_counter(); r = pick.iloc[0]
            model = dict(np.load(D/'candidates'/(r.candidate+'.npz')))
            seed = run.CONFIG['propagation_seed_base']+100000*run.SCENES.index(args.scene)+1000*k
            z, wz = run.propagate_any(np.load(D/'candidates'/(r.candidate+'.npz')), run.CONFIG['propagation_points'], seed)
            xq = np.concatenate([quad.direct(center+zz@T.T) for zz in np.array_split(z, 16)])
            del z
            terms = component_terms(model, family, At, bt, ct)
            row = dict(K=k, family=family)
            for qi, j in enumerate(objects):
                items = events[events.object_index == j]
                F, info = cdf_at(terms[qi], [level(e) for e in items.itertuples()])
                for e, f in zip(items.itertuples(), F):
                    row[e.event] = float(f if e.side == 'below' else 1-f)
                    numeric = float(wz[xq[:, qi] < level(e)].sum() if e.side == 'below' else wz[xq[:, qi] > level(e)].sum())
                    checks.append(dict(K=k, family=family, event=e.event, analytic=row[e.event], numerical_integration=numeric,
                                       difference=row[e.event]-numeric, tail_cf=info['tail_cf']))
            rows.append(row); timing.append(dict(K=k, family=family, seconds=time.perf_counter()-clock))
            print('done', k, family, round(time.perf_counter()-clock, 1), flush=True)
    table = pd.DataFrame(rows); table.to_csv(D/'analytic_limit_risk.csv', index=False)
    check = pd.DataFrame(checks); check.to_csv(D/'analytic_limit_risk_check.csv', index=False)
    summary = dict(case=args.case, scene=args.scene, objects=[run.OBJECTS[j] for j in objects],
                   quadratic_reconstruction_max_error=quadratic_check, strength_nodes=STRENGTH_NODES,
                   eigenvalue_merge_relative_tolerance=RANK_TOLERANCE, reference_scenarios=len(ref),
                   max_abs_difference_analytic_vs_numerical=float(check.difference.abs().max()),
                   max_tail_cf=float(check.tail_cf.max()), timing_seconds=timing,
                   total_seconds=time.perf_counter()-start,
                   script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with open(D/'analytic_limit_risk_summary.json', 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    print('quadratic check', quadratic_check, 'max |analytic-numerical|', check.difference.abs().max())


if __name__ == '__main__':
    main()
