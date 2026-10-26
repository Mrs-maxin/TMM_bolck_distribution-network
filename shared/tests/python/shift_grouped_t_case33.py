"""Grouped t mixture whose strength also shifts the mean of the weather-driven voltage directions (derivation.md (M1)-(M3)).

u = m_k + gamma_k * (s(U) - 1) + D(U) z: a thicker cloud enlarges the deviation and moves the voltage towards the
PV output-drop side. Strength groups, synchronous truncated strengths, degrees of freedom and the regression of the
remaining directions are those of grouped_strength_tmm_case33.grouped_model.
"""
import numpy as np
from scipy import stats

import grouped_strength_tmm_case33 as gs
from cloud_numeric_core import log_density, floor_matrix, FLOOR


def strength_grid(t):
    if t.gauss:
        return np.ones(t.GRID)
    u = (np.arange(t.GRID)+.5)/t.GRID
    return t.nu/stats.chi2.isf(u*t.pmax, t.nu)/t.mean


def solve_shift(mu3, c, v, kappa):
    if v <= 0 or c <= 0:
        return 0.
    roots = np.roots([kappa-3*v*v, 0., 3*v*c, -mu3])
    real = roots[abs(roots.imag) < 1e-9].real
    bound = np.sqrt(0.9*c/v)
    ok = real[abs(real) <= bound]
    if len(ok):
        return float(ok[np.argmin(abs(ok))])
    return 0.


def shift_grouped_model(x, gmm, Q1, Q2, groups, s_max):
    _, resp, _ = log_density(x, gmm); n = len(x); U, W = x@Q1, x@Q2; design = np.c_[np.ones(n), U]; comps = []
    r = U.shape[1]; gid = np.empty(r, int)
    for i, g in enumerate(groups):
        gid[g] = i
    for j in range(resp.shape[1]):
        w = resp[:, j]; ws = w.sum(); mu = w@U/ws; E = U-mu; C = (E.T@(E*w[:, None]))/ws
        delta = [np.einsum('ij,ij->i', E[:, g]@np.linalg.inv(C[np.ix_(g, g)]), E[:, g]) for g in groups]
        m = [float(w@(d**2)/(ws*len(g)*(len(g)+2))) for d, g in zip(delta, groups)]
        trunc = [gs.TruncatedStrength(v, s) for v, s in zip(m, s_max)]
        grid = np.stack([strength_grid(t) for t in trunc])
        var = ((grid-1)**2).mean(1); kappa = ((grid-1)**3).mean(1)
        mu3 = (w@(E**3))/ws
        gamma = np.array([solve_shift(mu3[i], C[i, i], var[gid[i]], kappa[gid[i]]) for i in range(r)])
        cov_s = (grid-1)@(grid-1).T/grid.shape[1]; pair = np.sqrt(grid)@np.sqrt(grid).T/grid.shape[1]
        same = gid[:, None] == gid[None, :]; shift_scale = 1.
        while True:
            scat = (C-np.outer(gamma, gamma)*cov_s[np.ix_(gid, gid)])/pair[np.ix_(gid, gid)]
            shrink = 1.
            while np.linalg.eigvalsh(scat).min() <= 0 and shrink > 1e-3:
                shrink *= .9; scat = np.where(same, scat, scat*.9)
            if np.linalg.eigvalsh(scat).min() > 0 or shift_scale < 1e-3:
                break
            gamma *= .9; shift_scale *= .9
        b = np.linalg.solve(design.T@(design*w[:, None]), design.T@(W*w[:, None])); e = W-design@b
        comps.append(dict(weight=ws/n, mu=mu, groups=groups, gid=gid, trunc=trunc, nu=[np.inf if t.gauss else t.nu for t in trunc], m=m,
                          shift=gamma, shift_scale=shift_scale, scatter=floor_matrix(scat, FLOOR)[0], cross_shrink=shrink,
                          intercept=b[0], coefficient=b[1:].T, residual=floor_matrix((e.T@(e*w[:, None]))/ws)[0]))
    return dict(shifted=comps, Q1=Q1, Q2=Q2)


def sample_shift(model, n, rng):
    comps = model['shifted']; p = np.array([c['weight'] for c in comps]); k = rng.choice(len(comps), size=n, p=p/p.sum())
    out = np.empty((n, model['Q1'].shape[0]))
    for j, c in enumerate(comps):
        sel = k == j; nm = int(sel.sum())
        if nm == 0:
            continue
        u0 = rng.random(nm); S = np.stack([t.draw(nm, rng, u0) for t in c['trunc']], 1)[:, c['gid']]
        z = rng.standard_normal((nm, len(c['mu'])))@np.linalg.cholesky(c['scatter']).T
        Uu = c['mu']+c['shift']*(S-1)+z*np.sqrt(S)
        Wd = c['intercept']+Uu@c['coefficient'].T+rng.standard_normal((nm, len(c['intercept'])))@np.linalg.cholesky(c['residual']).T
        out[sel] = Uu@model['Q1'].T+Wd@model['Q2'].T
    return out, np.full(n, 1/n)


def model_outputs(model, data, seed, draws=262144):
    rng = np.random.default_rng(seed); z, w = sample_shift(model, draws, rng)
    y = np.concatenate([gs.base.outputs(gs.base.voltage((zz*data['scale'])@data['basis'].T+data['center'])) for zz in np.array_split(z, 16)])
    return y, w
