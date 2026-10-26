"""Case 1: moment-matched TMM against maximum-likelihood TMM at small K.

Component assignment comes from the AQGMM fit at the same K. Each component keeps the weighted mean
and covariance of the scenarios (as GMM does, so the variance is preserved); the degree of freedom
follows the weighted multivariate kurtosis b = E[delta^2]/(r(r+2)) = (nu-2)/(nu-4), nu = (4b-2)/(b-1)
(Gaussian when b <= 1). Standard TMM uses all 64 directions; block TMM uses only the amplified block
(derivation.md section J split) and a weighted regression of the other directions (own or shared
residual). Evaluated against the MC method (32768 scenarios) with the tail-band metric of
tests/extreme_weather_case33.py.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import sys, json
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import run_pv_nrel_case33 as case
base = case.base
from cloud_numeric_core import log_density, floor_matrix, FLOOR

D = base.REPO/'result/case33bw/pv_nrel_case33_newsplit_20261022/nrel_ha4_40_60'
OUT = D/'moment_matched'; OUT.mkdir(exist_ok=True)
KS = [1, 2, 3, 4, 6]
DRAWS = 262144
LEVELS = np.linspace(.001, .05, 50)


def nu_from_kurtosis(delta, w, r):
    b = (w@delta**2)/(w.sum()*r*(r+2))
    return np.inf if b <= 1+1e-6 else max(4.1, (4*b-2)/(b-1))


def moment_models(x, gmm, Q1, Q2):
    resp = np.exp(log_density(x, gmm)[1]) if False else None
    dens, r_, _ = log_density(x, gmm); resp = r_
    n, d = x.shape; k = len(gmm['weights']); U, W = x@Q1, x@Q2; r = U.shape[1]; design = np.c_[np.ones(n), U]
    std = dict(weights=resp.sum(0)/n, means=[], scatter=[], df=[])
    blk = dict(weights=resp.sum(0)/n, means=[], scatter=[], df=[], intercept=[], coefficient=[], residual=[])
    omega = np.zeros((W.shape[1], W.shape[1]))
    for j in range(k):
        w = resp[:, j]; m = w@x/w.sum(); c = ((x-m).T@((x-m)*w[:, None]))/w.sum()
        delta = np.einsum('ij,ij->i', (x-m)@np.linalg.inv(c), x-m); nu = nu_from_kurtosis(delta, w, d)
        std['means'].append(m); std['scatter'].append(c*((nu-2)/nu if np.isfinite(nu) else 1)); std['df'].append(nu)
        mu = w@U/w.sum(); cu = ((U-mu).T@((U-mu)*w[:, None]))/w.sum()
        du = np.einsum('ij,ij->i', (U-mu)@np.linalg.inv(cu), U-mu); nuu = nu_from_kurtosis(du, w, r)
        blk['means'].append(mu); blk['scatter'].append(floor_matrix(cu*((nuu-2)/nuu if np.isfinite(nuu) else 1), FLOOR)[0]); blk['df'].append(nuu)
        b = np.linalg.solve(design.T@(design*w[:, None]), design.T@(W*w[:, None])); e = W-design@b
        blk['intercept'].append(b[0]); blk['coefficient'].append(b[1:].T)
        res = (e.T@(e*w[:, None]))/w.sum(); blk['residual'].append(floor_matrix(res)[0]); omega += (e.T@(e*w[:, None]))/n
    std = {kk: np.array(v) for kk, v in std.items()}; blk = {kk: np.array(v) for kk, v in blk.items()}
    blk.update(Q1=Q1, Q2=Q2, block=np.array(True))
    shared = {kk: np.array(v, copy=True) for kk, v in blk.items()}; shared['residual'][:] = floor_matrix(omega)[0]
    return std, blk, shared


def mst_moment_projection(x, gmm, scale):
    """AQPPF_TMM main method (fit_moment_projected_mst.m): GMM responsibilities, weighted mean and covariance in
    physical units, eigen-directions of each component covariance, nu = 4 + 6/excess kurtosis per direction."""
    _, resp, _ = log_density(x, gmm); c = scale/scale.max(); y = x*c; n = len(x); comps = []
    for j in range(resp.shape[1]):
        w = resp[:, j]/resp[:, j].sum(); m = w@y; e = y-m; C = (e.T@(e*w[:, None]))
        lam, D = np.linalg.eigh(C); lam = np.clip(lam, 0, None); p = e@D
        kurt = np.where(lam > 0, (w@(p**4))/np.where(lam > 0, lam**2, 1), 3.); ex = np.maximum(0, kurt-3)
        nu = np.where(ex > 0, 4+6/np.where(ex > 0, ex, 1), np.inf); a = lam*np.where(np.isfinite(nu), (nu-2)/nu, 1.)
        comps.append(dict(weight=resp[:, j].sum()/n, mean=m, D=D, a=a, nu=nu))
    return dict(mst=comps, c=c, df=np.concatenate([cp['nu'] for cp in comps]))


def sample_mst(model, n, rng):
    comps = model['mst']; k = rng.choice(len(comps), size=n, p=[cp['weight'] for cp in comps]); d = len(comps[0]['mean'])
    y = np.empty((n, d))
    for j, cp in enumerate(comps):
        m = k == j; nm = m.sum(); g = rng.standard_normal((nm, d))
        s = np.where(np.isfinite(cp['nu'])[None, :], np.where(np.isfinite(cp['nu']), cp['nu'], 1)[None, :]/rng.chisquare(np.where(np.isfinite(cp['nu']), cp['nu'], 1)[None, :].repeat(nm, 0)), 1.)
        y[m] = cp['mean']+(g*np.sqrt(cp['a']*s))@cp['D'].T
    return y/model['c'], np.full(n, 1/n)


def evaluate(model, fit_in, ref, targets, seed):
    z, w = sample_mst(model, DRAWS, np.random.default_rng(seed)) if 'mst' in model else base.propagate_any(model, DRAWS, seed)
    y = np.concatenate([base.outputs(base.voltage((zz*fit_in['scale'])@fit_in['basis'].T+fit_in['center'])) for zz in np.array_split(z, 16)])
    out = {}
    for _, t in targets.iterrows():
        j = int(t.object_index); r = np.sort(ref[:, j]); o = np.argsort(y[:, j]); ys, cw = y[o, j], np.cumsum(w[o])/w.sum()
        F = lambda v: np.interp(v, ys, cw, left=0., right=1.)
        q = r[(LEVELS*len(r)).astype(int)] if t.side == 'lower' else r[((1-LEVELS)*len(r)).astype(int)]
        err = (F(q)-LEVELS) if t.side == 'lower' else ((1-F(q))-LEVELS)
        out[f"{t.output}{int(t.node) if t.output == 'V' else int(t.branch)}"] = float(np.mean(abs(err)))
    ks = []
    for j in range(64):
        if j in set(targets.object_index):
            continue
        r = np.sort(ref[:, j]); o = np.argsort(y[:, j]); ys, cw = y[o, j], np.cumsum(w[o])/w.sum()
        grid = r[np.linspace(0, len(r)-1, 1001).astype(int)]; ks.append(np.max(abs(np.interp(grid, ys, cw, left=0., right=1.)-np.linspace(0, 1, 1001))))
    out['其余对象KS'] = float(np.median(ks)); return out


def main():
    fit_in = np.load(D/'fitting_inputs.npz'); x, xv = fit_in['x'], fit_in['validation']
    sp = np.load(D/'block_subspace.npz'); Q1, Q2 = sp['Q1'], sp['Q2']
    ref = np.load(D/'reference_outputs.npz')['values']; targets = pd.read_csv(D/'targets.csv')
    sel = pd.concat([pd.read_csv(D/f) for f in ['selected_models.csv', 'selected_models_block.csv', 'selected_models_shared.csv']])
    rows = []
    for k in KS:
        g = sel[(sel.family == 'GMM') & (sel.K == k)].candidate.iloc[0]; gmm = dict(np.load(D/'candidates'/(g+'.npz')))
        std, blk, shr = moment_models(x, gmm, Q1, Q2)
        cands = [('AQGMM', gmm), ('标准TMM（最大似然）', dict(np.load(D/'candidates'/(sel[(sel.family == 'TMM') & (sel.K == k)].candidate.iloc[0]+'.npz')))),
                 ('分块TMM（最大似然）', dict(np.load(D/'candidates'/(sel[(sel.family == 'BTMM') & (sel.K == k)].candidate.iloc[0]+'.npz')))),
                 ('共用残差（最大似然）', dict(np.load(D/'candidates'/(sel[(sel.family == 'SBTMM') & (sel.K == k)].candidate.iloc[0]+'.npz')))),
                 ('标准TMM（按矩）', std), ('分块TMM（按矩）', blk), ('共用残差（按矩）', shr),
                 ('多尺度t（矩投影，AQPPF_TMM）', mst_moment_projection(x, gmm, fit_in['scale']))]
        for name, m in cands:
            dfv = np.asarray(m['df'], float)
            row = dict(K=k, 模型=name, 自由度=[] if 'AQGMM' in name else (np.round([np.nanmin(dfv), np.nanmedian(dfv)], 1).tolist() if 'mst' in m else np.round(dfv, 1).tolist()))
            row.update(evaluate(m, fit_in, ref, targets, 2026102311+k)); rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    t = pd.DataFrame(rows)
    t['电压4个中位数'] = t[['V30', 'V33', 'V18', 'V13']].median(axis=1); t['支路4个中位数'] = t[['P32', 'P16', 'P17', 'P19']].median(axis=1)
    t.to_csv(OUT/'moment_vs_likelihood.csv', index=False)
    pd.set_option('display.width', 250); print(t.drop(columns=['自由度']).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
