"""Check a general rule for splitting the voltage into weather-amplified and ordinary directions.

Uses only the source-load scenarios, their voltages from the power-to-voltage step and one
deterministic power flow at the mean injection (anchor). No model fitting and no extra AC.
  1  amplified injections: for every pair of injections, co-amplification
     c_ij = corr(e_i^2, e_j^2) - corr(e_i, e_j)^2 (about 0 when the deviations are jointly Gaussian,
     positive when a common strength enlarges both). Threshold = largest c_ij of Gaussian scenarios with
     the same covariance and count. Injections linked above the threshold form one weather group
     (connected components); a group needs at least two injections.
  2  voltage directions: derivation.md (B8)-(B9), H_g = J0^{-1} L_{W_g}, whitened and orthonormalized.
  3  leak check: remaining voltage directions that are non-Gaussian (fourth-order test above the
     Gaussian range), carry at least LEAK_SHARE of the voltage variance and grow together with the
     amplified block (corr(w^2, |u|^2) above its Gaussian range) join the amplified block.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import sys, json
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import kurtosis
from scipy.sparse.csgraph import connected_components
import run_pv_nrel_case33 as case
base = case.base
from cloud_numeric_core import rect_jacobian

REPO = base.REPO; VAL = REPO/'result/case33bw'; OUT = VAL/'block_split_rule_check_20261022'; OUT.mkdir(exist_ok=True)
SCENES = ['pv_fixed_forecast_case33_20261021/common_strength', 'pv_fixed_forecast_case33_20261021_gaussian_control/gaussian_control',
          'pv_strong_weather_case33_20261021/strong_weather', 'pv_strong_weather_nominal_load_case33_20261021/strong_weather_nominal_load',
          'pv_two_cloud_regions_case33_20261021/two_cloud_regions', 'pv_nrel_case33_20261022/nrel_ha4_40_60']
NAMES = np.array([f'P{b}' for b in range(2, 34)]+[f'Q{b}' for b in range(2, 34)])
NULL_REPEATS = 5
LEAK_SHARE = 1e-3
WEATHER_LOAD_BUSES = [7, 8, 24, 25, 29, 30, 31, 32]


def coamplification(e):
    e = e-e.mean(0); rho = np.corrcoef(e.T); sq = np.corrcoef((e**2).T); c = sq-rho**2; np.fill_diagonal(c, 0); return c


def fobi(y):
    y = y-y.mean(0); r2 = (y*y).sum(1); lam, U = np.linalg.eigh((y*r2[:, None]).T@y/len(y)); o = np.argsort(-lam)
    return lam[o], U[:, o]


def weather_groups(target, rng):
    e = target-target.mean(0); keep = np.flatnonzero(e.std(0) > 1e-12); e = e[:, keep]; C = np.cov(e.T)
    c = coamplification(e)
    null = max(coamplification(rng.multivariate_normal(np.zeros(len(C)), C, size=len(e))).max() for _ in range(NULL_REPEATS))
    n, label = connected_components(c > null, directed=False)
    groups = [keep[label == g] for g in range(n) if np.sum(label == g) >= 2]
    groups.sort(key=lambda g: -len(g))
    return groups, float(null), c, keep


def split(target, x_white, basis, scale, rng, label, group_target=None, return_q=False, v0=None):
    groups, null, c, keep = weather_groups(target if group_target is None else group_target, rng)
    pos = {k: i for i, k in enumerate(keep)}
    if v0 is None:
        v0, _ = base.polar_ac(target.mean(0), base.Y, tolerance=1e-11)
    Ji = np.linalg.inv(rect_jacobian(v0, base.Y))
    Wm = basis.T/scale[:, None]; total = np.sum(scale**2)
    def phys_share(q):
        return float(np.sum((scale[:, None]*q)**2)/total)
    dim = x_white.shape[1]; q1 = np.zeros((dim, 0)); group_cols = []
    for g in groups:
        h = Wm@Ji[:, g]; h = h-q1@(q1.T@h); u, s, _ = np.linalg.svd(h, full_matrices=False)
        new = u[:, :int(np.sum(s > s[0]*1e-10))]; group_cols.append(list(range(q1.shape[1], q1.shape[1]+new.shape[1]))); q1 = np.c_[q1, new]
    r = q1.shape[1]
    q2 = np.linalg.svd(np.eye(dim)-q1@q1.T)[0][:, :dim-r]
    w = x_white@q2; leak = np.array([], int); shares = np.array([])
    if r and q2.shape[1] == 0:
        Q1, Q2 = q1, q2
        row = dict(scene=label, scenarios=len(target), coamplification_threshold=round(null, 3), groups=[NAMES[g].tolist() for g in groups],
                   r_first_order=r, voltage_share_first_order=phys_share(q1), leak_directions=0, amplified_voltage_share=1.0,
                   note='every voltage direction amplified: block TMM coincides with standard TMM', group_cols=group_cols)
        return (row, Q1, Q2) if return_q else row
    if r:
        lw, Uw = fobi(w)
        g = rng.multivariate_normal(np.zeros(w.shape[1]), np.cov(w.T), size=len(w)); topw = max(fobi(g)[0].max() for _ in range(1))
        z = w@Uw; unorm = np.sum((x_white@q1)**2, 1)
        co = np.array([np.corrcoef(z[:, j]**2, unorm)[0, 1] for j in range(z.shape[1])])
        gz = g@Uw; co_null = np.max([abs(np.corrcoef(gz[:, j]**2, np.sum(rng.standard_normal((len(g), r))**2, 1))[0, 1]) for j in range(gz.shape[1])])
        dirs = q2@Uw; shares = np.array([phys_share(dirs[:, [j]]) for j in range(dirs.shape[1])])
        leak = np.flatnonzero((lw > topw) & (shares >= LEAK_SHARE) & (co > co_null))
        Q1 = np.c_[q1, dirs[:, leak]]
        gsq = [np.sum((x_white@q1[:, cols])**2, 1) for cols in group_cols]
        for i, j in enumerate(leak):
            best = int(np.argmax([np.corrcoef(z[:, j]**2, s)[0, 1] for s in gsq])); group_cols[best].append(r+i)
    else:
        Q1 = q1
    Q2 = np.linalg.svd(np.eye(dim)-Q1@Q1.T)[0][:, :dim-Q1.shape[1]]
    kb, ka = kurtosis(w, axis=0), kurtosis(x_white@Q2, axis=0)
    within = [float(np.median(c[np.ix_([pos[i] for i in g], [pos[i] for i in g])][~np.eye(len(g), dtype=bool)])) for g in groups]
    row = dict(scene=label, scenarios=len(target), coamplification_threshold=round(null, 3),
               groups=[NAMES[g].tolist() for g in groups], group_median_coamplification=np.round(within, 3).tolist(),
               r_first_order=r, voltage_share_first_order=phys_share(q1), leak_directions=int(len(leak)),
               leak_shares=np.round(shares[leak], 4).tolist() if len(leak) else [], amplified_voltage_share=phys_share(Q1),
               remaining_kurtosis_max_before=float(kb.max()), remaining_kurtosis_max_after=float(ka.max()),
               remaining_kurtosis_median_after=float(np.median(ka)), group_cols=group_cols)
    return (row, Q1, Q2) if return_q else row


def weather_load_scene(rng):
    n = 8192; bus = base.BUS; lp, lq = 0.8*bus[1:, 2], 0.8*bus[1:, 3]; idx = np.array(WEATHER_LOAD_BUSES)-2
    S = np.exp(0.5*rng.standard_normal(n)-0.125)
    z = rng.standard_normal((n, len(idx)))@np.linalg.cholesky(0.6+0.4*np.eye(len(idx))).T
    fp = 1+0.01*rng.standard_normal((n, 32)); fq = 1+0.01*rng.standard_normal((n, 32))
    fp[:, idx] = fq[:, idx] = 1+0.08*np.sqrt(S)[:, None]*z
    pv = np.zeros((n, 32)); pv[:, base.SITES-2] = base.CAP*np.clip(0.5+0.02*rng.standard_normal((n, 8)), 0, 1)
    target = np.c_[(pv-lp*fp)/base.BASE, -(lq*fq)/base.BASE]
    v = np.array([base.polar_ac(t, base.Y, tolerance=1e-11)[0] for t in target])
    states = np.c_[v[:, 1:].real, v[:, 1:].imag]; center = states.mean(0)
    _, sv, vt = np.linalg.svd(states-center, full_matrices=False); scale = sv/np.sqrt(n-1)
    return target, (states-center)@vt.T/scale, vt.T, scale


def main():
    rng = np.random.default_rng(2026102251); rows = []
    for s in SCENES:
        d = VAL/s; f = np.load(d/'fitting_inputs.npz')
        rows.append(split(np.load(d/'training_inputs.npz')['target'], f['x'], f['basis'], f['scale'], rng, s.split('/')[-1]))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    target, x, basis, scale = weather_load_scene(rng)
    rows.append(split(target, x, basis, scale, rng, 'weather_driven_loads_constructed')); print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (OUT/'split_rule_results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
