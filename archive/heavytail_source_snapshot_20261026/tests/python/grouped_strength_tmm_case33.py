"""Grouped-strength TMM against AQGMM and the moment-projected multiple-scaled t (case_study_plan_20261022.md section 5.3).

Voltage directions are grouped by the split rule of derivation.md section J; directions in a group share
one fluctuation strength; the remaining directions are a Gaussian regression on the amplified part.
Parameters are moment matched on the AQGMM partition at the same K. Inter-group relations: independent,
synchronous (one uniform drives all groups, grouped t) and common background x regional local strength.
Outputs: summary table at equal K and, at K=PLOT_K, tail PDF and tail CDF plots of the target objects.
Usage: python grouped_strength_tmm_case33.py [scene ...]   scenes: case1 two_cloud shared_background
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
from scipy import stats
from scipy.ndimage import gaussian_filter1d
ARGS = sys.argv[1:] or ['case1', 'two_cloud', 'shared_background']
sys.argv = sys.argv[:1]
import run_pv_nrel_case33 as case
base = case.base
import block_split_rule_check as chk
import moment_matched_tmm_case33 as mm
from cloud_numeric_core import initialize, fit, log_density, floor_matrix, FLOOR
from fast_voltage_map import fast_map

REPO = base.REPO; VAL = REPO/'result/case33bw'; OUT = VAL/'grouped_strength_tmm_20261023'
CACHE = REPO/'cache/grouped_strength_tmm_20261023'
KS = [1, 2, 3, 4]; PLOT_K = 2; DRAWS = 262144; LEVELS = np.linspace(.001, .05, 50); WORKERS = 8
RELATIONS = {'indep': '分组TMM（组间独立）', 'sync': '分组TMM（组间同步）', 'background': '分组TMM（共同背景）'}


def fitting_coordinates(train, val):
    vt_, mt = fast_map(train); vv_, mv = fast_map(val)
    st, sv = np.c_[vt_[:, 1:].real, vt_[:, 1:].imag], np.c_[vv_[:, 1:].real, vv_[:, 1:].imag]
    center = st.mean(0); _, s, v = np.linalg.svd(st-center, full_matrices=False); scale = s/np.sqrt(len(st)-1); basis = v.T
    return dict(x=(st-center)@basis/scale, xv=(sv-center)@basis/scale, basis=basis, scale=scale, center=center, mapping=[mt, mv])


def load_existing(d):
    train, val = np.load(d/'training_inputs.npz')['target'], np.load(d/'validation_inputs.npz')['target']
    every = np.r_[train, val, np.load(d/'reference_inputs.npz')['target']]
    t = pd.read_csv(d/'targets.csv')
    return dict(train=train, every=every, **fitting_coordinates(train, val),
                ref=np.load(d/'reference_outputs.npz')['values'], targets=[(int(r.object_index), r.side) for _, r in t.iterrows()])


def shared_background_inputs(n, rng):
    areas = [np.array([18, 17, 16, 15, 14, 13]), np.array([33, 22])]
    s0 = np.exp(0.6*rng.standard_normal(n)-0.18); pv = np.zeros((n, 32))
    for a in areas:
        s = s0*np.exp(0.5*rng.standard_normal(n)-0.125); z = rng.standard_normal((n, len(a)))@np.linalg.cholesky(0.6+0.4*np.eye(len(a))).T
        pv[:, a-2] = base.CAP*np.clip(0.5+0.10*np.sqrt(s)[:, None]*z, 0, 0.9)
    lp = 0.8*base.BUS[None, 1:, 2]*(1+0.01*rng.standard_normal((n, 32))); lq = 0.8*base.BUS[None, 1:, 3]*(1+0.01*rng.standard_normal((n, 32)))
    return np.c_[(pv-lp)/base.BASE, -lq/base.BASE]


def solve(t):
    return np.array([base.polar_ac(x, base.Y, tolerance=1e-11)[0] for x in t])


def ac(t):
    with Pool(WORKERS) as p:
        return np.concatenate(p.map(solve, np.array_split(t, WORKERS*16)))


def load_constructed():
    rng = np.random.default_rng(2026102301); CACHE.mkdir(parents=True, exist_ok=True)
    train, val, mc = shared_background_inputs(8192, rng), shared_background_inputs(4096, rng), shared_background_inputs(1_000_000, rng)
    path = CACHE/'shared_background_mc_outputs.npy'
    if path.exists():
        ref = np.load(path)
    else:
        vm = ac(mc); ref = np.concatenate([base.outputs(c).astype(np.float32) for c in np.array_split(vm, 64)]); np.save(path, ref)
    ext = []
    for j in range(64):
        r = ref[:, j]; q = np.quantile(r, [.001, .5, .999]); sd = r.std()
        ext += [((q[1]-q[0])/sd, j, 'lower'), ((q[2]-q[1])/sd, j, 'upper')]
    targets = []
    for e, j, s_ in sorted(ext, reverse=True):
        if j not in [t[0] for t in targets] and len(targets) < 6:
            targets.append((j, s_))
    return dict(train=train, every=np.r_[train, val, mc[:32768]], **fitting_coordinates(train, val), ref=ref, targets=targets)


def fit_gmm(x, xv, k, scale):
    best = None
    for p in range(1 if k == 1 else 2):
        ini, _, _ = initialize(x, k, p, scale, population_covariance=True)
        g, _ = fit(x, ini, actual_covariance_floor=True, relative_tolerance=True); vl = float(np.mean(log_density(xv, g)[0]))
        if best is None or vl > best[0]:
            best = (vl, g)
    return best[1]


def ig_nu(m):
    return np.inf if m <= 1+1e-6 else max(4.1, (4*m-2)/(m-1))


def draw_ig(nu, n, rng, uniform=None):
    if not np.isfinite(nu):
        return np.ones(n)
    if uniform is None:
        return (nu-2)/rng.chisquare(nu, n)
    return (nu-2)/stats.chi2.ppf(1-uniform, nu)


class TruncatedStrength:
    """Inverse-gamma strength V = nu/chi2_nu truncated at v_max and normalized to mean one, so that the normalized
    strength never exceeds s_max (the largest physically possible PV deviation over the error standard deviation,
    squared). nu is chosen so that E[S^2] matches the group fourth moment m."""
    GRID = 2000

    def __init__(self, m, s_max):
        self.s_max = s_max; self.gauss = (not np.isfinite(m)) or m <= 1.05
        if self.gauss:
            return
        m = min(m, 0.98*s_max)
        lo, hi = np.log(0.3), np.log(1e4)
        for _ in range(60):
            mid = (lo+hi)/2; self._set(np.exp(mid))
            lo, hi = (mid, hi) if self.ratio > m else (lo, mid)
        self._set(np.exp((lo+hi)/2))
        if not (np.isfinite(self.mean) and self.mean > 0 and np.isfinite(self.ratio)) or abs(self.ratio-m) > 0.05*m:
            self.gauss = self.gauss or m < 1.1
        if not (np.isfinite(self.mean) and self.mean > 0 and np.isfinite(self.ratio)) or abs(self.ratio-m) > 0.05*m:
            self.gauss = self.gauss or m < 1.1

    def _set(self, nu):
        self.nu = nu
        lo, hi = np.log(1e-6), np.log(1e12)
        for _ in range(80):
            mid = (lo+hi)/2; v = np.exp(mid); pmax = stats.chi2.sf(nu/v, nu)
            vals = nu/stats.chi2.isf((np.arange(self.GRID)+.5)/self.GRID*pmax, nu); mean = vals.mean()
            lo, hi = (mid, hi) if v/mean < self.s_max else (lo, mid)
        self.v_max = np.exp((lo+hi)/2); self.pmax = stats.chi2.sf(nu/self.v_max, nu)
        vals = nu/stats.chi2.isf((np.arange(self.GRID)+.5)/self.GRID*self.pmax, nu); self.mean = vals.mean()
        self.ratio = (vals**2).mean()/self.mean**2

    def draw(self, n, rng, uniform=None):
        if self.gauss:
            return np.ones(n)
        u = rng.random(n) if uniform is None else uniform
        return self.nu/stats.chi2.isf(np.clip(u, 1e-12, 1-1e-12)*self.pmax, self.nu)/self.mean


def strengths(comp, relation, n, rng):
    G = len(comp['groups'])
    if 'trunc' in comp:
        u = rng.random(n) if relation == 'sync' else None
        return np.stack([t.draw(n, rng, u) for t in comp['trunc']], 1)
    if G == 1 or relation == 'indep':
        return np.stack([draw_ig(nu, n, rng) for nu in comp['nu']], 1)
    if relation == 'sync':
        u = rng.random(n); return np.stack([draw_ig(nu, n, rng, u) for nu in comp['nu']], 1)
    s0 = draw_ig(comp['nu0'], n, rng); return np.stack([s0*draw_ig(nu, n, rng) for nu in comp['nu_local']], 1)


def profile_nu(delta, w, r):
    """Degree of freedom maximizing the weighted t likelihood of a group with its covariance held fixed:
    scale = C (nu-2)/nu, so the group variance is preserved for every nu."""
    from scipy.special import gammaln
    from scipy.optimize import minimize_scalar
    def nll(t):
        nu = 4.0+np.exp(t)
        ll = gammaln((nu+r)/2)-gammaln(nu/2)-r/2*np.log((nu-2)*np.pi)-(nu+r)/2*np.log1p(delta/(nu-2))
        return -(w@ll)/w.sum()
    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(1e4)), method='bounded')
    gauss = -(w@(-r/2*np.log(2*np.pi)-delta/2))/w.sum()
    return np.inf if gauss <= res.fun else 4.0+np.exp(res.x)


def grouped_model(x, gmm, Q1, Q2, groups, relation, rng, nu_method='moment', s_max=None):
    _, resp, _ = log_density(x, gmm); n = len(x); U, W = x@Q1, x@Q2; design = np.c_[np.ones(n), U]; comps = []
    for j in range(resp.shape[1]):
        w = resp[:, j]; ws = w.sum(); mu = w@U/ws; E = U-mu; C = (E.T@(E*w[:, None]))/ws
        delta = [np.einsum('ij,ij->i', E[:, g]@np.linalg.inv(C[np.ix_(g, g)]), E[:, g]) for g in groups]
        m = [float(w@(d**2)/(ws*len(g)*(len(g)+2))) for d, g in zip(delta, groups)]
        nu = [ig_nu(v) for v in m] if nu_method == 'moment' else [profile_nu(d, w, len(g)) for d, g in zip(delta, groups)]
        comp = dict(weight=ws/n, mu=mu, groups=groups, nu=nu, m=m)
        if s_max is not None:
            comp['trunc'] = [TruncatedStrength(v, s) for v, s in zip(m, s_max)]
            comp['nu'] = [np.inf if t.gauss else t.nu for t in comp['trunc']]
        if len(groups) > 1:
            cross = [float(w@(delta[a]*delta[b])/(ws*len(groups[a])*len(groups[b]))) for a in range(len(groups)) for b in range(a+1, len(groups))]
            B = float(np.clip(np.mean(cross), 1., min(m))); comp.update(cross=cross, background_m=B, nu0=ig_nu(B), nu_local=[ig_nu(v/B) for v in m])
        S = strengths(comp, relation, 200000, rng); gid = np.concatenate([[i]*len(g) for i, g in enumerate(groups)])
        order = np.concatenate(groups); sq = np.sqrt(S[:, gid]); mean_pair = (sq.T@sq)/len(S)
        assert np.isfinite(mean_pair).all() and np.isfinite(C).all(), ('non-finite strength moments', j, ws, m,
            [(getattr(t, 'nu', None), getattr(t, 'mean', None)) for t in comp.get('trunc', [])])
        scat = np.zeros_like(C); scat[np.ix_(order, order)] = C[np.ix_(order, order)]/mean_pair
        shrink = 1.
        while np.linalg.eigvalsh(scat).min() <= 0 and shrink > 1e-3:
            shrink *= .9; off = scat-np.diag(np.diag(scat)); scat = np.diag(np.diag(scat))+off*shrink
        comp['scatter'] = floor_matrix(scat, FLOOR)[0]; comp['cross_shrink'] = shrink; comp['order'] = order; comp['gid'] = gid
        b = np.linalg.solve(design.T@(design*w[:, None]), design.T@(W*w[:, None])); e = W-design@b
        comp.update(intercept=b[0], coefficient=b[1:].T, residual=floor_matrix((e.T@(e*w[:, None]))/ws)[0]); comps.append(comp)
    return dict(grouped=comps, relation=relation, Q1=Q1, Q2=Q2)


def sample_grouped(model, n, rng):
    comps = model['grouped']; k = rng.choice(len(comps), size=n, p=np.array([c['weight'] for c in comps])/sum(c['weight'] for c in comps))
    out = np.empty((n, model['Q1'].shape[0]))
    for j, c in enumerate(comps):
        m = k == j; nm = int(m.sum())
        if nm == 0:
            continue
        S = strengths(c, model['relation'], nm, rng); z = rng.standard_normal((nm, len(c['mu'])))@np.linalg.cholesky(c['scatter']).T
        sq = np.ones((nm, len(c['mu']))); sq[:, c['order']] = np.sqrt(S[:, c['gid']]); U = c['mu']+z*sq
        Wd = c['intercept']+U@c['coefficient'].T+rng.standard_normal((nm, len(c['intercept'])))@np.linalg.cholesky(c['residual']).T
        out[m] = U@model['Q1'].T+Wd@model['Q2'].T
    return out, np.full(n, 1/n)


def model_outputs(model, data, seed):
    rng = np.random.default_rng(seed)
    if 'grouped' in model:
        z, w = sample_grouped(model, DRAWS, rng)
    elif 'mst' in model:
        z, w = mm.sample_mst(model, DRAWS, rng)
    else:
        z, w = base.propagate_any(model, DRAWS, seed)
    y = np.concatenate([base.outputs(base.voltage((zz*data['scale'])@data['basis'].T+data['center'])) for zz in np.array_split(z, 16)])
    return y, w


def metrics(y, w, data):
    ref = data['ref']; row = {}; tails = {}
    for j, side in data['targets']:
        r = np.sort(ref[:, j]); o = np.argsort(y[:, j]); ys, cw = y[o, j], np.cumsum(w[o])/w.sum()
        F = lambda v: np.interp(v, ys, cw, left=0., right=1.)
        q = r[(LEVELS*len(r)).astype(int)] if side == 'lower' else r[((1-LEVELS)*len(r)).astype(int)]
        err = (F(q)-LEVELS) if side == 'lower' else ((1-F(q))-LEVELS)
        name = f"{base.OBJECTS[j]['output']}{base.OBJECTS[j]['node'] or base.OBJECTS[j]['branch']}"
        row[name] = float(np.mean(abs(err))); row[name+'_signed'] = float(np.mean(err))
        tails[j] = (ys, cw)
    ks = []
    tset = {t[0] for t in data['targets']}
    for j in range(64):
        if j in tset:
            continue
        r = np.sort(ref[:, j]); o = np.argsort(y[:, j]); grid = r[np.linspace(0, len(r)-1, 1001).astype(int)]
        ks.append(np.max(abs(np.interp(grid, y[o, j], np.cumsum(w[o])/w.sum(), left=0., right=1.)-np.linspace(0, 1, 1001))))
    row['其余对象KS'] = float(np.median(ks)); return row, tails


def plot(scene, data, curves, out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']; plt.rcParams['axes.unicode_minus'] = False
    styles = {'MC方法（10⁶场景）': ('k', '-', 2.6), 'AQGMM': ('tab:blue', '--', 1.6), '多尺度t（矩投影）': ('tab:green', '-.', 1.6),
              '分块TMM（按矩）': ('tab:red', '-', 1.8), RELATIONS['indep']: ('tab:red', '-', 1.8), RELATIONS['sync']: ('tab:purple', ':', 1.8),
              RELATIONS['background']: ('tab:orange', (0, (5, 1, 1, 1)), 1.8)}
    for j, side in data['targets']:
        r = np.sort(data['ref'][:, j]); n = len(r); sd = r.std(); iqr = np.subtract(*np.quantile(r, [.75, .25])); h = .9*min(sd, iqr/1.349)*32768**-.2
        a, b = (r[int(.0005*n)], r[int(.05*n)]) if side == 'lower' else (r[int(.95*n)], r[int(.9995*n)])
        grid = np.linspace(a-2*h, b+2*h, 600); dx = grid[1]-grid[0]
        name = f"{base.OBJECTS[j]['output']}{base.OBJECTS[j]['node'] or base.OBJECTS[j]['branch']}"
        unit = '节点%d电压幅值 (p.u.)' % base.OBJECTS[j]['node'] if base.OBJECTS[j]['output'] == 'V' else '支路%d首端有功 (MW)' % base.OBJECTS[j]['branch']
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for lab, (ys, cw) in curves[j].items():
            F = np.interp(grid, ys, cw, left=0., right=1.)
            edges = np.r_[grid-dx/2, grid[-1]+dx/2]; Fe = np.interp(edges, ys, cw, left=0., right=1.)
            pdf = gaussian_filter1d(np.diff(Fe)/dx, h/dx, mode='nearest')
            c, ls, lw = styles['MC方法（10⁶场景）'] if lab.startswith('MC方法') else styles[lab]
            ax[0].plot(grid, pdf, color=c, ls=ls, lw=lw, label=lab)
            tail = F if side == 'lower' else 1-F
            ax[1].semilogy(grid, np.maximum(tail, 1e-6), color=c, ls=ls, lw=lw, label=lab)
        ax[0].set_xlim(a, b); ax[1].set_xlim(a, b); ax[1].set_ylim(3e-4, 0.08)
        ax[0].set_title('尾部概率密度'); ax[1].set_title('尾部累积概率' + ('（低于该值的概率）' if side == 'lower' else '（高于该值的概率）'))
        for x_ in ax:
            x_.set_xlabel(unit); x_.grid(alpha=.3, which='both')
        ax[0].set_ylabel('概率密度'); ax[1].set_ylabel('概率'); ax[0].legend(fontsize=8)
        fig.suptitle(f'{scene}，K={PLOT_K}，{"下尾" if side == "lower" else "上尾"}（MC方法0.05%—5%分位范围）', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(out/f'{name}_{side}_K{PLOT_K}_tail.png', dpi=130); plt.close(fig)


def main():
    loaders = {'case1': lambda: load_existing(VAL/'pv_nrel_case33_newsplit_20261022/nrel_ha4_40_60'),
               'two_cloud': lambda: load_existing(VAL/'pv_two_cloud_regions_case33_20261021/two_cloud_regions'),
               'shared_background': load_constructed}
    for scene in ARGS:
        t0 = time.perf_counter(); out = OUT/scene; out.mkdir(parents=True, exist_ok=True); data = loaders[scene]()
        rng = np.random.default_rng(2026102302)
        split, Q1, Q2 = chk.split(data['train'], data['x'], data['basis'], data['scale'], rng, scene, group_target=data['every'], return_q=True)
        groups = [np.array(g) for g in split['group_cols']]
        print(json.dumps(dict(stage='split', scene=scene, groups=split['groups'], group_dims=[len(g) for g in groups])), flush=True)
        rows, curves = [], {}
        for k in KS:
            gmm = fit_gmm(data['x'], data['xv'], k, data['scale'])
            models = [('AQGMM', gmm), ('多尺度t（矩投影）', mm.mst_moment_projection(data['x'], gmm, data['scale']))]
            if len(groups) == 1:
                models.append(('分块TMM（按矩）', grouped_model(data['x'], gmm, Q1, Q2, groups, 'indep', rng)))
            else:
                models += [(RELATIONS[rel], grouped_model(data['x'], gmm, Q1, Q2, groups, rel, rng)) for rel in RELATIONS]
            for name, m in models:
                y, w = model_outputs(m, data, 2026102310+k); row, tails = metrics(y, w, data)
                info = {}
                if 'grouped' in m:
                    info = dict(nu=[[round(v, 1) for v in c['nu']] for c in m['grouped']],
                                background_nu=[round(c.get('nu0', np.inf), 1) for c in m['grouped']],
                                cross_shrink=[round(c['cross_shrink'], 3) for c in m['grouped']])
                rows.append(dict(K=k, 模型=name, **row, **info)); print(json.dumps(dict(stage='evaluated', K=k, model=name, ks=round(row['其余对象KS'], 4)), ensure_ascii=False), flush=True)
                if k == PLOT_K:
                    for j, side in data['targets']:
                        curves.setdefault(j, {})[name] = tails[j]
        mc_label = f"MC方法（{len(data['ref'])}个场景）"
        for j, side in data['targets']:
            r = np.sort(data['ref'][:, j]); curves[j] = {mc_label: (r, np.arange(1, len(r)+1)/len(r)), **curves[j]}
        tab = pd.DataFrame(rows); tab.to_csv(out/'summary_same_K.csv', index=False)
        plot(scene, data, curves, out)
        (out/'run_info.json').write_text(json.dumps(dict(scene=scene, split={k_: v for k_, v in split.items() if k_ != 'group_cols'},
                                                         group_dims=[len(g) for g in groups], targets=data['targets'], seconds=time.perf_counter()-t0), ensure_ascii=False, indent=1, default=str))
        print(json.dumps(dict(stage='DONE', scene=scene, seconds=time.perf_counter()-t0)), flush=True)


if __name__ == '__main__':
    main()
