"""Tail-oriented comparison methods for case 1 (01_IDEA/plans/case_study_plan_20261022.md section 5).

All methods use the same NREL-fitted input law as tests/run_pv_nrel_case33.py and are judged on the
same AC reference and locked limit events.
  Williams2012     linearized power flow at the expected injections; electrical-quantity moments up to
                   order 4 from the input law; maximum-entropy density on the linearized range.
  Mohammadi2017    mean-value first-order expansion from n+1 load flows; empirical cumulant generating
                   function of the linearized quantity; Lugannani-Rice saddlepoint CDF.
  LeiteDaSilva2018 cross-entropy importance sampling of each locked limit event with full AC load flow;
                   the standard-normal inputs behind the PV strength, PV deviations and loads are shifted.
Phases: analytic (Williams2012, Mohammadi2017), ceis (one event index or all), summary.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
from pathlib import Path
import sys, json, time, argparse
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.integrate import trapezoid
import run_pv_nrel_case33 as case
base = case.base

SCENE = case.SCENE
MOMENT_INPUTS = 400000
MOMENT_SEED = 2026102221
ME_ORDER = 4
ME_GRID = 4001
SPA_INPUTS = 50000
SPA_POINTS = 241
CE = dict(per_iteration=2000, elite=0.1, max_iterations=12, final=20000, seed=2026102231)
N_NORMALS = 1+8+2*(1+4+32)


def law():
    return case.nrel_law()


def inputs_from_normals(xi):
    """Map standard normals to bus injections exactly as run_pv_nrel_case33.generate does."""
    lw = law(); sigma = lw['log_sd']; n = len(xi)
    strength = np.exp(-sigma**2/2+sigma*xi[:, 0])
    z = xi[:, 1:9]
    pv_pu = np.clip(base.CONFIG['forecast_fraction']+lw['mean']+np.sqrt(strength)[:, None]*(z@np.linalg.cholesky(lw['covariance']).T),
                    0, lw['upper'])
    rho_g, rho_a, rho_n = base.CONFIG['load_PQ_correlation']
    wp, wq = np.array(base.CONFIG['load_P_weights'], float), np.array(base.CONFIG['load_Q_weights'], float)
    wp *= base.CONFIG['load_relative_sd']/np.linalg.norm(wp); wq *= base.CONFIG['load_relative_sd']/np.linalg.norm(wq)
    col = 9; parts = {}
    for name, width, rho in [('g', 1, rho_g), ('a', 4, rho_a), ('n', 32, rho_n)]:
        a = xi[:, col:col+width]; b = xi[:, col+width:col+2*width]; col += 2*width
        parts[name] = (a, rho*a+np.sqrt(1-rho*rho)*b)
    fp = 1+wp[0]*parts['g'][0]+wp[1]*parts['a'][0][:, base.REGIONS]+wp[2]*parts['n'][0]
    fq = 1+wq[0]*parts['g'][1]+wq[1]*parts['a'][1][:, base.REGIONS]+wq[2]*parts['n'][1]
    load_p = base.load_multiplier(SCENE)*base.BUS[None, 1:, 2]*fp
    load_q = base.load_multiplier(SCENE)*base.BUS[None, 1:, 3]*fq
    pbus = np.zeros((n, 32)); pbus[:, base.SITES-2] = base.CAP*pv_pu
    return np.c_[(pbus-load_p)/base.BASE, -load_q/base.BASE]


def normals_like_generate(n, seed):
    rng = np.random.Generator(np.random.MT19937(seed)); sigma = law()['log_sd']
    s = rng.normal(-sigma**2/2, sigma, n); z = rng.standard_normal((n, 8))
    load_rng = np.random.Generator(np.random.MT19937(seed+base.CONFIG['load_stream_offset']))
    blocks = []
    for width in [1, 4, 32]:
        blocks += [load_rng.standard_normal((n, width)), load_rng.standard_normal((n, width))]
    return np.c_[(s+sigma**2/2)/sigma, z, *blocks]


def check_input_map():
    d = base.O/SCENE; n = base.SPLITS['training']
    saved = np.load(d/'training_inputs.npz')['target']
    rebuilt = inputs_from_normals(normals_like_generate(n, base.CONFIG['seeds']['training']))
    err = float(np.max(abs(rebuilt-saved)))
    assert err < 1e-12, f'input map differs from the case runner: {err}'
    return err


def ac_outputs(target):
    out = np.empty((len(target), len(base.OBJECTS)))
    for i, t in enumerate(target):
        v, _ = base.polar_ac(t, base.Y, tolerance=1e-11); out[i] = base.outputs(v[None])[0]
    return out


def linearization(point):
    h = 1e-5; y0 = ac_outputs(point[None])[0]
    jac = np.empty((len(y0), len(point)))
    for k in range(len(point)):
        e = np.zeros_like(point); e[k] = h
        jac[:, k] = (ac_outputs((point+e)[None])[0]-ac_outputs((point-e)[None])[0])/(2*h)
    return y0, jac


def maxent_cdf(samples, grid):
    m, s = samples.mean(), samples.std()
    u = (samples-m)/s; mom = np.array([np.mean(u**k) for k in range(1, ME_ORDER+1)])
    lo, hi = (min(u.min(), (grid[0]-m)/s)-1, max(u.max(), (grid[-1]-m)/s)+1)
    x = np.linspace(lo, hi, ME_GRID); powers = np.vstack([x**k for k in range(1, ME_ORDER+1)])
    def dual(lam):
        e = lam@powers; c = e.max(); p = np.exp(e-c); z = trapezoid(p, x); q = p/z
        mean = trapezoid(powers*q, x, axis=1)
        cov = trapezoid(powers[:, None]*powers[None]*q, x, axis=2)-np.outer(mean, mean)
        return np.log(z)+c-lam@mom, mean-mom, cov
    lam = np.r_[0., -0.5, np.zeros(ME_ORDER-2)]
    value, grad, hess = dual(lam)
    for it in range(200):
        step = np.linalg.solve(hess+1e-12*np.eye(ME_ORDER), grad); a = 1.0
        while a > 1e-8:
            trial = dual(lam-a*step)
            if trial[0] <= value-1e-4*a*grad@step:
                break
            a /= 2
        lam = lam-a*step; value, grad, hess = trial
        if np.max(abs(grad)) < 1e-9:
            break
    e = lam@powers; p = np.exp(e-e.max()); cdf = np.concatenate([[0], np.cumsum((p[1:]+p[:-1])/2*np.diff(x))]); cdf /= cdf[-1]
    return np.interp((grid-m)/s, x, cdf), dict(converged=bool(np.max(abs(grad)) < 1e-7), moment_residual=float(np.max(abs(grad))))


def saddlepoint_cdf(samples, grid):
    m, s = samples.mean(), samples.std(); u = (samples[:SPA_INPUTS]-m)/s
    def cgf(t):
        parts = []
        for chunk in np.array_split(t, max(1, len(t)//40)):
            a = np.outer(chunk, u); c = a.max(1, keepdims=True); e = np.exp(a-c); w = e/e.sum(1, keepdims=True)
            k1 = w@u; parts.append((np.log(e.mean(1))+c[:, 0], k1, w@(u*u)-k1**2))
        return tuple(np.concatenate(p) for p in zip(*parts))
    fine = grid; grid = np.linspace(fine[0], fine[-1], SPA_POINTS)
    y = (grid-m)/s; t = np.clip(y, -20, 20).astype(float)
    inside = (y > u.min()) & (y < u.max())
    for _ in range(60):
        _, k1, k2 = cgf(t); step = (k1-y)/np.maximum(k2, 1e-12); t = np.where(inside, np.clip(t-step, -200, 200), t)
        if np.max(abs(step[inside])) < 1e-10:
            break
    _, k1, _ = cgf(t); redo = inside & ~(abs(k1-y) < 1e-8)
    if np.any(redo):
        lo, hi = np.full(redo.sum(), -400.), np.full(redo.sum(), 400.)
        for _ in range(80):
            mid = (lo+hi)/2; below = cgf(mid)[1] < y[redo]; lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
        t[redo] = (lo+hi)/2
    k0, k1, k2 = cgf(t)
    w = np.sign(t)*np.sqrt(np.maximum(2*(t*y-k0), 0)); r = t*np.sqrt(np.maximum(k2, 1e-300))
    with np.errstate(divide='ignore', invalid='ignore'):
        cdf = stats.norm.cdf(w)+stats.norm.pdf(w)*(1/w-1/r)
    near = np.abs(t) < 1e-4
    cdf[near] = stats.norm.cdf(y[near])-stats.norm.pdf(y[near])*np.mean(u**3)/6*(y[near]**2-1)
    cdf[y <= u.min()] = 0; cdf[y >= u.max()] = 1
    cdf = np.maximum.accumulate(np.clip(np.nan_to_num(cdf, nan=0.5), 0, 1))
    solved = inside & (abs(k1-y) < 1e-6)
    return np.interp(fine, grid, cdf), dict(saddlepoint_solved_share=float(solved.sum()/max(inside.sum(), 1)),
                                            unsolved_range=[float(grid[inside & ~solved].min()), float(grid[inside & ~solved].max())]
                                            if np.any(inside & ~solved) else [])


def phase_analytic():
    d = base.O/SCENE; out = d/'tail_baselines'; out.mkdir(exist_ok=True)
    err = check_input_map()
    xi = normals_like_generate(MOMENT_INPUTS, MOMENT_SEED); x = inputs_from_normals(xi)
    point = x.mean(0); y0, jac = linearization(point)
    lin = y0+(x-point)@jac.T
    ref = np.load(d/'reference_outputs.npz')['values']
    targets = pd.read_csv(d/'targets.csv'); events = json.loads((d/'limit_events_locked.json').read_text())['events']
    levels = np.array([.001, .005, .01, .05, .95, .99, .995, .999])
    rows, lim_rows, curves = [], [], {}
    for j, obj in enumerate(base.OBJECTS):
        r = np.sort(ref[:, j]); grid = np.linspace(r[0]-3*r.std(), r[-1]+3*r.std(), 1601)
        ecdf = np.searchsorted(r, grid, side='right')/len(r)
        for name, fn in [('Williams2012', maxent_cdf), ('Mohammadi2017', saddlepoint_cdf)]:
            cdf, audit = fn(lin[:, j], grid)
            ks = float(np.max(abs(cdf-ecdf)))
            rq = np.quantile(ref[:, j], levels); mq = np.interp(levels, np.maximum.accumulate(cdf)+np.arange(len(cdf))*1e-15, grid)
            rows.append(dict(method=name, object_index=j, **obj, ks=ks, **audit,
                             **{f'ref_q{l}': a for l, a in zip(levels, rq)}, **{f'model_q{l}': b for l, b in zip(levels, mq)},
                             lower_tail_cdf_sup_abs=float(np.max(abs(cdf-ecdf)[ecdf <= .05])) if np.any(ecdf <= .05) else np.nan,
                             upper_tail_cdf_sup_abs=float(np.max(abs(cdf-ecdf)[ecdf >= .95])) if np.any(ecdf >= .95) else np.nan))
            curves[(name, j)] = cdf
            for e in [e for e in events if e['object_index'] == j]:
                p = float(np.interp(e['value'], grid, cdf)); p = p if e['op'] == '<' else 1-p
                lim_rows.append(dict(method=name, event=e['name'], probability=p))
    base.csv(out/'analytic_baseline_objects.csv', rows); base.csv(out/'analytic_baseline_limits.csv', lim_rows)
    base.save_npz(out/'analytic_baseline_cdfs.npz', **{f'{m}_{j}': c for (m, j), c in curves.items()})
    base.write_json(out/'analytic_baseline_protocol.json', dict(input_map_max_error=err, moment_inputs=MOMENT_INPUTS,
        moment_seed=MOMENT_SEED, maxent_order=ME_ORDER, expansion_point='mean of the input law', sensitivity='central difference, h=1e-5 p.u.',
        script_SHA256=base.sha(Path(__file__)), targets=targets.to_dict('records')))
    base.event('TAIL_ANALYTIC_COMPLETE', objects=len(base.OBJECTS))


def object_metrics(name, j, cdf, grid, ecdf, ref_col, levels, audit):
    rq = np.quantile(ref_col, levels); mq = np.interp(levels, np.maximum.accumulate(cdf)+np.arange(len(cdf))*1e-15, grid)
    return dict(method=name, object_index=j, **base.OBJECTS[j], ks=float(np.max(abs(cdf-ecdf))), **audit,
                **{f'ref_q{l}': a for l, a in zip(levels, rq)}, **{f'model_q{l}': b for l, b in zip(levels, mq)},
                lower_tail_cdf_sup_abs=float(np.max(abs(cdf-ecdf)[ecdf <= .05])) if np.any(ecdf <= .05) else np.nan,
                upper_tail_cdf_sup_abs=float(np.max(abs(cdf-ecdf)[ecdf >= .95])) if np.any(ecdf >= .95) else np.nan)


def phase_gao():
    """Gao2024: joint GMM of the bus injections, power flow linearized at every component mean (piecewise linear),
    electrical quantities as Gaussian mixtures. EM on the same input scenarios replaces KDE + density-preserving HEM."""
    d = base.O/SCENE; out = d/'tail_baselines'; out.mkdir(exist_ok=True)
    xt = np.load(d/'training_inputs.npz')['target']; xv = np.load(d/'validation_inputs.npz')['target']
    center = xt.mean(0); _, sv, vt = np.linalg.svd(xt-center, full_matrices=False)
    basis = vt.T; scale = sv/np.sqrt(len(xt)-1)
    zt = (xt-center)@basis/scale; zv = (xv-center)@basis/scale
    selection = []
    for k in base.KS:
        best = None
        for p in range(1 if k == 1 else 2):
            ini, _, _ = base.initialize(zt, k, p, scale, population_covariance=True)
            m, audit = base.fit(zt, ini, actual_covariance_floor=True, relative_tolerance=True)
            val = float(base.log_density(zv, m)[0].mean())
            row = dict(K=k, partition=p+1, status=audit['status'], validation_ll=val, train_ll=float(audit['train_ll']))
            if best is None or val > best[0]['validation_ll']:
                best = (row, m)
        row, m = best; selection.append(row)
        comps = []
        for w, mu, S in zip(m['weights'], m['means'], m['scatter']):
            mu_x = center+basis@(scale*mu); cov_x = (basis*scale)@S@(basis*scale).T
            y0, jac = linearization(mu_x); comps.append((w, y0, np.sqrt(np.einsum('ij,jk,ik->i', jac, cov_x, jac))))
        base.save_npz(out/f'gao_K{k}.npz', weights=np.array([c[0] for c in comps]), means=np.array([c[1] for c in comps]),
                      sds=np.array([c[2] for c in comps]))
        base.event('GAO_K', K=k, validation_ll=row['validation_ll'], status=row['status'])
    base.csv(out/'gao_selection.csv', selection)
    best_k = max(selection, key=lambda r: r['validation_ll'])['K']
    ref = np.load(d/'reference_outputs.npz')['values']; events = json.loads((d/'limit_events_locked.json').read_text())['events']
    levels = np.array([.001, .005, .01, .05, .95, .99, .995, .999]); rows, lim_rows = [], []
    for k in base.KS:
        g = np.load(out/f'gao_K{k}.npz')
        for j in range(len(base.OBJECTS)):
            r = np.sort(ref[:, j]); grid = np.linspace(r[0]-3*r.std(), r[-1]+3*r.std(), 1601)
            ecdf = np.searchsorted(r, grid, side='right')/len(r)
            cdf_of = lambda v: g['weights']@stats.norm.cdf((np.atleast_1d(v)[None]-g['means'][:, j, None])/g['sds'][:, j, None])
            rows.append(dict(K=k, selected=k == best_k, **object_metrics('Gao2024', j, cdf_of(grid), grid, ecdf, ref[:, j], levels, {})))
            for e in [e for e in events if e['object_index'] == j]:
                p = float(cdf_of(e['value'])[0]); lim_rows.append(dict(method='Gao2024', K=k, selected=k == best_k, event=e['name'],
                                                                      probability=p if e['op'] == '<' else 1-p))
    base.csv(out/'gao_objects.csv', rows); base.csv(out/'gao_limits.csv', lim_rows)
    base.write_json(out/'gao_protocol.json', dict(best_K_by_validation=best_k, K=base.KS, input_dimension=xt.shape[1],
        substitution='EM on the same 8192 input scenarios replaces KDE + density-preserving hierarchical EM',
        linearization='central difference at every component input mean, h=1e-5 p.u.', script_SHA256=base.sha(Path(__file__))))
    base.event('GAO_COMPLETE', best_K=best_k)


def phase_ceis(index):
    d = base.O/SCENE; out = d/'tail_baselines'; out.mkdir(exist_ok=True)
    events = json.loads((d/'limit_events_locked.json').read_text())['events']
    for i in ([index] if index is not None else range(len(events))):
        e = events[i]; path = out/f'ceis_event{i:02d}.json'
        if path.exists():
            continue
        rng = np.random.Generator(np.random.MT19937(CE['seed']+i)); mean = np.zeros(N_NORMALS)
        sign = -1 if e['op'] == '<' else 1; clock = time.perf_counter(); history = []; runs = 0
        for it in range(CE['max_iterations']):
            xi = mean+rng.standard_normal((CE['per_iteration'], N_NORMALS)); y = ac_outputs(inputs_from_normals(xi))[:, e['object_index']]
            runs += len(xi); score = sign*y; level = min(np.quantile(score, 1-CE['elite']), sign*e['value'])
            elite = score >= level; w = np.exp(-xi[elite]@mean+mean@mean/2)
            mean = (w@xi[elite])/w.sum(); history.append(dict(iteration=it, elite_level=float(sign*level)))
            if level >= sign*e['value']:
                break
        xi = mean+rng.standard_normal((CE['final'], N_NORMALS)); y = ac_outputs(inputs_from_normals(xi))[:, e['object_index']]
        runs += len(xi); hit = (y < e['value']) if e['op'] == '<' else (y > e['value'])
        w = np.exp(-xi@mean+mean@mean/2); est = w*hit; p = float(est.mean()); cv = float(est.std()/np.sqrt(len(est))/p) if p > 0 else np.inf
        base.write_json(path, dict(event=e['name'], probability=p, coefficient_of_variation=cv, AC_load_flows=runs,
                                   seconds=time.perf_counter()-clock, history=history, shift_norm=float(np.linalg.norm(mean)),
                                   shift_largest=dict(zip(map(int, np.argsort(-abs(mean))[:5]), np.sort(-abs(mean))[:5].tolist())),
                                   settings=CE, script_SHA256=base.sha(Path(__file__))))
        base.event('CEIS_EVENT', event=e['name'], probability=p, cv=cv, load_flows=runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['analytic', 'ceis', 'gao'])
    parser.add_argument('--K-order', default='1,2,3,4,6,8,12')
    parser.add_argument('--out', default=str(base.REPO/'result/case33bw/pv_nrel_case33_20261022'))
    parser.add_argument('--event', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    base.CONFIG['K_order'] = [int(k) for k in args.K_order.split(',')]
    base.O = Path(args.out); base.apply_mode(args.smoke)
    if args.smoke:
        global MOMENT_INPUTS; MOMENT_INPUTS = 20000; CE.update(per_iteration=300, final=1000, max_iterations=4)
    {'analytic': phase_analytic, 'ceis': lambda: phase_ceis(args.event), 'gao': phase_gao}[args.phase]()


if __name__ == '__main__':
    main()
