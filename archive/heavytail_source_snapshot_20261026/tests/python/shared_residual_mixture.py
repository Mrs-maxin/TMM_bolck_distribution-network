"""Joint-voltage block mixtures whose remaining directions share one residual covariance (derivation B20-B22).

The remaining directions w keep a per-component intercept and regression on the weather-driven block u,
but one residual covariance Omega serves all components. SBTMM is the t version, SGMM the same structure
with Gaussian components. install() binds the fixed-forecast runner module, the weather-driven subspace
function and the limit events, so the same phases serve the wind and PV entries.
"""
import json, time
import numpy as np
import pandas as pd

base = None
SUBSPACE = None
LIMIT_EVENTS = None
FAMILIES = {'SBTMM': False, 'SGMM': True}
SELECTION_FILES = None


def install(runner, subspace, limit_events):
    global base, SUBSPACE, LIMIT_EVENTS, SELECTION_FILES
    base, SUBSPACE, LIMIT_EVENTS = runner, subspace, limit_events
    SELECTION_FILES = {'GMM': 'selected_models.csv', 'TMM': 'selected_models.csv', base.BLOCK_FAMILY: 'selected_models_block.csv',
                       'SBTMM': 'selected_models_shared.csv', 'SGMM': 'selected_models_shared.csv'}


def shared_from_gmm(gmm, Q1, Q2, nu, gaussian):
    model = base.block_from_gmm(gmm, Q1, Q2, 1e300 if gaussian else nu)
    if gaussian:
        model['df'][:] = np.inf
    model['residual'][:] = np.einsum('k,kij->ij', model['weights'], model['residual'])
    return model


def fit_shared_block(U, W, initial, gaussian, maximum=300, tolerance=1e-6):
    from scipy.special import digamma
    from scipy.optimize import brentq
    start = time.perf_counter(); model = {key: np.array(v, copy=True) for key, v in initial.items()}
    n, r = U.shape; design = np.c_[np.ones(n), U]
    density, resp, delta = base.block_density(U, W, model); history = [density.mean()]; status = 'maximum_iterations'
    for iteration in range(1, maximum+1):
        count = resp.sum(0)
        if count.min() <= 4*(r+1):
            status = 'collapsed_component'; break
        new = {key: v.copy() for key, v in model.items()}; new['weights'] = count/n
        omega = np.zeros_like(model['residual'][0])
        for j, nu in enumerate(model['df']):
            tau = np.ones(n) if gaussian else (nu+r)/(nu+delta[:, j]); w = resp[:, j]*tau
            factor = 1.0 if gaussian else (nu-2)/nu
            new['means'][j] = (w@U)/w.sum(); res = U-new['means'][j]
            new['scatter'][j] = base.floor_matrix((res.T@(res*w[:, None]))/count[j], base.FLOOR*factor)[0]
            if not gaussian:
                statistic = resp[:, j]@(digamma((nu+r)/2)-np.log((nu+delta[:, j])/2)-tau)/count[j]
                score = lambda v: np.log(v/2)-digamma(v/2)+1+statistic
                new['df'][j] = 4.1 if score(4.1) <= 0 else 1e5 if score(1e5) >= 0 else brentq(score, 4.1, 1e5, xtol=1e-8)
            b = np.linalg.solve(design.T@(design*resp[:, j, None]), design.T@(W*resp[:, j, None]))
            new['intercept'][j], new['coefficient'][j] = b[0], b[1:].T
            e = W-design@b; omega += e.T@(e*resp[:, j, None])
        new['residual'][:] = base.floor_matrix(omega/n)[0]
        density, new_resp, new_delta = base.block_density(U, W, new); value = density.mean(); gain = value-history[-1]
        if gain < -1e-8:
            status = 'rejected_decreasing_update'; break
        model, resp, delta = new, new_resp, new_delta; history.append(value)
        if abs(gain) < tolerance*max(1., abs(history[-2])):
            status = 'converged'; break
    return model, dict(status=status, iterations=iteration, train_ll=history[-1], history=np.array(history),
                       df=model['df'], weights=model['weights'], effective_counts=resp.sum(0),
                       seconds=time.perf_counter()-start)


def parameter_count(k, r, dw, gaussian):
    return (k-1)+k*(r+r*(r+1)//2+(0 if gaussian else 1)+dw+dw*r)+dw*(dw+1)//2


def fit_candidates(scene, k, Q1, Q2):
    d = base.O/scene; c = d/'candidates'; fit_in = np.load(d/'fitting_inputs.npz')
    U, W = fit_in['x']@Q1, fit_in['x']@Q2; Uv, Wv = fit_in['validation']@Q1, fit_in['validation']@Q2
    selected, attempts = [], []
    for family, gaussian in FAMILIES.items():
        rows = []
        for p in range(1 if k == 1 else 2):
            gmm = np.load(c/f'K{k}_p{p+1:02d}_GMM.npz')
            for nu in ([None] if gaussian else [12, 200]):
                stem = c/(f'K{k}_p{p+1:02d}_{family}' if gaussian else f'K{k}_p{p+1:02d}_gmm_warm_df{nu}_{family}')
                if not stem.with_suffix('.json').exists():
                    base.event('fitting', scene=scene, K=k, family=family, partition=p+1, initial_df=nu)
                    model, audit = fit_shared_block(U, W, shared_from_gmm(gmm, Q1, Q2, nu, gaussian), gaussian)
                    model.update(Q1=Q1, Q2=Q2, block=np.array(True), shared_residual=np.array(True))
                    base.save_npz(stem.with_suffix('.npz'), **model, history=audit['history'])
                    audit.pop('history'); audit.update(scene=scene, K=k, family=family, partition=p+1, initial_df=nu,
                        validation_ll=float(base.block_density(Uv, Wv, model)[0].mean()), model_sha256=base.sha(stem.with_suffix('.npz')))
                    base.write_json(stem.with_suffix('.json'), audit)
                row = json.loads(stem.with_suffix('.json').read_text()); row['candidate'] = stem.name; rows.append(row)
        conv = [r for r in rows if r['status'] == 'converged']; best = max(conv or rows, key=lambda r: r['validation_ll'])
        selected.append(dict(scene=scene, K=k, family=family, candidate=best['candidate'], validation_ll=best['validation_ll'],
            train_ll=best['train_ll'], status=best['status'], iterations=best['iterations'], dimension=Q1.shape[0],
            parameter_count=parameter_count(k, Q1.shape[1], Q2.shape[1], gaussian), df=json.dumps(best['df']),
            weights=json.dumps(best['weights']), model_SHA256=best['model_sha256']))
        attempts += rows
    return selected, attempts


def phase_shared(scene):
    base.check_lock(scene); d = base.O/scene
    Q1, Q2 = SUBSPACE(scene)
    selected, attempts = [], []
    for k in base.KS:
        rows, tried = fit_candidates(scene, k, Q1, Q2); selected += rows; attempts += tried
        base.event('selected', scene=scene, K=k, family='shared', df=[r['df'] for r in rows])
    base.csv(d/'selected_models_shared.csv', selected); base.csv(d/'all_fit_attempts_shared.csv', attempts)
    base.archived_functions(base.SOURCE/'adapted_routines.py', {'evaluate'}, vars(base), replace={'evaluate': [
        ('2026101928', str(base.CONFIG['propagation_seed_base'])), ('def evaluate(scene,k):', 'def evaluate_shared(scene,k):'),
        ("'selected_models.csv'", "'selected_models_shared.csv'"),
        ("propagate(m,N,seed,'sobol',randomizer='MT19937')", 'propagate_any(m,N,seed)')]})
    for k in base.KS:
        base.evaluate_shared(scene, k)
    for kind, metrics, keys in [('global', base.SUMMARY_GLOBAL, base.SUMMARY_KEYS), ('quantile', base.SUMMARY_QUANTILE, base.SUMMARY_KEYS+['q'])]:
        frames = [pd.read_csv(p) for f in FAMILIES for p in sorted((d/'propagation').glob(f'{f}_*_{kind}s.csv' if kind == 'quantile' else f'{f}_*_{kind}.csv'))]
        base.repeat_summary(pd.concat(frames), metrics, keys).to_csv(base.O/f'shared_{kind}_repeat_summary.csv', index=False)
    base.event('SHARED_COMPLETE', scene=scene)


def selected_all(d):
    frames = [pd.read_csv(d/f) for f in sorted(set(SELECTION_FILES.values())) if (d/f).exists()]
    return pd.concat(frames, ignore_index=True)


def phase_limits(scene):
    base.check_lock(scene); d = base.O/scene; n = base.CONFIG['propagation_points']
    selected = selected_all(d)
    fit_in = np.load(d/'fitting_inputs.npz'); ref = np.load(d/'reference_outputs.npz')['values']
    rows = [dict(K='AC', family='reference', **{e: float(f(ref).mean()) for e, f in LIMIT_EVENTS.items()},
                 **{e+' count': int(f(ref).sum()) for e, f in LIMIT_EVENTS.items()})]
    for k in base.KS:
        for family in [f for f in SELECTION_FILES if f in set(selected.family)]:
            r = selected[(selected.K == k) & (selected.family == family)].iloc[0]
            m = np.load(d/'candidates'/(r.candidate+'.npz'))
            seed = base.CONFIG['propagation_seed_base']+100000*base.SCENES.index(scene)+1000*k
            z, w = base.propagate_any(m, n, seed)
            y = base.outputs(base.voltage((z*fit_in['scale'])@fit_in['basis'].T+fit_in['center']))
            saved = np.load(d/'propagation'/f'{family}_K{k}_r0_saved_outputs.npz')
            assert np.max(abs(y[:, list(saved['object_indices'])]-saved['values'])) < 1e-10
            rows.append(dict(K=k, family=family, **{e: float(w[f(y)].sum()) for e, f in LIMIT_EVENTS.items()}))
    base.csv(d/'limit_probabilities.csv', rows)
    base.event('LIMITS_COMPLETE', scene=scene)


def phase_compare(scene, criteria):
    targets = base.check_lock(scene); d = base.O/scene
    fit_in = np.load(d/'fitting_inputs.npz')
    with np.load(d/'reference_AC.npz') as a:
        v = a['voltage']
    z = (np.c_[v[:, 1:].real, v[:, 1:].imag]-fit_in['center'])@fit_in['basis']/fit_in['scale']
    selected = selected_all(d); curve = []
    for _, r in selected.iterrows():
        m = np.load(d/'candidates'/(r.candidate+'.npz'))
        ref_ll = base.block_density(z@m['Q1'], z@m['Q2'], m)[0].mean() if 'block' in m.files else np.mean(base.log_density(z, m)[0])
        curve.append(dict(family=r.family, K=int(r.K), parameter_count=int(r.parameter_count), train_ll=r.train_ll,
                          validation_ll=r.validation_ll, reference_ll=float(ref_ll), df=r.df))
    curve = pd.DataFrame(curve).sort_values(['family', 'K']); base.csv(d/'compare_likelihood_curve.csv', curve.to_dict('records'))
    best = {f: int(g.loc[g.validation_ll.idxmax(), 'K']) for f, g in curve.groupby('family')}
    gs = pd.concat([pd.read_csv(base.O/n) for n in ['global_repeat_summary.csv', 'block_global_repeat_summary.csv',
                    'shared_global_repeat_summary.csv']], ignore_index=True).query('scene==@scene')
    def metric(family, k, t, name):
        row = gs[(gs.family == family) & (gs.K == k) & (gs.output == t.output) & (gs.node == t.node) & (gs.branch == t.branch)]
        return float(row.iloc[0][name+'_mean'])
    target_rows = []
    for _, t in targets.iterrows():
        row = dict(output=t.output, node=int(t.node), branch=int(t.branch), side=t.side)
        for f, k in best.items():
            row[f'{f}_K{k}_tail_cdf_mean'] = metric(f, k, t, t.side+'_tail_cdf_mean_abs')
            row[f'{f}_K{k}_KS'] = metric(f, k, t, 'ks')
        for f in best:
            ok = [k for k in sorted(curve.K.unique()) if not gs[(gs.family == f) & (gs.K == k)].empty and
                  metric(f, k, t, 'ks') <= base.CONFIG['full_cdf_level'] and metric(f, k, t, t.side+'_tail_cdf_sup_abs') <= base.CONFIG['tail_cdf_level']]
            row[f'{f}_min_K_both_levels'] = ok[0] if ok else None
        target_rows.append(row)
    base.csv(d/'compare_targets_at_best_K.csv', target_rows)
    limits = pd.read_csv(d/'limit_probabilities.csv'); ref = limits[limits.family == 'reference'].iloc[0]
    limit_rows = [dict(family='reference', K='AC', **{e: ref[e] for e in LIMIT_EVENTS})]
    for f, k in best.items():
        r = limits[(limits.family == f) & (limits.K.astype(str) == str(k))].iloc[0]
        limit_rows.append(dict(family=f, K=k, **{e: r[e] for e in LIMIT_EVENTS}))
    base.csv(d/'compare_limits_at_best_K.csv', limit_rows)
    g, s = best['GMM'], best['SBTMM']
    tail_better = sum(r[f'SBTMM_K{s}_tail_cdf_mean'] < r[f'GMM_K{g}_tail_cdf_mean'] for r in target_rows)
    lg = limits[(limits.family == 'GMM') & (limits.K.astype(str) == str(g))].iloc[0]
    ls = limits[(limits.family == 'SBTMM') & (limits.K.astype(str) == str(s))].iloc[0]
    limit_closer = sum(abs(ls[e]-ref[e]) < abs(lg[e]-ref[e]) for e in LIMIT_EVENTS)
    best_val = curve.groupby('family').validation_ll.max().to_dict()
    summary = dict(scene=scene, best_K_by_validation=best, best_validation_ll=best_val,
                   prediction_1_shared_raises_validation={'SBTMM_over_BTMM': best_val['SBTMM'] > best_val[base.BLOCK_FAMILY],
                                                          'SGMM_over_GMM': best_val['SGMM'] > best_val['GMM'],
                                                          'best_K_not_smaller': {'SBTMM_vs_BTMM': best['SBTMM'] >= best[base.BLOCK_FAMILY],
                                                                                 'SGMM_vs_GMM': best['SGMM'] >= best['GMM']}},
                   prediction_2={'targets_tail_SBTMM_better_than_GMM': f'{tail_better}/{len(target_rows)}',
                                 'limit_events_SBTMM_closer_than_GMM': f'{limit_closer}/{len(LIMIT_EVENTS)}'},
                   criteria=criteria)
    base.write_json(d/'compare_summary.json', summary); base.event('COMPARE_COMPLETE', **summary)
