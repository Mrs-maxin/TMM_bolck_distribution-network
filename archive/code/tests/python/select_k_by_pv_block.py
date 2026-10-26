"""Choose the component count by the photovoltaic-driven voltage directions (pv_component_count_by_pv_block_20261021.md).

Uses the models already estimated for each scene; no parameter estimation or AC power flow is repeated.
The fit of the PV-driven directions is the marginal mean log-likelihood on the component-count scenarios.
"""
import importlib.util, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('run', REPO/'tests/python/run_pv_fixed_forecast_case33.py')
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
V = REPO/'result/case33bw'
SCENES = [('base', 'pv_fixed_forecast_case33_20261021', 'common_strength'),
          ('replicate', 'pv_fixed_forecast_case33_20261021_rep1', 'common_strength'),
          ('constant_weather', 'pv_fixed_forecast_case33_20261021_gaussian_control', 'gaussian_control'),
          ('A', 'pv_strong_weather_case33_20261021', 'strong_weather'),
          ('A_plus_B', 'pv_strong_weather_nominal_load_case33_20261021', 'strong_weather_nominal_load'),
          ('C', 'pv_two_cloud_regions_case33_20261021', 'two_cloud_regions')]


def projected(model, Q):
    return dict(weights=model['weights'], means=model['means']@Q,
                scatter=np.einsum('ia,kij,jb->kab', Q, model['scatter'], Q), df=model['df'])


def pv_block_loglik(family, model, z, Q):
    if family in ('GMM', 'TMM'):
        return float(run.log_density(z@Q, projected(model, Q))[0].mean())
    if family == 'BTMM':
        return float(run.log_density(z@model['Q1'], {k: model[k] for k in ['weights', 'means', 'scatter', 'df']})[0].mean())
    # Two-block: joint density of both weather-driven groups, dropping the remaining-direction factor.
    from scipy.special import gammaln
    from scipy.linalg import cholesky, solve_triangular
    U1, U2 = z@model['Q1'], z@model['Q2']; logs = []
    def tlog(e, scatter, nu):
        L = cholesky(scatter, lower=True); y = solve_triangular(L, e.T, lower=True).T; d2 = (y**2).sum(1); r = e.shape[1]
        return gammaln((nu+r)/2)-gammaln(nu/2)-r/2*np.log(nu*np.pi)-(nu+r)/2*np.log1p(d2/nu)-np.log(L.diagonal()).sum()
    for j, pi in enumerate(model['weights']):
        e1 = U1-model['means1'][j]; e2 = U2-model['means2'][j]-U1@model['coupling'][j].T
        logs.append(np.log(pi)+tlog(e1, model['scatter1'][j], model['df1'][j])+tlog(e2, model['scatter2'][j], model['df2'][j]))
    return float(logsumexp(np.array(logs).T, axis=1).mean())


rows, choices = [], []
for label, folder, scene in SCENES:
    O = V/folder; D = O/scene
    if not (D/'selected_models_block.csv').exists():
        continue
    z = np.load(D/'fitting_inputs.npz')['validation']
    Qpv = np.load(D/'block_subspace.npz')['Q1']
    if (D/'multiblock_subspace.npz').exists():
        mb = np.load(D/'multiblock_subspace.npz'); Qpv_mb = np.c_[mb['Q0'], mb['Q1']]
    tables = [pd.read_csv(D/'selected_models.csv'), pd.read_csv(D/'selected_models_block.csv')]
    if (D/'selected_models_multiblock.csv').exists():
        tables.append(pd.read_csv(D/'selected_models_multiblock.csv'))
    sel = pd.concat(tables, ignore_index=True)
    summaries = [pd.read_csv(O/'global_repeat_summary.csv'), pd.read_csv(O/'block_global_repeat_summary.csv')]
    if (O/'multiblock_global_repeat_summary.csv').exists():
        summaries.append(pd.read_csv(O/'multiblock_global_repeat_summary.csv'))
    gs = pd.concat(summaries, ignore_index=True).query('scene==@scene')
    targets = pd.read_csv(D/'targets.csv')
    for _, r in sel.iterrows():
        model = dict(np.load(D/'candidates'/(r.candidate+'.npz')))
        Q = Qpv_mb if (r.family == 'MBTMM') else Qpv
        if r.family == 'MBTMM':
            model['Q1'], model['Q2'] = mb['Q0'], mb['Q1']
        tail, ks = [], []
        for _, t in targets.iterrows():
            g = gs[(gs.family == r.family) & (gs.K == r.K) & (gs.output == t.output) & (gs.node == t.node) & (gs.branch == t.branch)].iloc[0]
            tail.append(g[t.side+'_tail_cdf_mean_abs_mean']); ks.append(g['ks_mean'])
        rows.append(dict(scene=label, family=r.family, K=int(r.K), joint_loglik=float(r.validation_ll),
                         pv_block_loglik=pv_block_loglik(r.family, model, z, Q),
                         target_tail_error_mean=float(np.mean(tail)), target_KS_mean=float(np.mean(ks))))
    print('done', label, flush=True)
table = pd.DataFrame(rows)
for (label, family), g in table.groupby(['scene', 'family'], sort=False):
    kj = int(g.loc[g.joint_loglik.idxmax(), 'K']); kp = int(g.loc[g.pv_block_loglik.idxmax(), 'K']); ko = int(g.loc[g.target_tail_error_mean.idxmin(), 'K'])
    e = g.set_index('K')
    choices.append(dict(scene=label, family=family, K_joint=kj, K_pv=kp, K_best_tail=ko,
                        tail_error_K_joint=e.loc[kj, 'target_tail_error_mean'], tail_error_K_pv=e.loc[kp, 'target_tail_error_mean'],
                        tail_error_best=e.loc[ko, 'target_tail_error_mean'], KS_K_joint=e.loc[kj, 'target_KS_mean'], KS_K_pv=e.loc[kp, 'target_KS_mean']))
out = V/'pv_component_count_by_pv_block_20261021'; out.mkdir(exist_ok=True)
table.to_csv(out/'loglik_and_errors_by_K.csv', index=False); pd.DataFrame(choices).to_csv(out/'component_count_choices.csv', index=False)
pd.set_option('display.width', 220)
print(pd.DataFrame(choices).round(4).to_string(index=False))
