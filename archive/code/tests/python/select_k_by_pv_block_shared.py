"""Component count by the PV-driven directions for the shared-residual retest (plan section 4 of
01_IDEA/plans/pv_component_count_by_pv_block_20261021.md). Reads the other conversation's results without changing them.
"""
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('run', REPO/'tests/python/run_pv_fixed_forecast_case33.py')
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
O = REPO/'result/case33bw/pv_shared_residual_case33_20261021'; D = O/'common_strength'; SCENE = 'common_strength'
z = np.load(D/'fitting_inputs.npz')['validation']; Qpv = np.load(D/'block_subspace.npz')['Q1']
sel = pd.concat([pd.read_csv(D/f) for f in ['selected_models.csv', 'selected_models_block.csv', 'selected_models_shared.csv']], ignore_index=True)
gs = pd.concat([pd.read_csv(O/f) for f in ['global_repeat_summary.csv', 'block_global_repeat_summary.csv', 'shared_global_repeat_summary.csv']],
               ignore_index=True).query('scene==@SCENE')
targets = pd.read_csv(D/'targets.csv'); limits = pd.read_csv(D/'limit_probabilities.csv')
events = [c for c in limits.columns if c not in ('K', 'family') and not c.endswith('count')]
ac = limits[limits.family == 'reference'].iloc[0]
rows = []
for _, r in sel.iterrows():
    m = dict(np.load(D/'candidates'/(r.candidate+'.npz')))
    if r.family in ('GMM', 'TMM'):
        part = dict(weights=m['weights'], means=m['means']@Qpv, scatter=np.einsum('ia,kij,jb->kab', Qpv, m['scatter'], Qpv), df=m['df'])
        u = z@Qpv
    else:
        part = {k: m[k] for k in ['weights', 'means', 'scatter', 'df']}; u = z@m['Q1']
    pv_ll = float(run.log_density(u, part)[0].mean())
    tail = [gs[(gs.family == r.family) & (gs.K == r.K) & (gs.output == t.output) & (gs.node == t.node) & (gs.branch == t.branch)].iloc[0][t.side+'_tail_cdf_mean_abs_mean']
            for _, t in targets.iterrows()]
    lim = limits[(limits.family == r.family) & (limits.K.astype(str) == str(int(r.K)))]
    rel = float(np.mean([abs(lim.iloc[0][e]-ac[e])/ac[e] for e in events])) if len(lim) else np.nan
    rows.append(dict(family=r.family, K=int(r.K), joint_loglik=float(r.validation_ll), pv_block_loglik=pv_ll,
                     target_tail_error_mean=float(np.mean(tail)), limit_relative_error_mean=rel))
table = pd.DataFrame(rows).sort_values(['family', 'K']); choices = []
for fam, g in table.groupby('family'):
    e = g.set_index('K'); kj, kp, ko = e.joint_loglik.idxmax(), e.pv_block_loglik.idxmax(), e.target_tail_error_mean.idxmin()
    choices.append(dict(family=fam, K_joint=kj, K_pv=kp, K_best_tail=ko, tail_K_joint=e.loc[kj, 'target_tail_error_mean'],
                        tail_K_pv=e.loc[kp, 'target_tail_error_mean'], tail_best=e.loc[ko, 'target_tail_error_mean'],
                        limit_rel_K_joint=e.loc[kj, 'limit_relative_error_mean'], limit_rel_K_pv=e.loc[kp, 'limit_relative_error_mean']))
out = REPO/'result/case33bw/pv_component_count_by_pv_block_20261021'; out.mkdir(exist_ok=True)
table.to_csv(out/'shared_retest_loglik_and_errors_by_K.csv', index=False); pd.DataFrame(choices).to_csv(out/'shared_retest_component_count_choices.csv', index=False)
pd.set_option('display.width', 220)
print(table.round(4).to_string(index=False)); print(); print(pd.DataFrame(choices).round(4).to_string(index=False))
