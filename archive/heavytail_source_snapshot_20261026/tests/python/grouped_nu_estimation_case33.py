"""Degree-of-freedom estimation for the grouped t mixture: fourth-moment rule against covariance-preserving likelihood.

Both keep the AQGMM partition and the weighted covariance of each component; they differ only in how each
group's degree of freedom is chosen. Groups are synchronous (grouped t, one uniform drives all groups).
Reports tail errors, KS of the other quantities, and mean / standard deviation / excess kurtosis of selected
quantities against the MC method, at equal K.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import sys, json
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import stats
sys.argv = sys.argv[:1]
import grouped_strength_tmm_case33 as gs
base = gs.base

KS = [1, 2, 3, 4]
MOMENT_OBJECTS = {'case1': [('V', 18), ('V', 30), ('P', 32), ('P', 19), ('loss', 0)], 'two_cloud': [('V', 31), ('P', 31), ('P', 3), ('loss', 0)]}


def index(kind, n):
    for j, o in enumerate(base.OBJECTS):
        if (kind == 'V' and o['output'] == 'V' and o['node'] == n) or (kind == 'P' and o['output'] == 'P' and o['branch'] == n) or (kind == 'loss' and o['output'] == 'loss'):
            return j


def moments(y, w):
    m = w@y/w.sum(); v = w@(y-m)**2/w.sum(); return m, np.sqrt(v), (w@(y-m)**4/w.sum())/v**2-3


def main():
    loaders = {'case1': lambda: gs.load_existing(gs.VAL/'pv_nrel_case33_newsplit_20261022/nrel_ha4_40_60'),
               'two_cloud': lambda: gs.load_existing(gs.VAL/'pv_two_cloud_regions_case33_20261021/two_cloud_regions')}
    rows, mrows = [], []
    for scene, objs in MOMENT_OBJECTS.items():
        data = loaders[scene](); rng = np.random.default_rng(2026102302)
        split, Q1, Q2 = gs.chk.split(data['train'], data['x'], data['basis'], data['scale'], rng, scene, group_target=data['every'], return_q=True)
        groups = [np.array(g) for g in split['group_cols']]
        for k in KS:
            gmm = gs.fit_gmm(data['x'], data['xv'], k, data['scale'])
            models = [('AQGMM', gmm), ('多尺度t（矩投影）', gs.mm.mst_moment_projection(data['x'], gmm, data['scale'])),
                      ('分组t（四阶矩定自由度）', gs.grouped_model(data['x'], gmm, Q1, Q2, groups, 'sync', rng, 'moment')),
                      ('分组t（似然定自由度）', gs.grouped_model(data['x'], gmm, Q1, Q2, groups, 'sync', rng, 'profile'))]
            for name, m in models:
                y, w = gs.model_outputs(m, data, 2026102310+k); row, _ = gs.metrics(y, w, data)
                V = [v for c, v in row.items() if c.startswith('V') and not c.endswith('_signed')]
                P = [v for c, v in row.items() if c.startswith('P') and not c.endswith('_signed')]
                info = dict(自由度=[[round(v, 1) for v in c['nu']] for c in m['grouped']]) if 'grouped' in m else {}
                rows.append(dict(场景=scene, K=k, 模型=name, 电压尾段误差中位数=np.median(V), 支路尾段误差中位数=np.median(P), 其余对象KS=row['其余对象KS'], **info))
                for kind, n in objs:
                    j = index(kind, n); r = data['ref'][:, j]; m0, s0, k0 = r.mean(), r.std(), stats.kurtosis(r); m1, s1, k1 = moments(y[:, j], w)
                    mrows.append(dict(场景=scene, K=k, 对象=f'{kind}{n}' if kind != 'loss' else 'loss', 模型=name, 均值相对误差=(m1-m0)/abs(m0),
                                      标准差相对误差=(s1-s0)/s0, 超额峰度=k1, MC超额峰度=k0))
                print(json.dumps(dict(stage='done', scene=scene, K=k, model=name), ensure_ascii=False), flush=True)
    out = gs.OUT/'nu_estimation'; out.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows); t.to_csv(out/'tail_and_ks.csv', index=False); mt = pd.DataFrame(mrows); mt.to_csv(out/'moments.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
    print(t.round(4).to_string(index=False))
    print(mt.pivot_table(index=['场景', '对象', 'K'], columns='模型', values='超额峰度').round(2).join(
        mt.groupby(['场景', '对象', 'K']).MC超额峰度.first().round(2)).to_string())
    print(mt.pivot_table(index=['场景', '对象', 'K'], columns='模型', values='标准差相对误差').round(4).to_string())


if __name__ == '__main__':
    main()
