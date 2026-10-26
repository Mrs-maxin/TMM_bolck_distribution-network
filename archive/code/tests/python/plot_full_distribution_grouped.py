"""Full PDF and CDF of selected electrical quantities, with mean, standard deviation, skewness and kurtosis.

Same models and inputs as tests/grouped_strength_tmm_case33.py at K=2: MC method, AQGMM, moment-projected
multiple-scaled t, grouped TMM (moment matched; one group equals block TMM). Diagnostic figures only.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import sys, json
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
sys.argv = sys.argv[:1]
import grouped_strength_tmm_case33 as gs
base = gs.base
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']; plt.rcParams['axes.unicode_minus'] = False

K = 2
OBJECTS = {'case1': [('V', 18), ('V', 30), ('P', 32), ('P', 19), ('loss', 0)],
           'two_cloud': [('V', 31), ('P', 31), ('P', 3), ('loss', 0)]}
STYLE = {'MC方法': ('k', '-', 2.6), 'AQGMM': ('tab:blue', '--', 1.6), '多尺度t（矩投影）': ('tab:green', '-.', 1.6),
         '分组TMM（按矩）': ('tab:red', '-', 1.8)}


def index(kind, n):
    for j, o in enumerate(base.OBJECTS):
        if (kind == 'V' and o['output'] == 'V' and o['node'] == n) or (kind == 'P' and o['output'] == 'P' and o['branch'] == n) or (kind == 'loss' and o['output'] == 'loss'):
            return j


def main():
    loaders = {'case1': lambda: gs.load_existing(gs.VAL/'pv_nrel_case33_newsplit_20261022/nrel_ha4_40_60'),
               'two_cloud': lambda: gs.load_existing(gs.VAL/'pv_two_cloud_regions_case33_20261021/two_cloud_regions')}
    rows = []
    for scene, objs in OBJECTS.items():
        out = gs.OUT/scene/'full_distribution'; out.mkdir(parents=True, exist_ok=True); data = loaders[scene]()
        rng = np.random.default_rng(2026102302)
        split, Q1, Q2 = gs.chk.split(data['train'], data['x'], data['basis'], data['scale'], rng, scene, group_target=data['every'], return_q=True)
        groups = [np.array(g) for g in split['group_cols']]
        gmm = gs.fit_gmm(data['x'], data['xv'], K, data['scale'])
        models = {'AQGMM': gmm, '多尺度t（矩投影）': gs.mm.mst_moment_projection(data['x'], gmm, data['scale']),
                  '分组TMM（按矩）': gs.grouped_model(data['x'], gmm, Q1, Q2, groups, 'indep', rng)}
        ys = {name: gs.model_outputs(m, data, 2026102310+K) for name, m in models.items()}
        for kind, n in objs:
            j = index(kind, n); r = data['ref'][:, j]
            label = f'节点{n}电压幅值 (p.u.)' if kind == 'V' else f'支路{n}首端有功 (MW)' if kind == 'P' else '系统有功网损 (MW)'
            sd = r.std(); iqr = np.subtract(*np.quantile(r, [.75, .25])); h = .9*min(sd, iqr/1.349)*len(r)**-.2
            lo, hi = np.quantile(r, [.0002, .9998]); grid = np.linspace(lo-3*h, hi+3*h, 1200); dx = grid[1]-grid[0]
            curves = {'MC方法': (np.sort(r), np.arange(1, len(r)+1)/len(r))}
            stat = {'MC方法': (r.mean(), r.std(), stats.skew(r), stats.kurtosis(r))}
            for name, (y, w) in ys.items():
                o = np.argsort(y[:, j]); curves[name] = (y[o, j], np.cumsum(w[o])/w.sum())
                m1 = w@y[:, j]/w.sum(); v = w@(y[:, j]-m1)**2/w.sum()
                stat[name] = (m1, np.sqrt(v), (w@(y[:, j]-m1)**3/w.sum())/v**1.5, (w@(y[:, j]-m1)**4/w.sum())/v**2-3)
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
            for name, (yy, cw) in curves.items():
                Fe = np.interp(np.r_[grid-dx/2, grid[-1]+dx/2], yy, cw, left=0., right=1.); pdf = gaussian_filter1d(np.diff(Fe)/dx, h/dx, mode='nearest')
                c, ls, lw = STYLE[name]; lab = f'{name}（{len(r)}个场景）' if name == 'MC方法' else name
                ax[0].plot(grid, pdf, color=c, ls=ls, lw=lw, label=lab); ax[1].plot(grid, np.interp(grid, yy, cw, left=0., right=1.), color=c, ls=ls, lw=lw, label=lab)
            for a in ax:
                a.set_xlabel(label); a.grid(alpha=.3); a.set_xlim(grid[0], grid[-1])
            ax[0].set_title('完整概率密度'); ax[1].set_title('完整累积分布'); ax[0].set_ylabel('概率密度'); ax[1].set_ylabel('累积概率'); ax[0].legend(fontsize=8)
            txt = '\n'.join(f"{nm}: 均值{s[0]:.4f} 标准差{s[1]:.4f}" for nm, s in stat.items())
            ax[1].text(0.02, 0.97, txt, transform=ax[1].transAxes, va='top', fontsize=7.5, family='Arial Unicode MS',
                       bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
            fig.suptitle(f'{scene}，K={K}', fontsize=11); fig.tight_layout(rect=(0, 0, 1, .94))
            tag = f'{kind}{n}' if kind != 'loss' else 'loss'; fig.savefig(out/f'{tag}_K{K}_full.png', dpi=130); plt.close(fig)
            m0, s0, g0, k0 = stat['MC方法']
            for name, (m1, s1, g1, k1) in stat.items():
                rows.append(dict(场景=scene, 对象=tag, 模型=name, 均值=m1, 标准差=s1, 偏度=g1, 超额峰度=k1,
                                 均值相对误差=(m1-m0)/abs(m0), 标准差相对误差=(s1-s0)/s0))
    t = pd.DataFrame(rows); t.to_csv(gs.OUT/'full_distribution_moments_K2.csv', index=False)
    pd.set_option('display.width', 250); print(t.round(5).to_string(index=False))


if __name__ == '__main__':
    main()
