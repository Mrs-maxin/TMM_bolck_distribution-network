"""Grouped t with the fluctuation strength capped by the PV output limits (scheme A).

For each weather group, s_max = median over its injections of (largest possible deviation / error standard
deviation)^2, where the largest possible deviation is the distance from the scenario mean injection to the
observed output limits over all injection scenarios (PV between zero and its upper limit). The truncated
inverse-gamma strength is normalized to mean one and its degree of freedom matches the group fourth
moment. Compared at equal K with AQGMM, moment-projected multiple-scaled t and untruncated grouped t.
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

KS = [1, 2, 3, 4]; PLOT_K = 2
OBJ = {'case1': [('V', 18), ('V', 30), ('P', 32), ('P', 19), ('loss', 0)], 'two_cloud': [('V', 31), ('P', 31), ('P', 3), ('loss', 0)]}
STYLE = {'MC方法': ('k', '-', 2.6), 'AQGMM': ('tab:blue', '--', 1.5), '多尺度t（矩投影）': ('tab:green', '-.', 1.5),
         '分组t（未截断）': ('tab:purple', ':', 1.6), '分组t（强度按出力限值截断）': ('tab:red', '-', 1.9)}
OUT = gs.OUT/'truncated_strength'


def index(kind, n):
    for j, o in enumerate(base.OBJECTS):
        if (kind == 'V' and o['output'] == 'V' and o['node'] == n) or (kind == 'P' and o['output'] == 'P' and o['branch'] == n) or (kind == 'loss' and o['output'] == 'loss'):
            return j


def strength_cap(every, names):
    e = every-every.mean(0); out = []
    for g in names:
        idx = [list(gs.chk.NAMES).index(nm) for nm in g]; sd = e[:, idx].std(0)
        dmax = np.maximum(e[:, idx].max(0), -e[:, idx].min(0)); out.append(float(np.median((dmax/sd)**2)))
    return out


def plot(scene, data, curves, objs, out):
    for kind, n in objs:
        j = index(kind, n); r = np.sort(data['ref'][:, j]); N = len(r); sd = r.std(); iqr = np.subtract(*np.quantile(r, [.75, .25])); h = .9*min(sd, iqr/1.349)*N**-.2
        label = f'节点{n}电压幅值 (p.u.)' if kind == 'V' else f'支路{n}首端有功 (MW)' if kind == 'P' else '系统有功网损 (MW)'
        side = next((s for jj, s in data['targets'] if jj == j), 'lower' if kind == 'V' else 'upper')
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        full = np.linspace(r[int(.0002*N)]-3*h, r[int(.9998*N)]+3*h, 1200)
        a, b = (r[int(.0005*N)], r[int(.05*N)]) if side == 'lower' else (r[int(.95*N)], r[int(.9995*N)])
        tail = np.linspace(a, b, 500)
        for lab, (ys, cw) in curves[j].items():
            c, ls, lw = STYLE['MC方法' if lab.startswith('MC') else lab]
            dx = full[1]-full[0]; Fe = np.interp(np.r_[full-dx/2, full[-1]+dx/2], ys, cw, left=0., right=1.)
            ax[0].plot(full, gaussian_filter1d(np.diff(Fe)/dx, h/dx, mode='nearest'), color=c, ls=ls, lw=lw, label=lab)
            ax[1].plot(full, np.interp(full, ys, cw, left=0., right=1.), color=c, ls=ls, lw=lw, label=lab)
            F = np.interp(tail, ys, cw, left=0., right=1.); ax[2].semilogy(tail, np.maximum(F if side == 'lower' else 1-F, 1e-6), color=c, ls=ls, lw=lw, label=lab)
        ax[2].set_ylim(3e-4, .08); ax[2].set_xlim(a, b)
        for x_, t in zip(ax, ['完整概率密度', '完整累积分布', '尾部累积概率（' + ('低于' if side == 'lower' else '高于') + '该值，对数坐标）']):
            x_.set_title(t, fontsize=10); x_.set_xlabel(label); x_.grid(alpha=.3, which='both')
        ax[0].set_xlim(full[0], full[-1]); ax[1].set_xlim(full[0], full[-1]); ax[0].legend(fontsize=7.5)
        fig.suptitle(f'{scene}，K={PLOT_K}', fontsize=11); fig.tight_layout(rect=(0, 0, 1, .93))
        fig.savefig(out/f"{kind}{n if kind != 'loss' else ''}_K{PLOT_K}.png", dpi=125); plt.close(fig)


def main():
    loaders = {'case1': lambda: gs.load_existing(gs.VAL/'pv_nrel_case33_newsplit_20261022/nrel_ha4_40_60'),
               'two_cloud': lambda: gs.load_existing(gs.VAL/'pv_two_cloud_regions_case33_20261021/two_cloud_regions')}
    rows, mrows = [], []
    for scene, objs in OBJ.items():
        out = OUT/scene; out.mkdir(parents=True, exist_ok=True); data = loaders[scene](); rng = np.random.default_rng(2026102302)
        split, Q1, Q2 = gs.chk.split(data['train'], data['x'], data['basis'], data['scale'], rng, scene, group_target=data['every'], return_q=True)
        groups = [np.array(g) for g in split['group_cols']]; s_max = strength_cap(data['every'], split['groups'])
        print(json.dumps(dict(stage='cap', scene=scene, groups=split['groups'], s_max=[round(s, 2) for s in s_max]), ensure_ascii=False), flush=True)
        curves = {}
        for k in KS:
            gmm = gs.fit_gmm(data['x'], data['xv'], k, data['scale'])
            models = [('AQGMM', gmm), ('多尺度t（矩投影）', gs.mm.mst_moment_projection(data['x'], gmm, data['scale'])),
                      ('分组t（未截断）', gs.grouped_model(data['x'], gmm, Q1, Q2, groups, 'sync', rng)),
                      ('分组t（强度按出力限值截断）', gs.grouped_model(data['x'], gmm, Q1, Q2, groups, 'sync', rng, s_max=s_max))]
            for name, m in models:
                y, w = gs.model_outputs(m, data, 2026102310+k); row, _ = gs.metrics(y, w, data)
                V = [v for c, v in row.items() if c.startswith('V') and not c.endswith('_signed')]
                P = [v for c, v in row.items() if c.startswith('P') and not c.endswith('_signed')]
                info = dict(自由度=[[round(float(v), 2) for v in c['nu']] for c in m['grouped']]) if 'grouped' in m else {}
                rows.append(dict(场景=scene, K=k, 模型=name, 电压尾段误差中位数=np.median(V), 支路尾段误差中位数=np.median(P), 其余对象KS=row['其余对象KS'], **info))
                for kind, n in objs:
                    j = index(kind, n); r = data['ref'][:, j]; mm1 = w@y[:, j]/w.sum(); v1 = w@(y[:, j]-mm1)**2/w.sum()
                    mrows.append(dict(场景=scene, K=k, 对象=f'{kind}{n}' if kind != 'loss' else 'loss', 模型=name, 标准差相对误差=(np.sqrt(v1)-r.std())/r.std(),
                                      超额峰度=(w@(y[:, j]-mm1)**4/w.sum())/v1**2-3, MC超额峰度=stats.kurtosis(r)))
                    if k == PLOT_K:
                        o = np.argsort(y[:, j]); curves.setdefault(j, {})[name] = (y[o, j], np.cumsum(w[o])/w.sum())
                print(json.dumps(dict(stage='done', scene=scene, K=k, model=name), ensure_ascii=False), flush=True)
        for kind, n in objs:
            j = index(kind, n); r = np.sort(data['ref'][:, j]); curves[j] = {f'MC方法（{len(r)}个场景）': (r, np.arange(1, len(r)+1)/len(r)), **curves[j]}
        plot(scene, data, curves, objs, out)
    t = pd.DataFrame(rows); t.to_csv(OUT/'tail_and_ks.csv', index=False); mt = pd.DataFrame(mrows); mt.to_csv(OUT/'moments.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 50)
    print(t.round(4).to_string(index=False))
    print(mt.pivot_table(index=['场景', '对象', 'K'], columns='模型', values='超额峰度').round(2).join(mt.groupby(['场景', '对象', 'K']).MC超额峰度.first().round(2)).to_string())
    print(mt.pivot_table(index=['场景', '对象', 'K'], columns='模型', values='标准差相对误差').round(4).to_string())


if __name__ == '__main__':
    main()
