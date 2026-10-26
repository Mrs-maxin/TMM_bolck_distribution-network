"""Case 1: full-distribution and tail accuracy of AQGMM, standard TMM and block TMM with the old and new split.

Old split: hand-set PV sensitivity at the forecast point (result/case33bw/pv_nrel_case33_20261022).
New split: derivation.md section J (result/case33bw/pv_nrel_case33_newsplit_20261022). Same inputs,
MC reference and AQGMM/TMM fits. Every model at the K of highest likelihood on the model-selection
scenarios, and at equal K. Metrics: KS and W1 of all 97 quantities; mean and maximum absolute CDF error
over the 5% tail of interest for the 8 target objects.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]; VAL = REPO/'result/case33bw'; SCENE = 'nrel_ha4_40_60'
DIRS = {'old': VAL/'pv_nrel_case33_20261022', 'new': VAL/'pv_nrel_case33_newsplit_20261022'}
MODELS = [('AQGMM', 'old', 'GMM'), ('标准TMM', 'old', 'TMM'), ('分块TMM（旧分块）', 'old', 'BTMM'), ('分块TMM共用残差（旧分块）', 'old', 'SBTMM'),
          ('分块TMM（新分块）', 'new', 'BTMM'), ('分块TMM共用残差（新分块）', 'new', 'SBTMM')]
KEY = ['output', 'node', 'branch']


def summaries(d):
    parts = [pd.read_csv(d/f) for f in ['global_repeat_summary.csv', 'block_global_repeat_summary.csv', 'shared_global_repeat_summary.csv'] if (d/f).exists()]
    return pd.concat(parts, ignore_index=True)


def main():
    targets = pd.read_csv(DIRS['old']/SCENE/'targets.csv')
    gs = {k: summaries(d) for k, d in DIRS.items()}
    curves = {k: pd.read_csv(d/SCENE/'compare_likelihood_curve.csv') for k, d in DIRS.items()}
    best = {}
    for label, src, fam in MODELS:
        c = curves[src][curves[src].family == fam]; r = c.loc[c.validation_ll.idxmax()]
        best[label] = (int(r.K), float(r.validation_ll), int(r.parameter_count))
    def table(label, src, fam, k):
        g = gs[src]; return g[(g.family == fam) & (g.K == k)].set_index(KEY)
    ref = table('AQGMM', 'old', 'GMM', best['AQGMM'][0])
    tid = set(map(tuple, targets[KEY].values))
    rows, per_target = [], []
    for label, src, fam in MODELS:
        k = best[label][0]; t = table(label, src, fam, k)
        tail_mean = [t.loc[tuple(r[KEY])][f'{r.side}_tail_cdf_mean_abs_mean'] for _, r in targets.iterrows()]
        tail_sup = [t.loc[tuple(r[KEY])][f'{r.side}_tail_cdf_sup_abs_mean'] for _, r in targets.iterrows()]
        g_tail = [ref.loc[tuple(r[KEY])][f'{r.side}_tail_cdf_mean_abs_mean'] for _, r in targets.iterrows()]
        others = [i for i in t.index if i not in tid]
        ks, ks_g = t.loc[others].ks_mean, ref.loc[others].ks_mean
        w1r = (t.loc[others].w1_mean/ref.loc[others].w1_mean)
        rows.append({'模型': label, 'K': k, '参数个数': best[label][2], '模型选择场景似然': round(best[label][1], 3),
                     '目标尾段CDF平均误差中位数': np.median(tail_mean), '目标尾段CDF平均误差最大': np.max(tail_mean),
                     '目标尾段CDF最大误差中位数': np.median(tail_sup), '目标尾段优于AQGMM': f"{int(np.sum(np.array(tail_mean) < np.array(g_tail)))}/8",
                     '其余对象KS中位数': ks.median(), '其余对象KS最大': ks.max(), '其余对象KS不超过AQGMM1.2倍': float(np.mean(ks <= 1.2*ks_g)),
                     '其余对象W1与AQGMM之比中位数': w1r.median(), '全部对象KS中位数': t.ks_mean.median()})
        for (_, r), a, b in zip(targets.iterrows(), tail_mean, tail_sup):
            per_target.append({'模型': label, '对象': f"{r.output}{int(r.node) if r.output == 'V' else int(r.branch)}", '尾侧': r.side,
                               '尾段CDF平均误差': a, '尾段CDF最大误差': b, 'KS': t.loc[tuple(r[KEY])].ks_mean})
    out = DIRS['new']/SCENE/'split_comparison'; out.mkdir(exist_ok=True)
    s = pd.DataFrame(rows); s.to_csv(out/'summary_best_K.csv', index=False)
    p = pd.DataFrame(per_target); p.to_csv(out/'targets_best_K.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(s.round(4).to_string(index=False))
    print(p.pivot(index='对象', columns='模型', values='尾段CDF平均误差').round(4).to_string())
    same = []
    for k in [1, 2, 3, 4, 6, 8, 12]:
        for label, src, fam in MODELS:
            t = table(label, src, fam, k)
            if t.empty:
                continue
            tm = [t.loc[tuple(r[KEY])][f'{r.side}_tail_cdf_mean_abs_mean'] for _, r in targets.iterrows()]
            same.append({'K': k, '模型': label, '目标尾段CDF平均误差中位数': np.median(tm),
                         '其余对象KS中位数': t.loc[[i for i in t.index if i not in tid]].ks_mean.median()})
    sm = pd.DataFrame(same); sm.to_csv(out/'same_K.csv', index=False)
    print(sm.pivot(index='K', columns='模型', values='目标尾段CDF平均误差中位数').round(4).to_string())
    print(sm.pivot(index='K', columns='模型', values='其余对象KS中位数').round(4).to_string())


if __name__ == '__main__':
    main()
