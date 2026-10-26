"""Split three-method comparison tables of one case into the tables used by each paper.

The standard-TMM paper compares AQGMM with standard TMM; the grouped-t paper compares AQGMM, standard TMM and
the grouped t mixture. Long tables keep the rows of the paper's methods. Wide tables drop the grouped-t columns
for the standard-TMM paper, whose ordering, difference and win-count columns are then recomputed from the
remaining two methods with the same strict "<" rule as the original tables. No error value is recomputed.

Usage: python3 split_results_by_paper.py heavytail_injection/case33bw_20261026
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

SHARED = Path(__file__).resolve().parents[2]
TOP = SHARED.parent
PAPERS = {
    'paper_standardTMM': dict(methods=['AQGMM', '标准TMM'], models=['AQGMM', 'standard_TMM']),
    'paper_groupedT': dict(methods=['AQGMM', '标准TMM', '分组t'], models=['AQGMM', 'standard_TMM', 'grouped_t_shift']),
}
TABLE_DIRS = ['results', 'tail_screening', 'tail_distribution', 'model_parameter_checks']
GROUPED_ONLY = {'model_parameter_checks/grouped_t_voltage_fourth_moments.csv',
                'model_parameter_checks/grouped_t_strength_moments.csv',
                'model_parameter_checks/source_groups_residual_voltage_directions.csv'}
FIGURE_INDEXES = {'tail_distribution/figure_index.csv'}
WIN_COUNTS = {'tail_screening/tail_cdf_rmse_win_counts_by_K.csv', 'tail_screening/extreme_quantile_win_counts_by_K_level.csv'}
EXTRA_FILES = ['case_settings.csv', 'error_metrics.md', 'injection_power_pdf.png']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order(row, cols):
    pairs = sorted(cols.items(), key=lambda kv: row[kv[1]])
    text = pairs[0][0]
    for (_, a), (name, b) in zip(pairs, pairs[1:]):
        text += (' = ' if row[a] == row[b] else ' < ') + name
    return text


def two_method_wide(df, rel):
    df = df[[c for c in df.columns if '分组t' not in c]].copy()
    if 'AQGMM_尾部CDF_RMSE_百分点' in df:
        a, s = 'AQGMM_尾部CDF_RMSE_百分点', '标准TMM_尾部CDF_RMSE_百分点'
        df['误差数值由小到大'] = df.apply(order, axis=1, cols={'AQGMM': a, '标准TMM': s})
        df['标准TMM减AQGMM_百分点'] = df[s]-df[a]
        df['标准TMM误差小于AQGMM'] = df[s] < df[a]
    if '绝对分位误差_AQGMM' in df:
        a, s = '绝对分位误差_AQGMM', '绝对分位误差_标准TMM'
        df['绝对误差数值由小到大'] = df.apply(order, axis=1, cols={'AQGMM': a, '标准TMM': s})
        df['标准TMM绝对误差小于AQGMM'] = df[s] < df[a]
    return df


def standard_win_counts(case, rel):
    if rel.endswith('tail_cdf_rmse_win_counts_by_K.csv'):
        w = pd.read_csv(case/'tail_screening/tail_cdf_rmse_method_compare.csv')
        g = w.assign(win=w['标准TMM_尾部CDF_RMSE_百分点'] < w['AQGMM_尾部CDF_RMSE_百分点']).groupby('K')
        return pd.DataFrame({'指标': '尾部CDF-RMSE', 'K': g.size().index, '比较项数': g.size().values,
                             '标准TMM误差小于AQGMM项数': g.win.sum().values,
                             '每项含义': '注入情形×电气量×尾侧；只含MC尾部延伸标记'})
    w = pd.read_csv(case/'tail_screening/extreme_quantile_error_method_compare.csv')
    g = w.assign(win=w['绝对分位误差_标准TMM'] < w['绝对分位误差_AQGMM']).groupby(['K', '分位水平'])
    out = g.size().rename('比较项数').reset_index()
    out['标准TMM绝对误差小于AQGMM项数'] = g.win.sum().values
    return out


def split(case_rel):
    case = SHARED/'result'/case_rel
    assert case.is_dir(), case
    tables = sorted(p for d in TABLE_DIRS for p in (case/d).glob('*.csv'))
    for paper, spec in PAPERS.items():
        out = TOP/paper/'result'/case_rel
        if out.exists():
            shutil.rmtree(out)
        rows = []
        for src in tables:
            rel = str(src.relative_to(case))
            if rel in FIGURE_INDEXES or (rel in GROUPED_ONLY and paper == 'paper_standardTMM'):
                continue
            dst = out/rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(src, encoding='utf-8-sig')
            if paper == 'paper_standardTMM' and rel in WIN_COUNTS:
                df, how = standard_win_counts(case, rel), 'win_counts_recomputed_two_methods'
            elif '方法' in df:
                n = len(df)
                df, how = df[df['方法'].isin(spec['methods'])], 'rows_filtered'
                assert len(df) and (len(df) < n or paper == 'paper_groupedT'), rel
            elif 'model' in df:
                df, how = df[df['model'].isin(spec['models'])], 'rows_filtered'
            elif any('分组t' in c for c in df.columns) and paper == 'paper_standardTMM':
                df, how = two_method_wide(df, rel), 'grouped_t_columns_removed'
            else:
                how = 'copied'
            if how == 'copied':
                shutil.copyfile(src, dst)
            else:
                df.to_csv(dst, index=False, encoding='utf-8-sig')
            rows.append(dict(table=rel, source_SHA256=sha(src), SHA256=sha(dst), rows=len(df), treatment=how))
        for name in EXTRA_FILES:
            if (case/name).exists():
                shutil.copyfile(case/name, out/name)
                rows.append(dict(table=name, source_SHA256=sha(case/name), SHA256=sha(out/name), rows='', treatment='copied'))
        manifest = dict(case=case_rel, source=str(case.relative_to(TOP)), methods=spec['methods'],
                        command=f'python3 shared/tests/python/split_results_by_paper.py {case_rel}', tables=rows)
        (out/'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        print(paper, len(rows), 'files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', help='case path under shared/result, e.g. heavytail_injection/case33bw_20261026')
    split(parser.parse_args().case)
