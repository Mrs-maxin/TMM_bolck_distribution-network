"""Step 2 of the 2026-10-26 reorganization: import /Users/maxin/Documents/Codex/IEEE33重尾 with English names.

Every source file is copied, then source and target SHA256 are compared. Only reading entries, indexes, figure
manifests and post-processing programs have their file-name strings rewritten; their original SHA256 is kept in
step2_import_map.csv. Run logs, numerical tables, model files and MC arrays are copied byte for byte.
"""
import csv
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

TOP = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC = Path('/Users/maxin/Documents/Codex/IEEE33重尾')
CASE = 'shared/result/heavytail_injection/case33bw_20261026'
SNAP = 'archive/heavytail_source_snapshot_20261026'
POST = 'shared/tests/python/heavytail_postprocess'
CASE_TO_SHARED = '../../../'
CASE_TO_SNAP = '../../../../' + SNAP + '/'
CASE_TO_POST = '../../../tests/python/heavytail_postprocess/'

HEAVYTAIL_SCRIPTS = ['run_direct_heavytail_case33.py', 'report_direct_heavytail_case33.py',
                     'finish_direct_heavytail_distributions.py', 'check_direct_heavytail_integration.py',
                     'check_direct_heavytail_probabilities.py', 'check_fixed_reactive_injection.py',
                     'check_statewise_inversion.py']
GENERALIZED_SCRIPTS = ['block_split_rule_check.py', 'fast_voltage_map.py']
RECORDS = ['NEW_TEST_README_zh.md', 'THEORY_VALIDATION_STATUS.json', 'archive_receipt.json',
           'source_snapshot_manifest.json', 'direct_heavytail_review.zip']

DIRS = {
    '历史结果': 'legacy_metrics', '尾部分布': 'tail_distribution', '尾部筛选': 'tail_screening',
    '模型参数核查': 'model_parameter_checks', '结果': 'results', '计算记录': 'computation_records',
    '诊断记录': 'diagnostics', '图': 'figures', '支路检查': 'branch_check', '根因检查': 'root_cause_check',
    '未完成解析检查': 'incomplete_analytic_check', '旧程序缓存': 'old_bytecode_cache'}

FILES = {
    '说明.md': 'README.md', '字段说明.csv': 'field_dictionary.csv', '文件索引.csv': 'file_index.csv',
    '注入功率PDF.png': 'injection_power_pdf.png', '算例设置.csv': 'case_settings.csv',
    '结果.html': 'results_browser.html', '误差标准.md': 'error_metrics.md',
    '尾部CDF_MAE_全部电气量.csv': 'tail_cdf_mae_all_quantities.csv',
    '尾部CDF_MAE_筛选电气量.csv': 'tail_cdf_mae_screened_quantities.csv',
    'MC极端分位值及95区间.csv': 'mc_extreme_quantiles_ci95.csv',
    'MC电气量分布特征.csv': 'mc_quantity_distribution_features.csv',
    '全分布均值方差_三方法对比.csv': 'full_mean_variance_method_compare.csv',
    '全分布均值方差_全部电气量.csv': 'full_mean_variance_all_quantities.csv',
    '尾部CDF_RMSE_三方法对比.csv': 'tail_cdf_rmse_method_compare.csv',
    '尾部CDF_RMSE_全部电气量.csv': 'tail_cdf_rmse_all_quantities.csv',
    '尾部CDF取值与概率误差_逐阈值.csv': 'tail_cdf_per_threshold.csv',
    '尾部两类误差_合并查看表.csv': 'tail_two_metrics_merged.csv',
    '尾部分位区间W1_三方法对比.csv': 'tail_quantile_w1_method_compare.csv',
    '尾部分位区间W1_全部电气量.csv': 'tail_quantile_w1_all_quantities.csv',
    '极端分位数误差_三方法对比.csv': 'extreme_quantile_error_method_compare.csv',
    '极端分位数误差_全部电气量.csv': 'extreme_quantile_error_all_quantities.csv',
    '电气量编号与单位.csv': 'quantity_ids_units.csv',
    '尾部CDF_RMSE_筛选电气量.csv': 'tail_cdf_rmse_screened_quantities.csv',
    '尾部CDF_RMSE比较项数_按K.csv': 'tail_cdf_rmse_win_counts_by_K.csv',
    '尾部两类误差_筛选合并查看表.csv': 'tail_two_metrics_screened_merged.csv',
    '尾部延伸电气量清单.csv': 'tail_extended_quantity_list.csv',
    '极端分位数误差_筛选电气量.csv': 'extreme_quantile_error_screened_quantities.csv',
    '极端分位误差比较项数_按K及分位水平.csv': 'extreme_quantile_win_counts_by_K_level.csv',
    '分组t_电压四阶统计量.csv': 'grouped_t_voltage_fourth_moments.csv',
    '分组t_随机强度矩.csv': 'grouped_t_strength_moments.csv',
    '标准TMM_自由度与四阶统计量.csv': 'standard_TMM_dof_fourth_moments.csv',
    '电压分布_均值协方差三阶矩.csv': 'voltage_mean_cov_third_moments.csv',
    '电源分组及剩余电压方向.csv': 'source_groups_residual_voltage_directions.csv',
    'MC全分布均值方差_原单位.csv': 'mc_full_mean_variance_raw_units.csv',
    'PDF区间平均密度数据.csv': 'pdf_bin_mean_density.csv',
    'PDF文件核验.json': 'pdf_file_check.json',
    'PDF显示范围与数值检查.csv': 'pdf_display_range_check.csv',
    '全部电气量_MC尾部特征.csv': 'mc_tail_features_all_quantities.csv',
    '图册.md': 'figure_atlas.md', '图形索引.csv': 'figure_index.csv',
    '尾部延伸较弱的比较电气量.csv': 'weak_tail_reference_quantities.csv',
    '核验.json': 'verification.json', '独立公式核验.json': 'independent_formula_check.json',
    '绘图字体检查.json': 'plot_font_check.json', '设置.json': 'settings.json',
    '重点位置汇总.csv': 'focus_location_summary.csv',
    '重点电气量_MC先行选择.csv': 'focus_quantities_mc_selection.csv',
    '重点电气量_MC极端场景的方差贡献.csv': 'focus_quantities_mc_extreme_variance_share.csv',
    '重点电气量_三项误差及均值方差.csv': 'focus_quantities_three_metrics_mean_variance.csv',
    '重点电气量_全分布均值方差比较.csv': 'focus_quantities_full_mean_variance_method_compare.csv',
    '重点电气量_尾部分位区间W1比较.csv': 'focus_quantities_tail_quantile_w1_method_compare.csv',
    'MC比较图出图.log': 'mc_comparison_plot.log', 'MC表格更新.log': 'mc_table_update.log',
    'PDF出图.log': 'pdf_plot.log', '最终出图.log': 'final_plot.log',
    '冻结存档SHA256清单.csv': 'frozen_archive_sha256.csv',
    '分位数解析计算记录.jsonl': 'analytic_quantile_log.jsonl',
    '原尾部筛选核验.json': 'original_tail_screening_check.json',
    '原指标重算核验.json': 'original_metric_recompute_check.json',
    '原根因检查核验.json': 'original_root_cause_check.json',
    '整理前尾部CDF_RMSE比较表.csv': 'pre_reorg_tail_cdf_rmse_compare.csv',
    '整理前极端分位数比较表.csv': 'pre_reorg_extreme_quantile_compare.csv',
    '文件改名对照.csv': 'file_rename_map_20261026_cn.csv',
    '重算.cpython-313.pyc': 'recompute.cpython-313.pyc',
    '早期分位数计算检查点.jsonl': 'early_quantile_checkpoint.jsonl',
    '最初归档核验.json': 'initial_archive_check.json', '结果整理核验.json': 'results_reorg_check.json',
    '解析CDF与分位数求解检查.csv': 'analytic_cdf_quantile_solve_check.csv',
    '频率积分加密记录.jsonl': 'frequency_integration_refinement_log.jsonl',
    '频率积分抽查汇总.json': 'frequency_integration_spot_check.json',
    '三相单位换算与功率公式.csv': 'three_phase_unit_power_formula.csv',
    '同一电压分布_解析CDF与直接计算比较.csv': 'same_voltage_law_analytic_vs_direct_cdf.csv',
    '基础功率计算核验.json': 'base_power_calc_check.json', '核查设置.json': 'check_settings.json',
    '独立交流潮流与原MC比较.csv': 'independent_acpf_vs_original_mc.csv',
    'MC末端无功平衡核验.json': 'mc_end_reactive_balance_check.json',
    '固定无功_电压响应抵消.csv': 'fixed_reactive_voltage_response_cancellation.csv',
    '支路17无功_物理条件比较.csv': 'branch17_Q_physical_condition_compare.csv',
    '支路17无功_逐阈值CDF比较.csv': 'branch17_Q_per_threshold_cdf.csv',
    '自由度_保留电压方向检查.csv': 'dof_retained_voltage_direction_check.csv',
    '节点18电压_分量偏度与四阶矩.csv': 'bus18_V_component_skew_fourth_moment.csv',
    '节点18电压_自由度改变.csv': 'bus18_V_dof_change.csv',
    '节点18电压_自由度改变后的误差.csv': 'bus18_V_error_after_dof_change.csv',
}
PROGRAMS = {
    '均值方差.py': 'mean_variance.py', '尾部W1.py': 'tail_w1.py', '尾部分布报告.py': 'tail_distribution_report.py',
    '尾部分布核验.py': 'tail_distribution_verify.py', '尾部分布绘图.py': 'tail_distribution_plot.py',
    '尾部分析.py': 'tail_analysis.py', '支路修正_未完成.py': 'branch_fix_incomplete.py',
    '支路核查.py': 'branch_check.py', '查看.py': 'results_browser.py', '根因.py': 'root_cause.py',
    '生成说明.py': 'write_readmes.py', '积分检查.py': 'integration_check.py', '重算.py': 'recompute.py'}
FIGURE_SUFFIX = {'全分布PDF': 'full_pdf', '尾部PDF放大': 'tail_pdf_zoom', 'MC标准化PDF比较': 'mc_standardized_pdf_compare',
                 'MC标准化尾部PDF比较': 'mc_standardized_tail_pdf_compare', '全部电气量尾部比较': 'all_quantities_tail_compare'}
FILES.update(PROGRAMS)

REWRITE_SUFFIX = {'.md', '.py', '.html'}
REWRITE_NAMES = {'file_index.csv', 'field_dictionary.csv', 'figure_index.csv', 'pdf_file_check.json',
                 'verification.json', 'settings.json', 'independent_formula_check.json', 'plot_font_check.json'}
CN = re.compile(r'[\u4e00-\u9fff]')


def component(name):
    if name in DIRS:
        return DIRS[name]
    if name in FILES:
        return FILES[name]
    m = re.match(r'^(.+?)_(' + '|'.join(sorted(map(re.escape, FIGURE_SUFFIX), key=len, reverse=True)) + r')(\..+)$', name)
    if m:
        return f'{m.group(1)}_{FIGURE_SUFFIX[m.group(2)]}{m.group(3)}'
    assert not CN.search(name), f'no English name for {name}'
    return name


def target(rel):
    parts = rel.parts
    if parts[0] == '存档':
        sub = Path(*parts[1:])
        s = str(sub)
        if s.startswith('result/case33bw/direct_heavytail_20261026/'):
            return Path(CASE)/'run_outputs'/Path(*sub.parts[3:])
        if s.startswith('audit_theory_20261026/'):
            return Path(CASE)/'audit_theory'/Path(*sub.parts[1:])
        if s == '01_IDEA/direct_heavytail_protocol_20261026.json':
            return Path('shared/protocols/heavytail_injection_case33bw_20261026_protocol.json')
        if s == '01_IDEA/direct_heavytail_scope_20261026.md':
            return Path('shared/protocols/heavytail_injection_case33bw_20261026_scope.md')
        if s.startswith('cache/fast_map/'):
            return Path('shared/cache/fast_map')/sub.name
        if s in RECORDS:
            return Path(CASE)/'records'/s
        return Path(SNAP)/sub
    if parts[0] == '程序':
        if parts[1] == '整理结果.py':
            return HERE/'previous_cn_reorganization/reorganize_results_cn.py'
        return Path(POST)/PROGRAMS[parts[1]]
    return Path(CASE).joinpath(*[component(p) for p in parts])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def rewrite(text, is_py):
    text = text.replace('存档/result/case33bw/direct_heavytail_20261026', 'run_outputs')
    text = text.replace('存档/audit_theory_20261026', 'audit_theory')
    if is_py:
        text = text.replace("ROOT=Path(__file__).resolve().parents[1]\n",
                            "ROOT=Path(__file__).resolve().parents[3]/'result/heavytail_injection/case33bw_20261026'\n")
        text = text.replace("ARCH=ROOT/'存档'\n", "ARCH=ROOT/'" + CASE_TO_SNAP.rstrip('/') + "'\n")
        text = text.replace("OLD=ARCH/'result/case33bw/direct_heavytail_20261026'\n", "OLD=ROOT/'run_outputs'\n")
    text = re.sub(r'(?<![\w/.])((?:\.\./)*)存档/', lambda m: m.group(1) + CASE_TO_SNAP, text)
    text = re.sub(r'(?<![\w/.])((?:\.\./)*)程序/', lambda m: m.group(1) + CASE_TO_POST, text)
    for old in sorted(FILES, key=len, reverse=True):
        text = re.sub(r'(?<![\w])' + re.escape(old), FILES[old], text)
    for old, new in FIGURE_SUFFIX.items():
        text = text.replace('_' + old, '_' + new)
    for old, new in DIRS.items():
        text = re.sub(r'(?<=[/\'"`(,\s])' + re.escape(old) + r'(?=[/\'"`)])', new, text)
        text = re.sub(r'(?m)^' + re.escape(old) + r'(?=/)', new, text)
    if is_py:
        for old, new in PROGRAMS.items():
            text = re.sub(r'(?<=import )' + re.escape(old[:-3]) + r'\b', new[:-3], text)
            text = re.sub(r'(?<=from )' + re.escape(old[:-3]) + r'(?= import)', new[:-3], text)
    return text


def main():
    rows = []
    sources = sorted(p for p in SRC.rglob('*') if p.is_file() and p.name != '.DS_Store'
                     and ('__pycache__' not in p.parts or p.parts[len(SRC.parts)] == '存档'))
    for src in sources:
        rel = src.relative_to(SRC)
        dst = target(rel)
        dst = dst if dst.is_absolute() else TOP/dst
        before = sha(src)
        if str(rel).startswith('存档/cache/fast_map/') and dst.exists():
            assert sha(dst) == before, f'collision {dst}'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        assert sha(dst) == before, f'copy mismatch {rel}'
        edited = False
        if str(rel) != '程序/整理结果.py' and not str(rel).startswith('存档/') and \
                (dst.suffix in REWRITE_SUFFIX or dst.name in REWRITE_NAMES or dst.name.endswith('.manifest.json')):
            text = dst.read_text(encoding='utf-8')
            new = rewrite(text, dst.suffix == '.py')
            if new != text:
                dst.write_text(new, encoding='utf-8')
                edited = True
        rows.append((str(rel), str(dst.relative_to(TOP)), before, sha(dst), 'paths_rewritten' if edited else 'identical'))
        if src.parent.name == 'python' and src.name in HEAVYTAIL_SCRIPTS + GENERALIZED_SCRIPTS and str(rel).startswith('存档/tests/'):
            active = TOP/'shared/tests/python'/src.name
            git = subprocess.run(['git', 'show', f'f341202:tests/python/{src.name}'], cwd=TOP, capture_output=True)
            old = hashlib.sha256(git.stdout).hexdigest() if git.returncode == 0 else ''
            shutil.copy2(src, active)
            rows.append((str(rel), str(active.relative_to(TOP)), before, sha(active),
                         f'active_copy_replaced_sha256={old}' if old else 'active_copy'))
    frozen = TOP/CASE/'computation_records/frozen_archive_sha256.csv'
    with frozen.open(encoding='utf-8-sig') as f:
        entries = list(csv.DictReader(f))
    with frozen.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['original_path', 'path', 'bytes', 'SHA256'])
        for e in entries:
            new = TOP/target(Path(e['path']))
            assert sha(new) == e['SHA256'], f'frozen file changed: {e["path"]}'
            w.writerow([e['path'], os.path.relpath(new, TOP/CASE), e['bytes'], e['SHA256']])
    rows = [r[:3] + (sha(frozen), 'original_path_column_added') if r[1] == str(frozen.relative_to(TOP)) else r for r in rows]
    print(len(entries), 'frozen files verified at new paths')
    with (HERE/'step2_import_map.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source_path_under_IEEE33重尾', 'new_path', 'source_SHA256', 'new_SHA256', 'status'])
        w.writerows(rows)
    bad = [r for r in rows if r[4] == 'identical' and r[2] != r[3]]
    assert not bad, bad[:3]
    print(len(rows), 'rows;', sum(r[4] == 'paths_rewritten' for r in rows), 'rewritten')


if __name__ == '__main__':
    main()
