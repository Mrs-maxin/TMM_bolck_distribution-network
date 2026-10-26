"""Step 2b: file and folder names that follow a Chinese character in prose, and four output names left in programs.

Step 2 replaced names only at a path boundary; this pass handles the names listed by the post-import scan.
"""
import re
from pathlib import Path

TOP = Path(__file__).resolve().parents[2]
CASE = TOP/'shared/result/heavytail_injection/case33bw_20261026'
POST = TOP/'shared/tests/python/heavytail_postprocess'

PROSE_FILES = {'字段说明.csv': 'field_dictionary.csv',
               '自由度_保留电压方向检查.csv': 'dof_retained_voltage_direction_check.csv',
               '固定无功_电压响应抵消.csv': 'fixed_reactive_voltage_response_cancellation.csv'}
PROSE_DIRS = {'历史结果': 'legacy_metrics', '计算记录': 'computation_records', '尾部筛选': 'tail_screening',
              '支路检查': 'branch_check', '结果': 'results'}
CODE = {'branch_check.py': {"'基础核验.json'": "'base_power_calc_check.json'"},
        'branch_fix_incomplete.py': {"'无功CDF.csv'": "'reactive_cdf.csv'", "'无功尾段.csv'": "'reactive_tail_points.csv'",
                                     "'无功分位.csv'": "'reactive_quantiles.csv'", "'无功积分.csv'": "'reactive_integration.csv'"}}


def main():
    targets = [p for p in CASE.rglob('*') if p.suffix in {'.md', '.html'} or p.name in {'file_index.csv', 'field_dictionary.csv'}]
    targets = [p for p in targets if not {'run_outputs', 'audit_theory', 'records'} & set(p.relative_to(CASE).parts)]
    targets += sorted(POST.glob('*.py'))
    for p in targets:
        text = p.read_text(encoding='utf-8')
        new = text
        for old, rep in PROSE_FILES.items():
            new = new.replace(old, rep)
        for old, rep in PROSE_DIRS.items():
            new = re.sub(r'(?<=[\u4e00-\u9fff])' + old + r'(?=/)', rep, new)
        new = new.replace('../../../tests/python/heavytail_postprocess/整理结果.py',
                          '../../../../archive/reorganization_20261026/previous_cn_reorganization/reorganize_results_cn.py')
        for old, rep in CODE.get(p.name, {}).items():
            assert old in new, (p.name, old)
            new = new.replace(old, rep)
        if new != text:
            p.write_text(new, encoding='utf-8')
            print('updated', p.relative_to(TOP))


if __name__ == '__main__':
    main()
