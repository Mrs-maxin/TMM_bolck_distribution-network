"""Step 1 of the 2026-10-26 two-paper reorganization: move repository content into shared/, archive/.

Moves are renames on the same volume, so file bytes cannot change; every moved file is still listed with
its old and new path in step1_move_map.csv. The original combined 01_IDEA, 02_PAPER, note, changelog and
blog are frozen under archive/combined_paper_20261026/ before the two paper folders are created.
"""
import csv
import shutil
from pathlib import Path

TOP = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

ACTIVE_SCRIPTS = {
    'analytic_grouped_t_case33.py', 'block_split_rule_check.py', 'build_direct_pv8_cloud_bundle.py',
    'easyplot_py.py', 'fast_voltage_map.py', 'fit_nrel_pv_forecast_error_case33.py',
    'grouped_strength_tmm_case33.py', 'mainline_comparison_case33.py', 'moment_matched_tmm_case33.py',
    'run_direct_pv8_python_cloud.py', 'run_pv_fixed_forecast_case33.py', 'run_pv_nrel_case33.py',
    'select_nrel_direct_pv8.py', 'shared_residual_mixture.py', 'shift_grouped_t_case33.py',
    'truncated_strength_case33.py', 'requirements.txt', 'direct_pv8_cloud_README.md'}

MOVES = [
    ('+BTMM_core', 'shared/+BTMM_core'),
    ('+BTMM_utils', 'shared/+BTMM_utils'),
    ('+BTMM_sota', 'shared/+BTMM_sota'),
    ('BTMM.m', 'shared/BTMM.m'),
    ('BTMM_case33bw.m', 'shared/BTMM_case33bw.m'),
    ('BTMM_case141.m', 'shared/BTMM_case141.m'),
    ('03_REFERENCE', 'shared/03_REFERENCE'),
    ('data/README.md', 'shared/data/README.md'),
    ('data/cases', 'shared/data/cases'),
    ('data/nrel_sind', 'shared/data/nrel_sind'),
    ('data/experiments', 'archive/data/experiments'),
    ('cache/README.md', 'shared/cache/README.md'),
    ('cache/fast_map', 'shared/cache/fast_map'),
    ('cache/record_scenarios_20261024', 'archive/cache/record_scenarios_20261024'),
    ('cache/record_scenarios_20261024_band40_60', 'archive/cache/record_scenarios_20261024_band40_60'),
    ('cache/record_scenarios_20261024_band60_80_cap0p5805', 'archive/cache/record_scenarios_20261024_band60_80_cap0p5805'),
    ('result/README.md', 'shared/result/README.md'),
    ('result/case33bw/direct_pv8_cloud_package_20261026', 'shared/result/actual_output/case33bw_pv8_2006/cloud_package_20261026'),
    ('result/case33bw', 'archive/result/case33bw'),
    ('result/case141', 'archive/result/case141'),
    ('tests/README.md', 'shared/tests/README.md'),
    ('tests/map_injections_fast_case33.m', 'shared/tests/map_injections_fast_case33.m'),
    ('tests/python/archived', 'shared/tests/python/archived'),
    ('tests/python/result', 'archive/result/tests_python_result'),
    ('01_IDEA/plans/nrel_direct_output_selection_20261026.md', 'shared/protocols/actual_output_pv8_2006_selection_20261026.md'),
    ('01_IDEA/plans/direct_pv8_python_cloud_20261026.md', 'shared/protocols/actual_output_pv8_2006_python_cloud_20261026.md'),
    ('01_IDEA', 'archive/combined_paper_20261026/01_IDEA'),
    ('02_PAPER', 'archive/combined_paper_20261026/02_PAPER'),
    ('BTMM_note.md', 'archive/combined_paper_20261026/BTMM_note.md'),
    ('BTMM_changelog.md', 'archive/combined_paper_20261026/BTMM_changelog.md'),
    ('BTMM_blog.md', 'archive/combined_paper_20261026/BTMM_blog.md'),
    ('README.md', 'archive/combined_paper_20261026/README_root.md'),
    ('error.log', 'archive/misc/codex_cloud_error_20261026.log'),
]


def files_under(path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob('*') if p.is_file() and p.name != '.DS_Store')


def move(src_rel, dst_rel, rows):
    src, dst = TOP/src_rel, TOP/dst_rel
    assert src.exists(), f'missing source {src_rel}'
    assert not dst.exists(), f'target exists {dst_rel}'
    for f in files_under(src):
        rows.append((str(f.relative_to(TOP)), str((dst/f.relative_to(src)) if src.is_dir() else dst).replace(str(TOP)+'/', '')))
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def main():
    rows = []
    shutil.rmtree(TOP/'tests/python/__pycache__', ignore_errors=True)
    for name in sorted(p.name for p in (TOP/'tests/python').iterdir() if p.is_file()):
        target = 'shared/tests/python' if name in ACTIVE_SCRIPTS else 'archive/code/tests/python'
        move(f'tests/python/{name}', f'{target}/{name}', rows)
    for src, dst in MOVES:
        move(src, dst, rows)
    for p in sorted(TOP.rglob('.DS_Store')):
        if '.git' not in p.parts:
            p.unlink()
    for d in ['tests/python', 'tests', 'data', 'cache', 'result']:
        p = TOP/d
        if p.exists():
            leftovers = [q for q in p.rglob('*') if q.is_file()]
            assert not leftovers, f'unexpected files left in {d}: {leftovers[:5]}'
            shutil.rmtree(p)
    with (HERE/'step1_move_map.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['old_path', 'new_path'])
        w.writerows(rows)
    print(f'moved {len(rows)} files')


def finish_after_sandbox_stop():
    """The first run moved everything, then stopped at a sandbox-protected .DS_Store; rebuild the map from targets."""
    rows = []
    for f in files_under(TOP/'shared/tests/python'):
        if f.parent == TOP/'shared/tests/python':
            rows.append((f'tests/python/{f.name}', str(f.relative_to(TOP))))
    for f in files_under(TOP/'archive/code/tests/python'):
        rows.append((f'tests/python/{f.name}', str(f.relative_to(TOP))))
    for src, dst in MOVES:
        d = TOP/dst
        for f in files_under(d):
            rows.append((src if d.is_file() else str(Path(src)/f.relative_to(d)), str(f.relative_to(TOP))))
    for p in sorted(TOP.rglob('.DS_Store')):
        if '.git' not in p.parts:
            p.unlink()
    for d in ['tests/python', 'tests', 'data', 'cache', 'result']:
        p = TOP/d
        if p.exists():
            leftovers = [q for q in p.rglob('*') if q.is_file()]
            assert not leftovers, f'unexpected files left in {d}: {leftovers[:5]}'
            shutil.rmtree(p)
    with (HERE/'step1_move_map.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['old_path', 'new_path'])
        w.writerows(rows)
    print(f'moved {len(rows)} files')


if __name__ == '__main__':
    import sys
    finish_after_sandbox_stop() if '--finish' in sys.argv else main()
