"""Package cloud inputs and source; never execute scientific calculations locally."""
import ast
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'result/case33bw/direct_pv8_cloud_package_20261026'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    sources = set((ROOT / 'tests/python').rglob('*.py'))
    sources.update((ROOT / 'data/nrel_sind/pv8_direct_2006').glob('*'))
    sources.update((ROOT / '.cursor/rules').glob('*'))
    sources.update(ROOT / path for path in (
        'AGENTS.md', 'GEMINI.md', 'CLAUDE.md', 'BTMM_note.md', 'BTMM_changelog.md',
        'data/cases/case33bw_modify.m', 'tests/python/requirements.txt',
        'tests/python/direct_pv8_cloud_README.md',
        '01_IDEA/plans/nrel_direct_output_selection_20261026.md',
        '01_IDEA/plans/direct_pv8_python_cloud_20261026.md',
        '01_IDEA/figure_plan.md', '01_IDEA/derivation.md', '01_IDEA/claims.md',
        '02_PAPER/manuscript/method_zh.md', '02_PAPER/config/paper-profile.example.yaml',
        '.cursor/contracts/project-template-handoff.schema.yaml',
    ))
    payload = {str(p.relative_to(ROOT)): p.read_bytes() for p in sorted(sources) if p.is_file()}
    syntax = []
    for name, data in payload.items():
        if name.endswith('.py'):
            ast.parse(data, filename=name)
            compile(data, name, 'exec')  # Compile only: no import or evaluation.
            syntax.append(name)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    record = dict(status='packaged_not_executed', package_date='2026-10-26',
                  packaging_host_platform=platform.platform(), git_head=head,
                  numerical_execution_performed=False, figures_generated=False,
                  validation='AST parsing and Python bytecode compilation only; no modules imported',
                  syntax_checked=syntax,
                  contents=[dict(path=n, bytes=len(v), sha256=digest(v)) for n, v in payload.items()])
    payload['CLOUD_PACKAGE_MANIFEST.json'] = (json.dumps(record, ensure_ascii=False, indent=2)+'\n').encode()
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'direct_pv8_cloud_test.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None, 'ZIP CRC failure'
        for name, data in payload.items():
            assert digest(archive.read(name)) == digest(data), f'Member mismatch: {name}'
    receipt = dict(status='packaged_not_executed', path=str(target.relative_to(ROOT)),
                   bytes=target.stat().st_size, sha256=digest(target.read_bytes()),
                   members=len(payload), python_files_syntax_checked=len(syntax),
                   CRC_verified=True, member_SHA256_verified=True,
                   numerical_execution_performed=False, figures_generated=False)
    (OUT / 'package_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
