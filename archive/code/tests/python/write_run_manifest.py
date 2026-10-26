"""Write run_manifest.json for one result directory of run_pv_fixed_forecast.py.

Records the entry, the executed commands with their timing (from logs/*.txt), the Git state, the SHA256 of
every result file (large arrays are listed although .gitignore keeps them out of the repository), the
scripts, the plan and the case data.
"""
import argparse, hashlib, json, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--note', default='')
    args = parser.parse_args()
    out = Path(args.directory).resolve()
    commands = []
    for log in sorted((out/'logs').glob('*.txt')):
        lines = log.read_text(errors='replace').splitlines()
        row = dict(log=str(log.relative_to(REPO)))
        for line in lines:
            if line.startswith('COMMAND '):
                row['command'] = line[8:]
            elif line.startswith('START '):
                row['start_utc'] = line[6:]
            elif line.startswith('END '):
                parts = line.split()
                row.update(end_utc=parts[1], exit_code=int(parts[3]), seconds=int(parts[5]))
        if 'command' in row:
            commands.append(row)
    ignored = set(git('ls-files', '--others', '--ignored', '--exclude-standard', '--directory', str(out.relative_to(REPO))).splitlines())
    files = []
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name != 'run_manifest.json':
            rel = str(p.relative_to(REPO))
            files.append(dict(path=rel, bytes=p.stat().st_size, SHA256=sha(p),
                              kept_out_of_git=rel in ignored or p.suffix in ('.npz', '.npy')))
    protocol = json.loads((out/'protocol.json').read_text()) if (out/'protocol.json').exists() else {}
    scripts = sorted((REPO/'tests/python').glob('*.py')) + sorted((REPO/'tests/python/archived').glob('*'))
    manifest = dict(
        entry='tests/python/run_pv_fixed_forecast.py',
        case=protocol.get('case'), scene=protocol.get('scene'), plan_section=protocol.get('plan_section'),
        written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        git=dict(branch=git('rev-parse', '--abbrev-ref', 'HEAD'), commit=git('rev-parse', 'HEAD'),
                 status_short=git('status', '--short', '--untracked-files=no').splitlines()),
        protocol_script_SHA256_at_ac_only=protocol.get('script_SHA256'),
        scripts_SHA256={str(p.relative_to(REPO)): sha(p) for p in scripts if p.is_file()},
        plan_SHA256=sha(REPO/'01_IDEA/pv_fixed_forecast_common_scale_plan_20261021.md'),
        case_file_SHA256={protocol.get('case_file', ''): protocol.get('case_file_SHA256')},
        commands=commands, total_logged_seconds=sum(c.get('seconds', 0) for c in commands),
        files=files, note=args.note)
    (out/'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(out/'run_manifest.json', len(files), 'files', len(commands), 'commands')


if __name__ == '__main__':
    main()
