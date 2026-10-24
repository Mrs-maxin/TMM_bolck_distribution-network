"""Injection-to-voltage fast mapping for the fitting scenarios (PFIGMM power-to-voltage mapping).

Calls tests/map_injections_fast_case33.m: adaptive anchors (one AC power flow and one Jacobian factorization
each), exact quadratic AC power map and fixed-Jacobian correction for all scenarios in one batch. The MC
method (per-scenario AC power flow) is kept only for the reference distribution.
target: (n, 64) reduced injections [P_2..P_33, Q_2..Q_33] in p.u.; returns complex bus voltages (n, 33).
"""
import os, json, hashlib, subprocess
from pathlib import Path
import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO/'cache/fast_map'
MATLAB = os.environ.get('MATLAB_BIN', '/Applications/MATLAB_R2026a.app/bin/matlab')
MATPOWER = os.environ.get('MATPOWER_ROOT', str(REPO.parents[1]/'00codezip/matpower8.0'))


def fast_map(target):
    target = np.ascontiguousarray(target, dtype=float)
    key = hashlib.sha256(target.tobytes()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE/f'voltage_{key}.mat'; info = CACHE/f'voltage_{key}.json'
    if not out.exists():
        src = CACHE/f'injection_{key}.mat'
        sio.savemat(src, dict(P=np.c_[np.zeros(len(target)), target[:, :32]], Q=np.c_[np.zeros(len(target)), target[:, 32:]]))
        env = dict(os.environ, MATPOWER_ROOT=MATPOWER, FAST_MAP_IN=str(src), FAST_MAP_OUT=str(out))
        run = subprocess.run([MATLAB, '-batch', 'map_injections_fast_case33'], cwd=REPO/'tests', env=env, capture_output=True, text=True)
        log = '\n'.join(l for l in (run.stdout+run.stderr).splitlines() if 'xmlSec' not in l)
        if run.returncode != 0 or not out.exists():
            raise RuntimeError('fast mapping failed:\n'+log)
        src.unlink()
        m = sio.loadmat(out)
        info.write_text(json.dumps(dict(scenarios=len(target), anchors=int(m['anchorCount'].item()),
                        accepted=int(m['acceptedCount'].item()), ac_fallback=int(np.size(m['unresolved'])),
                        map_seconds=float(m['mapSeconds'].item()), fallback_seconds=float(m['fallbackSeconds'].item()),
                        max_power_residual_pu=float(m['maxPowerResidual'].item())), indent=1))
    s = sio.loadmat(out)['state']
    return (s[:33]+1j*s[33:]).T, json.loads(info.read_text())
