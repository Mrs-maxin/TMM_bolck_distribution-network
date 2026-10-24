"""Apply the block split rule of derivation.md section J to source-load scenarios with known answers.

A weather-driven loads; B PV and loads amplified by one weather strength; C PV and loads amplified by
two independent strengths; D large ordinary correlated Gaussian PV and load deviations (no common
amplification); E the frozen AQGMM 33-bus input (data/experiments/ieee33_20261004), whose bus
injections are recovered from the stored voltages. The rule sees only the injection scenarios and
their voltages; the expected split is printed for checking only.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import sys, json
sys.dont_write_bytecode = True
import numpy as np
import block_split_rule_check as chk
base = chk.base

LOAD_BUSES = [7, 8, 24, 25, 29, 30, 31, 32]
N_GROUP, N_VOLT = 45056, 8192


def inputs(n, rng, pv_sd, load_sd, pv_strength, load_strength, same_strength):
    lp, lq = 0.8*base.BUS[1:, 2], 0.8*base.BUS[1:, 3]; idx = np.array(LOAD_BUSES)-2
    s1 = np.exp(pv_strength*rng.standard_normal(n)-pv_strength**2/2)
    s2 = s1 if same_strength else np.exp(load_strength*rng.standard_normal(n)-load_strength**2/2)
    if load_strength == 0:
        s2 = np.ones(n)
    if pv_strength == 0:
        s1 = np.ones(n)
    zp = rng.standard_normal((n, 8))@np.linalg.cholesky(0.5+0.5*np.eye(8)).T
    zl = rng.standard_normal((n, len(idx)))@np.linalg.cholesky(0.6+0.4*np.eye(len(idx))).T
    fp = 1+0.01*rng.standard_normal((n, 32)); fq = 1+0.01*rng.standard_normal((n, 32))
    fp[:, idx] = fq[:, idx] = 1+load_sd*np.sqrt(s2)[:, None]*zl
    pv = np.zeros((n, 32)); pv[:, base.SITES-2] = base.CAP*np.clip(0.5+pv_sd*np.sqrt(s1)[:, None]*zp, 0, 1)
    return np.c_[(pv-lp*fp)/base.BASE, -(lq*fq)/base.BASE]


def voltages(target):
    v = np.array([base.polar_ac(t, base.Y, tolerance=1e-11)[0] for t in target]); return white(v)


def white(v):
    states = np.c_[v[:, 1:].real, v[:, 1:].imag]; center = states.mean(0)
    _, sv, vt = np.linalg.svd(states-center, full_matrices=False); scale = sv/np.sqrt(len(v)-1)
    return (states-center)@vt.T/scale, vt.T, scale


def aqgmm_input():
    d = base.REPO/'data/experiments/ieee33_20261004'; vs = []
    for f in ['train_samples', 'validation_samples', 'test_samples']:
        a = np.load(d/f'{f}.npz'); vs.append(a['voltage'][a['success'].astype(bool)])
    v = np.concatenate(vs); s = v*np.conj(v@base.Y.T)
    target = np.c_[s[:, 1:].real, s[:, 1:].imag]
    fac = np.load(d/'train_samples.npz')['factors'][:200]; vtr = np.load(d/'train_samples.npz')['voltage'][:200]
    st = vtr*np.conj(vtr@base.Y.T); ratio = -st[:, 1:].real*base.BASE/np.where(base.BUS[None, 1:, 2] > 0, base.BUS[None, 1:, 2], np.nan)
    check = float(np.nanmax(abs(ratio-fac)))
    return target, v, check


def main():
    rng = np.random.default_rng(2026102271)
    cases = {'A_weather_driven_loads': dict(pv_sd=.02, load_sd=.08, pv_strength=0, load_strength=.5, same_strength=False,
                                            expect='P,Q of buses 7,8,24,25,29,30,31,32 in one group'),
             'B_pv_and_loads_same_weather': dict(pv_sd=.10, load_sd=.08, pv_strength=.5, load_strength=.5, same_strength=True,
                                                 expect='8 PV P and the 8 loads P,Q in one group'),
             'C_pv_and_loads_two_weathers': dict(pv_sd=.10, load_sd=.08, pv_strength=.5, load_strength=.5, same_strength=False,
                                                 expect='8 PV P one group; 8 loads P,Q another group'),
             'D_large_ordinary_gaussian': dict(pv_sd=.10, load_sd=.08, pv_strength=0, load_strength=0, same_strength=False,
                                               expect='no group (GMM)')}
    rows = []
    for name, p in cases.items():
        expect = p.pop('expect'); every = inputs(N_GROUP, rng, **p); x, basis, scale = voltages(every[:N_VOLT])
        row = chk.split(every[:N_VOLT], x, basis, scale, rng, name, group_target=every); row['expected'] = expect; rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    target, v, check = aqgmm_input(); x, basis, scale = white(v[:8000])
    row = chk.split(target[:8000], x, basis, scale, rng, 'E_AQGMM_frozen_input', group_target=target)
    row.update(expected='unknown; injections recovered from stored voltages', load_factor_recovery_max_error=check, scenarios_for_groups=len(target))
    rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    (chk.OUT/'split_rule_scenarios.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
