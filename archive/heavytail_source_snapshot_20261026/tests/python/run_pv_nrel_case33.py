"""Case 1, IEEE 33-bus: PV forecast errors constructed from NREL forecast and actual records.

Plan: 01_IDEA/plans/case_study_plan_20261022.md sections 3-6. The PV fixed-forecast runner and the
shared-residual module are imported unchanged; only the PV input generation is replaced by the
NREL-fitted law (common lognormal weather strength times a correlated deviation, outputs held at
zero or at the observed clear-sky maximum). Load generation repeats the runner's rule and streams.
Phases: ac_only, limits_lock, fit, reference, block, shared, limits, compare.
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
from pathlib import Path
import sys, json, time, argparse, traceback
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import run_pv_fixed_forecast_case33 as base
import shared_residual_mixture as shared

REPO = base.REPO
PLAN = REPO/'01_IDEA/plans/case_study_plan_20261022.md'
INPUT_FIT = REPO/'result/case33bw/nrel_pv_input_case33_20261022/fit_HA4_0.4_0.6.json'
INPUT_FIT_SCRIPT = REPO/'tests/python/fit_nrel_pv_forecast_error_case33.py'
SCENE = 'nrel_ha4_40_60'
base.SCENES.append(SCENE)
base.CONFIG['seeds'] = {'training': 2026102211, 'validation': 2026102212, 'reference': 2026102213}
base.CONFIG['propagation_seed_base'] = 2026102214
base.CONFIG['strength_log_sd'][SCENE] = None
LIMIT_FRACTIONS = [0.01, 0.001]
LIMIT_EVENTS = {}
shared.install(base, base.pv_voltage_subspace, LIMIT_EVENTS)


def nrel_law():
    fit = json.loads(INPUT_FIT.read_text())
    return dict(mean=np.array(fit['mean_error']), covariance=np.array(fit['covariance']),
                log_sd=float(fit['lognormal_log_sd']), upper=np.array(fit['observed_max_output']),
                sites=fit['sites'], records=fit['records'])


def generate(scene, split):
    d = base.O/scene; d.mkdir(parents=True, exist_ok=True); path = d/f'{split}_inputs.npz'
    if path.exists():
        return
    law = nrel_law(); n = base.SPLITS[split]; seed = base.CONFIG['seeds'][split]
    rng = np.random.Generator(np.random.MT19937(seed))
    sigma = law['log_sd']; lower = np.linalg.cholesky(law['covariance'])
    strength = np.exp(rng.normal(-sigma**2/2, sigma, n))
    unbounded = base.CONFIG['forecast_fraction']+law['mean']+np.sqrt(strength)[:, None]*(rng.standard_normal((n, 8))@lower.T)
    pv_pu = np.clip(unbounded, 0, law['upper'])
    held_low, held_high = unbounded < 0, unbounded > law['upper']
    load_rng = np.random.Generator(np.random.MT19937(seed+base.CONFIG['load_stream_offset']))
    rho_g, rho_a, rho_n = base.CONFIG['load_PQ_correlation']
    wp, wq = np.array(base.CONFIG['load_P_weights'], float), np.array(base.CONFIG['load_Q_weights'], float)
    wp *= base.CONFIG['load_relative_sd']/np.linalg.norm(wp); wq *= base.CONFIG['load_relative_sd']/np.linalg.norm(wq)
    def paired(width, rho):
        a = load_rng.standard_normal((n, width))
        return a, rho*a+np.sqrt(1-rho*rho)*load_rng.standard_normal((n, width))
    gp, gq = paired(1, rho_g); ap, aq = paired(4, rho_a); zp, zq = paired(32, rho_n)
    fp = 1+wp[0]*gp+wp[1]*ap[:, base.REGIONS]+wp[2]*zp
    fq = 1+wq[0]*gq+wq[1]*aq[:, base.REGIONS]+wq[2]*zq
    load_p = base.load_multiplier(scene)*base.BUS[None, 1:, 2]*fp
    load_q = base.load_multiplier(scene)*base.BUS[None, 1:, 3]*fq
    pv = base.CAP*pv_pu; pbus = np.zeros((n, 32)); pbus[:, base.SITES-2] = pv
    target = np.c_[(pbus-load_p)/base.BASE, -load_q/base.BASE]
    assert np.isfinite(target).all() and pv.min() >= 0 and pv.max() <= base.CAP
    base.save_npz(path, target=target, PV_P_MW=pv, PV_pu=pv_pu, strength=strength,
                  load_P_MW=load_p, load_Q_MVAr=load_q, event_id=np.arange(n), process_id=np.arange(n), seed=np.array(seed))
    deviation = pv_pu-base.CONFIG['forecast_fraction']
    manifest = dict(scene=scene, split=split, events=n, seed=seed, input_law=str(INPUT_FIT.relative_to(REPO)),
        input_law_SHA256=base.sha(INPUT_FIT), NREL_records=law['records'], site_to_bus=dict(zip(map(str, base.SITES),
        [(s['lat'], s['lon'], s['cap']) for s in law['sites']])),
        held_at_zero_fraction_per_site=held_low.mean(0).tolist(), held_at_upper_fraction_per_site=held_high.mean(0).tolist(),
        any_site_held_fraction=float(np.mean(np.any(held_low | held_high, axis=1))),
        station_deviation_rms=np.sqrt(np.mean(deviation**2, axis=0)).tolist(),
        station_deviation_excess_kurtosis=kurtosis(deviation, axis=0).tolist(),
        station_deviation_skewness=skew(deviation, axis=0).tolist(),
        PV_total_MW=dict(mean=float(pv.sum(1).mean()), std=float(pv.sum(1).std()),
                         q005=float(np.quantile(pv.sum(1), .005)), q995=float(np.quantile(pv.sum(1), .995))),
        load_P_total_mean_MW=float(load_p.sum(1).mean()), load_Q_total_mean_MVAr=float(load_q.sum(1).mean()),
        SHA256=base.sha(path))
    base.write_json(d/f'{split}_input_manifest.json', manifest)
    base.event('inputs_saved', scene=scene, split=split, any_site_held=manifest['any_site_held_fraction'])


base.generate = generate


def save_case_protocol():
    path = base.O/'case_protocol.json'
    record = dict(plan=str(PLAN.relative_to(REPO)), plan_SHA256=base.sha(PLAN), script_SHA256=base.sha(Path(__file__)),
                  runner_SHA256=base.sha(Path(base.__file__)), shared_module_SHA256=base.sha(Path(shared.__file__)),
                  input_law=str(INPUT_FIT.relative_to(REPO)), input_law_SHA256=base.sha(INPUT_FIT),
                  input_fit_script_SHA256=base.sha(INPUT_FIT_SCRIPT), seeds=base.CONFIG['seeds'],
                  propagation_seed_base=base.CONFIG['propagation_seed_base'], K_order=base.CONFIG['K_order'],
                  limit_rule=f'per target object and side, AC thresholds at training+validation exceedance {LIMIT_FRACTIONS}, '
                             'rounded outward to 0.0025 p.u. or 0.0025 MW; locked after ac_only and before any fit',
                  limit_rule_revision_v2='outward rounding is replaced by inward rounding when it leaves less than half the '
                             'intended AC exceedance (branch reverse flow bounded by the clear-sky PV output); revised '
                             'before any model error was read')
    if path.exists():
        old = json.loads(path.read_text())
        assert old['runner_SHA256'] == record['runner_SHA256'] and old['input_law_SHA256'] == record['input_law_SHA256'], \
            'imported runner or input law changed during the run'
        return
    base.write_json(path, record)


def phase_limits_lock():
    d = base.O/SCENE; path = d/'limit_events_locked.json'
    lock = json.loads((d/'AC_ONLY_LOCK.json').read_text())
    if path.exists():
        return
    x = np.r_[base.outputs(np.load(d/'training_AC.npz')['voltage']), base.outputs(np.load(d/'validation_AC.npz')['voltage'])]
    events = []
    for _, t in pd.read_csv(d/'targets.csv').iterrows():
        j = int(t.object_index); name = f"V{int(t.node)}" if t.output == 'V' else f"P{int(t.branch)}"
        for frac in LIMIT_FRACTIONS:
            op = '<' if t.side == 'lower' else '>'
            q = np.quantile(x[:, j], frac if op == '<' else 1-frac)
            share = lambda v: float(np.mean(x[:, j] < v) if op == '<' else np.mean(x[:, j] > v))
            outward = (np.floor if op == '<' else np.ceil)(q/0.0025)*0.0025
            inward = (np.ceil if op == '<' else np.floor)(q/0.0025)*0.0025
            value, rounding = (outward, 'outward') if share(outward) >= frac/2 else (inward, 'inward')
            events.append(dict(name=f'{name} {op} {value:.4f}', object_index=j, op=op, value=float(value), rounding=rounding,
                               AC_train_validation_fraction=share(value)))
    base.write_json(path, dict(events=events, rule=json.loads((base.O/'case_protocol.json').read_text())['limit_rule'],
                               AC_ONLY_LOCK_SHA256=base.sha(d/'AC_ONLY_LOCK.json'), UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    base.event('LIMITS_LOCKED', events=[e['name'] for e in events])


def load_limit_events():
    path = base.O/SCENE/'limit_events_locked.json'
    if not path.exists():
        return
    for e in json.loads(path.read_text())['events']:
        j, v = e['object_index'], e['value']
        LIMIT_EVENTS[e['name']] = (lambda y, j=j, v=v: y[:, j] < v) if e['op'] == '<' else (lambda y, j=j, v=v: y[:, j] > v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['ac_only', 'limits_lock', 'fit', 'reference', 'block', 'shared', 'limits', 'compare'])
    parser.add_argument('--out', default=str(REPO/'result/case33bw/pv_nrel_case33_20261022'))
    parser.add_argument('--smoke', action='store_true', help='tiny counts to test the code path only')
    parser.add_argument('--K', type=int, help='fit one K; finish with a plain fit run to collect all candidates')
    parser.add_argument('--K-order', default='1,2,3,4,6,8,12')
    args = parser.parse_args()
    base.CONFIG['K_order'] = [int(k) for k in args.K_order.split(',')]
    base.O = Path(args.out); base.O.mkdir(parents=True, exist_ok=True)
    base.apply_mode(args.smoke)
    try:
        save_case_protocol(); load_limit_events()
        {'ac_only': lambda: base.phase_ac_only(SCENE, args.smoke),
         'limits_lock': phase_limits_lock,
         'fit': lambda: base.phase_fit(SCENE, args.K),
         'reference': lambda: base.phase_reference(SCENE),
         'block': lambda: base.phase_block(SCENE),
         'shared': lambda: shared.phase_shared(SCENE),
         'limits': lambda: shared.phase_limits(SCENE),
         'compare': lambda: shared.phase_compare(SCENE, 'case_study_plan_20261022 section 6, fixed before the fits')}[args.phase]()
    except Exception:
        base.write_json(base.O/f'failure_{args.phase}_{os.getpid()}.json',
                        dict(error=traceback.format_exc(), UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        raise


if __name__ == '__main__':
    main()
