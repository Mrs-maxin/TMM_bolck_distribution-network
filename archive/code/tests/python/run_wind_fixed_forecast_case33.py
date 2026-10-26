"""IEEE 33-bus wind at a fixed 15-min forecast: event-strength versus operating-state uncertainty.

Protocol: 01_IDEA/plans/wind_fixed_forecast_minimal_protocol_20261021.md. The PV fixed-forecast runner is
imported for AC, target locking, GMM/TMM/block-TMM fitting, propagation and the section 8.1
judgment; this file replaces only the wind inputs, the forecast operating point and the
wind-driven voltage directions, and adds the pre-registered limit probabilities.

Phases (run in order):
  ac_only    inputs, training/validation AC, forecast point, diagnostics, targets locked
  fit        joint 64-D voltage GMM/TMM candidates and validation-likelihood selection
  reference  reference AC, propagation, all-object errors and judgment for standard TMM
  block      block TMM on the same inputs, targets and reference
  shared     block TMM and Gaussian with one residual covariance shared by all components
  limits     pre-registered limit probabilities from the first propagation repeat
  compare    likelihood curves over K, validation-selected K and accuracy at each model's selected K
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
PLAN = REPO/'01_IDEA/plans/wind_fixed_forecast_minimal_protocol_20261021.md'
SITES = np.array([18, 25, 33])
CAP = 2.972/3
TAN_PHI = np.tan(np.arccos(0.98))
SCENES = ['common_strength', 'gaussian_control', 'state_mixture']
STATE_MIXTURE = dict(probabilities=[0.05, 0.90, 0.05], states=[-1, 0, 1], shift=0.241979,
                     within_sd=0.064379, within_correlation=0.51745)
LIMIT_EVENTS = {
    'any node V < 0.950': lambda y: np.any(y[:, :32] < .950, axis=1),
    'any node V < 0.955': lambda y: np.any(y[:, :32] < .955, axis=1),
    'P17 < -0.60 MW': lambda y: y[:, 32+16] < -.60,
    'P32 < -0.65 MW': lambda y: y[:, 32+31] < -.65,
    'P24 > +0.10 MW': lambda y: y[:, 32+23] > .10,
}

for key in ['site_xy_km', 'clear_sky_fraction', 'wvm_A_mps', 'timescale_s', 'stop_above_clear_sky_fraction',
            'scene_load_multiplier', 'scene_stop_resample_fraction']:
    base.CONFIG.pop(key, None)
base.CONFIG.update(
    version='wind-fixed-forecast-minimal-v1', plan_section='4 and 5', sites=SITES.tolist(),
    site_capacity_MW=CAP, power_factor_absorbing=0.98, forecast_fraction=0.50, error_rms_fraction=0.10,
    site_correlation=0.8, strength_log_sd={'common_strength': 0.383, 'gaussian_control': 0.0, 'state_mixture': 0.0},
    state_mixture=STATE_MIXTURE, load_multiplier=0.8, scene_load_multiplier={}, scene_stop_resample_fraction={},
    splits={'training': 8192, 'validation': 4096, 'reference': 32768},
    seeds={'training': 2026102131, 'validation': 2026102132, 'reference': 2026102133},
    stop_resample_fraction=0.005, K_order=[1, 3, 6], sameK_set=[1, 3, 6], sameK_required=2,
    propagation_seed_base=2026102134, limit_events=list(LIMIT_EVENTS))
base.SMOKE.pop('stop_above_clear_sky_fraction', None)
base.SITES, base.CAP, base.SCENES, base.PLAN = SITES, CAP, SCENES, PLAN


def correlation(rho):
    return rho*np.ones((len(SITES), len(SITES)))+(1-rho)*np.eye(len(SITES))


def save_protocol(scene):
    path = base.O/'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert old['scene'] == scene and old['version'] == base.CONFIG['version'], 'protocol changed'
        assert old['runner_SHA256'] == base.sha(Path(base.__file__)), 'imported PV runner changed during the run'
        return
    record = dict(base.CONFIG, scene=scene, plan_SHA256=base.sha(PLAN), script_SHA256=base.sha(Path(__file__)),
                  runner=str(Path(base.__file__).relative_to(REPO)), runner_SHA256=base.sha(Path(base.__file__)),
                  archived_source_SHA256={p.name: base.sha(p) for p in sorted(base.SOURCE.glob('*.py'))},
                  site_correlation=correlation(base.CONFIG['site_correlation']),
                  identity='synthetic wind forecast-error inputs; zero measured events')
    base.write_json(path, record)
    try:
        info = base.environment()
    except ModuleNotFoundError as missing:
        import platform, scipy
        info = dict(platform=platform.platform(), python=sys.version, numpy=np.__version__,
                    scipy=scipy.__version__, blas_report_unavailable=str(missing))
    base.write_json(base.O/'environment.json', info)


def wind_deviation(scene, rng, m):
    rms = base.CONFIG['error_rms_fraction']
    if scene == 'state_mixture':
        p = STATE_MIXTURE
        state = np.array(p['states'])[np.searchsorted(np.cumsum(p['probabilities']), rng.random(m), side='right')]
        z = rng.standard_normal((m, len(SITES)))@np.linalg.cholesky(correlation(p['within_correlation'])).T
        return p['shift']*state[:, None]+p['within_sd']*z, np.ones(m), state
    sigma = base.CONFIG['strength_log_sd'][scene]
    strength = np.exp(rng.normal(-sigma**2, sigma, m)) if sigma > 0 else np.ones(m)
    z = rng.standard_normal((m, len(SITES)))@np.linalg.cholesky(correlation(base.CONFIG['site_correlation'])).T
    return rms*strength[:, None]*z, strength, np.zeros(m, int)


def generate(scene, split):
    d = base.O/scene; d.mkdir(parents=True, exist_ok=True); path = d/f'{split}_inputs.npz'
    if path.exists():
        return
    n = base.SPLITS[split]; seed = base.CONFIG['seeds'][split]
    rng = np.random.Generator(np.random.MT19937(seed)); forecast = base.CONFIG['forecast_fraction']
    kept, rejected = [], 0
    while sum(len(a[0]) for a in kept) < n:
        m = n-sum(len(a[0]) for a in kept)+64
        dev, strength, state = wind_deviation(scene, rng, m); u = forecast+dev
        valid = np.all((u >= 0) & (u <= 1), axis=1); rejected += int((~valid).sum())
        kept.append((u[valid], strength[valid], state[valid]))
    wind_pu, strength, state = (np.concatenate([a[i] for a in kept])[:n] for i in range(3))
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
    load_p = base.CONFIG['load_multiplier']*base.BUS[None, 1:, 2]*fp
    load_q = base.CONFIG['load_multiplier']*base.BUS[None, 1:, 3]*fq
    wind_p = CAP*wind_pu; wind_q = -TAN_PHI*wind_p
    pbus, qbus = np.zeros((n, 32)), np.zeros((n, 32)); pbus[:, SITES-2] = wind_p; qbus[:, SITES-2] = wind_q
    target = np.c_[(pbus-load_p)/base.BASE, (qbus-load_q)/base.BASE]
    assert np.isfinite(target).all() and wind_p.min() >= 0 and wind_p.max() <= CAP
    base.save_npz(path, target=target, wind_P_MW=wind_p, wind_Q_MVAr=wind_q, wind_pu=wind_pu, strength=strength,
                  operating_state=state, load_P_MW=load_p, load_Q_MVAr=load_q, event_id=np.arange(n),
                  process_id=np.arange(n), seed=np.array(seed))
    deviation = wind_pu-forecast; resample_fraction = rejected/(n+rejected)
    manifest = dict(scene=scene, split=split, events=n, seed=seed, measured_events=0, resampled_events=rejected,
        resample_fraction=resample_fraction, station_deviation_mean=deviation.mean(0).tolist(),
        station_deviation_rms=np.sqrt(np.mean(deviation**2, axis=0)).tolist(),
        station_deviation_correlation=np.corrcoef(deviation, rowvar=False).tolist(),
        station_deviation_excess_kurtosis=kurtosis(deviation, axis=0).tolist(),
        station_deviation_skewness=skew(deviation, axis=0).tolist(),
        operating_state_fractions={str(s): float(np.mean(state == s)) for s in np.unique(state)},
        wind_total_MW=dict(mean=float(wind_p.sum(1).mean()), std=float(wind_p.sum(1).std()),
                           q005=float(np.quantile(wind_p.sum(1), .005)), q995=float(np.quantile(wind_p.sum(1), .995))),
        load_P_total_mean_MW=float(load_p.sum(1).mean()), load_Q_total_mean_MVAr=float(load_q.sum(1).mean()),
        SHA256=base.sha(path))
    base.write_json(d/f'{split}_input_manifest.json', manifest)
    base.event('inputs_saved', scene=scene, split=split, resample_fraction=resample_fraction)
    if resample_fraction > base.CONFIG['stop_resample_fraction']:
        base.write_json(d/'STOP_input_bounds.json', manifest)
        raise RuntimeError('Input bound stop condition met; parameters not applicable')


def forecast_injection():
    p, q = np.zeros(32), np.zeros(32)
    p[SITES-2] = CAP*base.CONFIG['forecast_fraction']; q[SITES-2] = -TAN_PHI*p[SITES-2]
    lm = base.CONFIG['load_multiplier']
    return np.r_[(p-lm*base.BUS[1:, 2])/base.BASE, (q-lm*base.BUS[1:, 3])/base.BASE]


def operating_point(scene):
    v, iterations = base.polar_ac(forecast_injection(), base.Y, tolerance=1e-11)
    out = base.outputs(v[None])[0]
    base.csv(base.O/scene/'forecast_operating_point.csv', [dict(**o, value=float(out[j])) for j, o in enumerate(base.OBJECTS)])
    base.event('forecast_operating_point', scene=scene, iterations=iterations,
               min_voltage_pu=float(out[:32].min()), loss_MW=float(out[base.LOSS]))


def wind_voltage_subspace(scene):
    d = base.O/scene
    if (d/'block_subspace.npz').exists():
        with np.load(d/'block_subspace.npz') as a:
            return a['Q1'], a['Q2']
    from cloud_numeric_core import rect_jacobian
    v, _ = base.polar_ac(forecast_injection(), base.Y, tolerance=1e-11)
    inverse = np.linalg.inv(rect_jacobian(v, base.Y))
    sensitivity = inverse[:, SITES-2]-TAN_PHI*inverse[:, 32+SITES-2]
    fit_in = np.load(d/'fitting_inputs.npz')
    whitened = (fit_in['basis'].T@sensitivity)/fit_in['scale'][:, None]
    q1, s, _ = np.linalg.svd(whitened, full_matrices=False)
    rank = int(np.sum(s > s[0]*1e-10))
    Q1 = q1[:, :rank]; Q2 = np.linalg.svd(np.eye(64)-Q1@Q1.T)[0][:, :64-rank]
    assert np.allclose(np.c_[Q1, Q2].T@np.c_[Q1, Q2], np.eye(64), atol=1e-10)
    x = fit_in['x']; U, W = x@Q1, x@Q2
    with np.load(d/'training_inputs.npz') as a:
        strength, state = a['strength'], a['operating_state']
    def corr(a, b):
        return [float(np.corrcoef(c**2, b)[0, 1]) if np.ptp(b) > 0 else None for c in a.T]
    diagnostics = dict(rank=rank, sensitivity_singular_values=s.tolist(),
        u_excess_kurtosis=kurtosis(U, axis=0).tolist(), w_excess_kurtosis_median=float(np.median(kurtosis(W, axis=0))),
        w_excess_kurtosis_max=float(kurtosis(W, axis=0).max()),
        u_square_strength_correlation=corr(U, strength**2), u_square_state_correlation=corr(U, state**2),
        training_variance_share_u=float(np.sum(((x*fit_in['scale'])@Q1)**2)/np.sum((x*fit_in['scale'])**2)),
        rule='wind P with power-factor Q sensitivity at the forecast operating point, fixed before any model error')
    base.save_npz(d/'block_subspace.npz', Q1=Q1, Q2=Q2, sensitivity=sensitivity)
    base.write_json(d/'block_subspace_diagnostics.json', diagnostics)
    base.event('block_subspace', scene=scene, rank=rank, u_kurtosis=np.round(diagnostics['u_excess_kurtosis'], 2).tolist(),
               variance_share_u=diagnostics['training_variance_share_u'])
    return Q1, Q2


base.save_protocol, base.generate, base.operating_point = save_protocol, generate, operating_point
base.pv_voltage_subspace = wind_voltage_subspace
shared.install(base, wind_voltage_subspace, LIMIT_EVENTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['ac_only', 'fit', 'reference', 'block', 'shared', 'limits', 'compare'])
    parser.add_argument('--scene', choices=SCENES, default='common_strength')
    parser.add_argument('--out', default=str(REPO/'result/case33bw/wind_fixed_forecast_case33_20261021'))
    parser.add_argument('--smoke', action='store_true', help='tiny counts to test the code path only')
    parser.add_argument('--K', type=int, help='fit one K; finish with a plain fit run to collect all candidates')
    parser.add_argument('--fixed-targets', help='targets.csv locked by the common-strength run')
    parser.add_argument('--training-events', type=int); parser.add_argument('--training-seed', type=int)
    parser.add_argument('--K-order', help='comma-separated K list, e.g. 1,3,6,12')
    args = parser.parse_args()
    if args.training_events:
        base.CONFIG['splits']['training'] = args.training_events
    if args.training_seed:
        base.CONFIG['seeds']['training'] = args.training_seed
    if args.K_order:
        base.CONFIG['K_order'] = [int(k) for k in args.K_order.split(',')]
    base.O = Path(args.out); base.O.mkdir(parents=True, exist_ok=True)
    base.FIXED_TARGETS = Path(args.fixed_targets) if args.fixed_targets else None
    base.apply_mode(args.smoke)
    if args.phase == 'ac_only' and args.scene != 'common_strength' and base.FIXED_TARGETS is None:
        raise RuntimeError('control scenes are judged on the targets locked by the common-strength run')
    try:
        {'ac_only': lambda: base.phase_ac_only(args.scene, args.smoke),
         'fit': lambda: base.phase_fit(args.scene, args.K),
         'reference': lambda: base.phase_reference(args.scene),
         'block': lambda: base.phase_block(args.scene),
         'shared': lambda: shared.phase_shared(args.scene),
         'limits': lambda: shared.phase_limits(args.scene),
         'compare': lambda: shared.phase_compare(args.scene, 'protocol section 7, fixed before the shared-residual fits')}[args.phase]()
    except Exception:
        base.write_json(base.O/f'failure_{args.phase}_{os.getpid()}.json',
                        dict(error=traceback.format_exc(), UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        raise


if __name__ == '__main__':
    main()
