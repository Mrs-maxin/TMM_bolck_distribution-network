"""IEEE 33-bus PV case at a fixed forecast interval with event-varying broken-cloud strength.

Inputs follow section 11 of 01_IDEA/pv_fixed_forecast_common_scale_plan_20261021.md.
AC, joint-voltage fitting, propagation and error statistics execute the archived
cloud routines unchanged, so results stay comparable with the earlier PV batches.

Phases (run in order; each checks the files locked by the previous one):
  ac_only    inputs for all splits, training/validation AC, operating-point check,
             strength diagnostics and target objects, locked before any model error
  fit        joint 64-D voltage GMM/TMM candidates and validation-likelihood selection
  reference  reference AC, propagation, all-object errors and the section 8.1 judgment
  block      block TMM of plan section 12 on the same inputs, targets and reference
  multiblock two-block TMM for two independent cloud areas (plan section 15 C)
"""
import os
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
from pathlib import Path
import sys, time, json, ast, traceback, argparse, zipfile, re
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import skew, kurtosis, binom
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO/'tests/python/archived'
FEATURE_SOURCE = SOURCE/'ac_only.py'
PLAN = REPO/'01_IDEA/pv_fixed_forecast_common_scale_plan_20261021.md'
sys.path.insert(0, str(SOURCE))
sys.dont_write_bytecode = True
# Importing the archived module creates this folder; keep the archive byte-identical.
IMPORT_SIDE_DIR = SOURCE.parent/'result'
_side_dir_existed = IMPORT_SIDE_DIR.exists()
from cloud_numeric_core import (sha, write_json, environment, admittance, polar_ac,
    evaluate_outputs, site_regions, initialize, fit, log_density, propagate, QUANTILES, floor_matrix, FLOOR)
if not _side_dir_existed:
    # Parallel fitting processes may remove the same empty folders first.
    for folder in sorted(IMPORT_SIDE_DIR.rglob('*'), reverse=True)+[IMPORT_SIDE_DIR]:
        try:
            folder.rmdir()
        except (FileNotFoundError, OSError):
            pass

CONFIG = dict(
    version='pv-fixed-forecast-literature-v1',
    plan_section='11 and 8.1',
    sites=[18, 17, 16, 15, 14, 13, 33, 22],
    site_xy_km={'13': [0.0, 0.0], '14': [0.4, 0.0], '15': [0.8, 0.0], '16': [1.2, 0.0],
                '17': [1.6, 0.0], '18': [2.0, 0.0], '33': [1.0, 2.0], '22': [-1.5, 1.5]},
    site_capacity_MW=0.58046875,
    forecast_fraction=0.50,
    clear_sky_fraction=(.85-.025*(12.375-12.5)**2)*(1-.0037*(28-25)),
    error_rms_fraction=0.10,
    strength_log_sd={'common_strength': 0.383, 'gaussian_control': 0.0, 'strong_weather': 0.489,
                     'strong_weather_nominal_load': 0.489, 'two_cloud_regions': 0.383},
    cloud_regions={'two_cloud_regions': [[18, 17, 16, 15, 14, 13], [33, 22]]},
    scene_load_multiplier={'strong_weather_nominal_load': 1.0},
    scene_stop_resample_fraction={'strong_weather': 0.01, 'strong_weather_nominal_load': 0.01},
    wvm_A_mps=4.0,
    timescale_s=900.0,
    load_multiplier=0.8,
    load_relative_sd=0.01,
    load_P_weights=[3, 8, 15],
    load_Q_weights=[4, 10, 20],
    load_PQ_correlation=[.35, .35, .25],
    splits={'training': 8192, 'validation': 4096, 'reference': 32768},
    seeds={'training': 2026102111, 'validation': 2026102112, 'reference': 2026102113},
    load_stream_offset=700001,
    stop_resample_fraction=0.005,
    stop_above_clear_sky_fraction=0.02,
    K_order=[1, 3, 6, 5, 12],
    propagation_points=65536,
    propagation_seed_base=2026102114,
    target_tail_factor=1.15,
    target_voltage_lower_pu=0.96,
    target_voltage_upper_pu=1.04,
    target_reverse_increase=0.20,
    targets_per_output_side=2,
    sameK_set=[1, 3, 5, 6],
    sameK_required=2,
    ks_body_factor=1.2,
    full_cdf_level=0.01,
    tail_cdf_level=0.005,
    comparable_factor=1.2,
    listed_factor=1.5,
)
SCENES = ['common_strength', 'gaussian_control', 'strong_weather', 'strong_weather_nominal_load', 'two_cloud_regions']
SMOKE = dict(splits={'training': 600, 'validation': 300, 'reference': 600},
             K_order=[1, 3], propagation_points=4096, sameK_set=[1, 3], sameK_required=1,
             stop_resample_fraction=1.0, stop_above_clear_sky_fraction=1.0)
START = time.perf_counter()
O = None
FIXED_TARGETS = None


def csv(path, rows):
    path = Path(path); tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False); os.replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path); tmp = path.with_name(path.stem+f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None
    os.replace(tmp, path)


def event(stage, **kwargs):
    row = dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), PID=os.getpid(),
               stage=stage, elapsed_seconds=time.perf_counter()-START, **kwargs)
    write_json(O/'progress.json', row)
    with (O/'progress_history.jsonl').open('a') as f:
        f.write(json.dumps(row, default=str)+'\n')
    print(json.dumps(row, default=str), flush=True)


def archived_functions(path, names, namespace, replace=None):
    source = Path(path).read_text(); tree = ast.parse(source)
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            code = ast.get_source_segment(source, node)
            for old, new in (replace or {}).get(node.name, []):
                assert old in code, (node.name, old)
                code = code.replace(old, new)
            exec(compile(code, str(path), 'exec'), namespace); found.add(node.name)
    assert found == set(names), (path, set(names)-found)


ROOT = REPO
archived_functions(SOURCE/'run_typhoon_comparison_K136.py', {'network', 'voltage'}, globals())
BASE, BUS, BRANCH = network()
Y, _, _ = admittance(BASE, BUS, BRANCH)
SITES = np.array(CONFIG['sites'])
CAP = CONFIG['site_capacity_MW']
_, REGIONS, _ = site_regions(BUS, BRANCH)
REGIONS = REGIONS[1:]
OBJECTS = [dict(output='V', node=i, branch=0, unit='p.u.', priority=False) for i in range(2, 34)]
OBJECTS += [dict(output=q, node=0, branch=i, unit='MW' if q == 'P' else 'MVAr', priority=False)
            for q in ['P', 'Q'] for i in range(1, 33)]
OBJECTS += [dict(output='loss', node=0, branch=0, unit='MW', priority=True)]
LOSS = len(OBJECTS)-1
PRIMARY = np.array([LOSS])


def outputs(v):
    return np.column_stack((abs(v[:, 1:]), evaluate_outputs(v, BASE, BUS, BRANCH)))


archived_functions(SOURCE/'run_cloud_study.py', {'save_fit', 'weighted_sorted', 'distribution_metrics'}, globals())
archived_functions(SOURCE/'run_cloud_passage_case33.py', {'ac'}, globals())
archived_functions(FEATURE_SOURCE, {'feature'}, globals())
Q = np.array([.005, .01, .025, .05, .25, .5, .75, .95, .975, .99, .995])
QN = ['q005', 'q01', 'q025', 'q05', 'q25', 'q50', 'q75', 'q95', 'q975', 'q99', 'q995']


def apply_mode(smoke):
    global SPLITS, KS, N
    if smoke:
        CONFIG.update({k: v for k, v in SMOKE.items()})
    SPLITS, KS, N = CONFIG['splits'], CONFIG['K_order'], CONFIG['propagation_points']
    archived_functions(SOURCE/'adapted_routines.py', {'fits', 'evaluate', 'summarize'}, globals(),
                       replace={'evaluate': [('2026101928', str(CONFIG['propagation_seed_base']))]})


def load_multiplier(scene):
    return CONFIG['scene_load_multiplier'].get(scene, CONFIG['load_multiplier'])


def site_region_index(scene):
    groups = CONFIG['cloud_regions'].get(scene)
    if groups is None:
        return np.zeros(len(SITES), int)
    return np.array([next(g for g, members in enumerate(groups) if b in members) for b in SITES])


def site_correlation(scene=None):
    xy = np.array([CONFIG['site_xy_km'][str(b)] for b in SITES])
    distance_m = 1000*np.linalg.norm(xy[:, None]-xy[None], axis=2)
    corr = np.exp(-distance_m/(CONFIG['wvm_A_mps']*CONFIG['timescale_s']))
    region = site_region_index(scene)
    return np.where(region[:, None] == region[None], corr, 0.)


def save_protocol(scene):
    path = O/'protocol.json'
    record = dict(CONFIG, scene=scene, plan_SHA256=sha(PLAN), script_SHA256=sha(Path(__file__)),
                  archived_source_SHA256={p.name: sha(p) for p in sorted(SOURCE.glob('*.py'))},
                  feature_source_SHA256=sha(FEATURE_SOURCE), site_correlation=site_correlation(scene),
                  identity='synthetic engineering inputs constrained by literature statistics; zero measured events')
    if path.exists():
        old = json.loads(path.read_text())
        assert old['scene'] == scene and old['version'] == CONFIG['version'], 'protocol changed'
        return
    write_json(path, record)
    try:
        info = environment()
    except ModuleNotFoundError as missing:
        import platform, scipy
        info = dict(platform=platform.platform(), python=sys.version, numpy=np.__version__,
                    scipy=scipy.__version__, blas_report_unavailable=str(missing))
    write_json(O/'environment.json', info)


def generate(scene, split):
    d = O/scene; d.mkdir(parents=True, exist_ok=True); path = d/f'{split}_inputs.npz'
    if path.exists():
        return
    n = SPLITS[split]; seed = CONFIG['seeds'][split]
    rng = np.random.Generator(np.random.MT19937(seed))
    sigma = CONFIG['strength_log_sd'][scene]
    lower = np.linalg.cholesky(site_correlation(scene)); region = site_region_index(scene); groups = region.max()+1
    forecast, rms = CONFIG['forecast_fraction'], CONFIG['error_rms_fraction']
    kept_s, kept_p, rejected_s = [], [], []
    while sum(len(a) for a in kept_s) < n:
        m = n-sum(len(a) for a in kept_s)+64
        strength = np.exp(rng.normal(-sigma**2, sigma, (m, groups))) if sigma > 0 else np.ones((m, groups))
        z = rng.standard_normal((m, len(SITES)))
        p = forecast+rms*strength[:, region]*(z@lower.T)
        valid = np.all((p >= 0) & (p <= 1), axis=1)
        kept_s.append(strength[valid]); kept_p.append(p[valid]); rejected_s.append(strength[~valid])
    strength = np.concatenate(kept_s)[:n]; pv_pu = np.concatenate(kept_p)[:n]
    if groups == 1:
        strength = strength[:, 0]
    rejected = np.concatenate(rejected_s)
    load_rng = np.random.Generator(np.random.MT19937(seed+CONFIG['load_stream_offset']))
    rho_g, rho_a, rho_n = CONFIG['load_PQ_correlation']
    wp, wq = np.array(CONFIG['load_P_weights'], float), np.array(CONFIG['load_Q_weights'], float)
    wp *= CONFIG['load_relative_sd']/np.linalg.norm(wp); wq *= CONFIG['load_relative_sd']/np.linalg.norm(wq)
    def paired(width, rho):
        a = load_rng.standard_normal((n, width))
        return a, rho*a+np.sqrt(1-rho*rho)*load_rng.standard_normal((n, width))
    gp, gq = paired(1, rho_g); ap, aq = paired(4, rho_a); zp, zq = paired(32, rho_n)
    fp = 1+wp[0]*gp+wp[1]*ap[:, REGIONS]+wp[2]*zp
    fq = 1+wq[0]*gq+wq[1]*aq[:, REGIONS]+wq[2]*zq
    load_p = load_multiplier(scene)*BUS[None, 1:, 2]*fp
    load_q = load_multiplier(scene)*BUS[None, 1:, 3]*fq
    pv = CAP*pv_pu; pbus = np.zeros((n, 32)); pbus[:, SITES-2] = pv
    target = np.c_[(pbus-load_p)/BASE, -load_q/BASE]
    assert np.isfinite(target).all() and pv.min() >= 0 and pv.max() <= CAP
    save_npz(path, target=target, PV_P_MW=pv, PV_pu=pv_pu, strength=strength,
             load_P_MW=load_p, load_Q_MVAr=load_q, event_id=np.arange(n), process_id=np.arange(n),
             seed=np.array(seed))
    deviation = pv_pu-forecast
    resample_fraction = len(rejected)/(n+len(rejected))
    above_clear = float(np.mean(np.any(pv_pu > CONFIG['clear_sky_fraction'], axis=1)))
    manifest = dict(scene=scene, split=split, events=n, seed=seed, measured_events=0,
        resampled_events=len(rejected), resample_fraction=resample_fraction,
        resampled_strength_quantiles=np.quantile(rejected, [.5, .9, .99], axis=0).tolist() if len(rejected) else [],
        above_clear_sky_event_fraction=above_clear,
        station_deviation_rms=np.sqrt(np.mean(deviation**2, axis=0)).tolist(),
        station_deviation_excess_kurtosis=kurtosis(deviation, axis=0).tolist(),
        station_deviation_skewness=skew(deviation, axis=0).tolist(),
        PV_total_MW=dict(mean=float(pv.sum(1).mean()), std=float(pv.sum(1).std()),
                         q005=float(np.quantile(pv.sum(1), .005)), q995=float(np.quantile(pv.sum(1), .995))),
        load_P_total_mean_MW=float(load_p.sum(1).mean()), load_Q_total_mean_MVAr=float(load_q.sum(1).mean()),
        SHA256=sha(path))
    write_json(d/f'{split}_input_manifest.json', manifest)
    event('inputs_saved', scene=scene, split=split, resample_fraction=resample_fraction,
          above_clear_sky=above_clear)
    stop_resample = CONFIG['scene_stop_resample_fraction'].get(scene, CONFIG['stop_resample_fraction'])
    if resample_fraction > stop_resample or above_clear > CONFIG['stop_above_clear_sky_fraction']:
        write_json(d/'STOP_input_bounds.json', manifest)
        raise RuntimeError('Input bound stop condition met; parameters not applicable')


def operating_point(scene):
    d = O/scene
    load_p = load_multiplier(scene)*BUS[1:, 2]; load_q = load_multiplier(scene)*BUS[1:, 3]
    pbus = np.zeros(32); pbus[SITES-2] = CAP*CONFIG['forecast_fraction']
    v, iterations = polar_ac(np.r_[(pbus-load_p)/BASE, -load_q/BASE], Y, tolerance=1e-11)
    out = outputs(v[None])[0]
    rows = [dict(**o, value=float(out[j])) for j, o in enumerate(OBJECTS)]
    csv(d/'forecast_operating_point.csv', rows)
    event('forecast_operating_point', scene=scene, iterations=iterations,
          min_voltage_pu=float(out[:32].min()), loss_MW=float(out[LOSS]))


def subtree_has_pv():
    children = {}
    for f, t in BRANCH[:, :2].astype(int):
        children.setdefault(f, []).append(t)
    def has_pv(bus):
        return bus in SITES or any(has_pv(c) for c in children.get(bus, []))
    return np.array([has_pv(int(t)) for t in BRANCH[:, 1]])


def line_impact(feat, side):
    median, tail = feat['q50'], feat['q005'] if side == 'lower' else feat['q995']
    if np.sign(tail) != np.sign(median) and median != 0:
        return 'reversal_import_to_reverse' if median > 0 else 'reversal_reverse_to_import'
    if median < 0 and side == 'lower' and abs(tail) >= (1+CONFIG['target_reverse_increase'])*abs(median):
        return 'reverse_flow_increase'
    return ''


def select_targets(scene, x):
    rule = (f"AC-only targets before any model error (plan 8.1/11). Training+validation AC only. "
            f"Tail: (q50-q005)/std or (q995-q50)/std > 2.575829*{CONFIG['target_tail_factor']}; "
            "histogram_peak_hints<=1; zero and largest atom mass <1%. Voltage impact: "
            f"q005<={CONFIG['target_voltage_lower_pu']} or q995>={CONFIG['target_voltage_upper_pu']} p.u. "
            "Line: active power only, branch with PV downstream, tail reverses flow direction or "
            f"raises reverse-flow magnitude by >={CONFIG['target_reverse_increase']:.0%}. "
            f"Up to {CONFIG['targets_per_output_side']} per output/side ranked by tail extension; "
            "near-proportional duplicates (|r|>=0.995, residual std<=5%) removed. Loss reported separately.")
    (O/scene/'target_rules.txt').write_text(rule, encoding='utf-8')
    on_path = subtree_has_pv(); rows = []
    for j, obj in enumerate(OBJECTS[:LOSS]):
        f = feature(x[:, j]); f.update(object_index=j, **obj)
        f['near_deterministic'] = f['std'] <= max(1e-8 if obj['output'] == 'V' else 1e-7, 1e-10*max(1, abs(f['mean'])))
        for side in ['lower', 'upper']:
            extension = f[side+'_std_extension']
            heavy = bool(np.isfinite(extension) and extension > 2.575829*CONFIG['target_tail_factor'])
            shape = bool(f['histogram_peak_hints'] <= 1 and f['zero_mass'] < .01 and f['max_atom_mass'] < .01)
            if obj['output'] == 'V':
                impact = ('undervoltage_side' if side == 'lower' and f['q005'] <= CONFIG['target_voltage_lower_pu'] else
                          'overvoltage_side' if side == 'upper' and f['q995'] >= CONFIG['target_voltage_upper_pu'] else '')
            elif obj['output'] == 'P' and on_path[obj['branch']-1]:
                impact = line_impact(f, side)
            else:
                impact = ''
            f[side+'_heavier_tail'] = heavy; f[side+'_single_peak_no_atom'] = shape; f[side+'_operating_impact'] = impact
            f[side+'_candidate'] = bool(heavy and shape and impact and not f['near_deterministic'])
        f['on_PV_collection_path'] = bool(obj['output'] in ['P', 'Q'] and on_path[obj['branch']-1])
        rows.append(f)
    table = pd.DataFrame(rows); csv(O/scene/'AC_features_train_validation.csv', rows)
    chosen = []
    for output in ['V', 'P']:
        for side in ['lower', 'upper']:
            pool = table[(table.output == output) & table[side+'_candidate']].sort_values(
                [side+'_std_extension', 'object_index'], ascending=[False, True])
            kept = []
            for _, r in pool.iterrows():
                j = int(r.object_index); duplicate = False
                for k in kept:
                    corr = np.corrcoef(x[:, j], x[:, k])[0, 1]
                    if abs(corr) >= .995 and np.sqrt(max(0, 1-corr*corr)) <= .05:
                        duplicate = True; break
                if duplicate:
                    continue
                kept.append(j)
                chosen.append(dict(object_index=j, output=output, node=int(r.node), branch=int(r.branch),
                                   unit=r.unit, side=side, tail_extension=float(r[side+'_std_extension']),
                                   operating_impact=r[side+'_operating_impact'], q005=r.q005, q50=r.q50, q995=r.q995))
                if len(kept) == CONFIG['targets_per_output_side']:
                    break
    csv(O/scene/'targets.csv', chosen or [dict(no_target=True)])
    return chosen


def strength_diagnostics(scene, x, strength, voltage_state):
    if strength.ndim == 2:
        # Two weather processes: group by the first (six-station) cloud area.
        strength = strength[:, 0]
    if np.ptp(strength) == 0:
        site_v = [int(np.flatnonzero([(o['output'] == 'V') and o['node'] == b for o in OBJECTS])[0]) for b in SITES]
        write_json(O/scene/'strength_diagnostics.json', dict(note='constant event strength; strength-group diagnostics not applicable',
                   site_voltage_excess_kurtosis=dict(zip(map(str, SITES), kurtosis(x[:, site_v], axis=0).tolist()))))
        return
    groups = np.digitize(strength, np.quantile(strength, [1/3, 2/3]))
    site_v = [int(np.flatnonzero([(o['output'] == 'V') and o['node'] == b for o in OBJECTS])[0]) for b in SITES]
    rows = []
    for g, name in enumerate(['weak', 'middle', 'strong']):
        m = groups == g
        rows.append(dict(group=name, events=int(m.sum()), strength_mean=float(strength[m].mean()),
                         **{f"V{OBJECTS[j]['node']}_mean": float(x[m, j].mean()) for j in site_v},
                         **{f"V{OBJECTS[j]['node']}_std": float(x[m, j].std()) for j in site_v},
                         loss_mean=float(x[m, LOSS].mean()), loss_std=float(x[m, LOSS].std())))
    csv(O/scene/'strength_group_summary.csv', rows)
    within = np.sqrt(np.mean([x[groups == g][:, site_v].var(0) for g in range(3)], axis=0))
    shift = np.max([abs(x[groups == g][:, site_v].mean(0)-x[:, site_v].mean(0))/within for g in range(3)], axis=0)
    ratio = x[groups == 2][:, site_v].std(0)/x[groups == 0][:, site_v].std(0)
    lead = []
    for g in [0, 2]:
        s = voltage_state[groups == g]; _, _, vt = np.linalg.svd(s-s.mean(0), full_matrices=False); lead.append(vt[0])
    sub = x[:, site_v]; mean, cov = sub.mean(0), np.cov(sub, rowvar=False)
    gauss = np.random.Generator(np.random.MT19937(2026102115)).multivariate_normal(mean, cov, 1000000)
    joint = {}
    for q in [.01, .05]:
        lo, hi = np.quantile(sub, q, axis=0), np.quantile(sub, 1-q, axis=0)
        joint[f'all_site_voltages_below_q{q}'] = dict(AC=float(np.mean(np.all(sub <= lo, axis=1))),
                                                      Gaussian_same_mean_cov=float(np.mean(np.all(gauss <= lo, axis=1))))
        joint[f'all_site_voltages_above_q{1-q}'] = dict(AC=float(np.mean(np.all(sub >= hi, axis=1))),
                                                        Gaussian_same_mean_cov=float(np.mean(np.all(gauss >= hi, axis=1))))
    summary = dict(site_voltage_nodes=SITES.tolist(),
                   max_group_mean_shift_over_within_std=dict(zip(map(str, SITES), shift.tolist())),
                   strong_over_weak_std_ratio=dict(zip(map(str, SITES), ratio.tolist())),
                   leading_voltage_direction_cosine_weak_strong=float(abs(lead[0]@lead[1])),
                   joint_large_deviation=joint,
                   site_voltage_excess_kurtosis=dict(zip(map(str, SITES), kurtosis(sub, axis=0).tolist())),
                   note='diagnostics only; no automatic gate besides target existence')
    write_json(O/scene/'strength_diagnostics.json', summary)


def load_split(scene, split):
    with np.load(O/scene/f'{split}_AC.npz') as a:
        v = a['voltage']
    with np.load(O/scene/f'{split}_inputs.npz') as a:
        strength = a['strength']
    return v, strength


def phase_ac_only(scene, smoke):
    save_protocol(scene)
    for split in SPLITS:
        generate(scene, split)
    for split in ['training', 'validation']:
        ac(scene, split)
    operating_point(scene)
    vt, st = load_split(scene, 'training'); vv, sv = load_split(scene, 'validation')
    v = np.r_[vt, vv]; x = outputs(v); strength = np.r_[st, sv]
    strength_diagnostics(scene, x, strength, np.c_[v[:, 1:].real, v[:, 1:].imag])
    targets = select_targets(scene, x)
    if FIXED_TARGETS is not None:
        (O/scene/'targets.csv').rename(O/scene/'targets_own_rule.csv')
        pd.read_csv(FIXED_TARGETS).to_csv(O/scene/'targets.csv', index=False)
        targets = pd.read_csv(O/scene/'targets.csv').to_dict('records')
        write_json(O/scene/'targets_source.json', dict(fixed_from=str(FIXED_TARGETS), SHA256=sha(FIXED_TARGETS),
                   own_rule_file='targets_own_rule.csv'))
    if not targets and not smoke:
        write_json(O/scene/'NO_TARGET.json', dict(stage='ac_only', reason='no object met plan 8.1; criteria not relaxed'))
        event('NO_TARGET', scene=scene); return
    if not targets:
        j = int(np.argmax(x[:, :32].std(0)))
        targets = [dict(object_index=j, output='V', node=OBJECTS[j]['node'], branch=0, unit='p.u.', side='lower',
                        operating_impact='smoke_only')]
        csv(O/scene/'targets.csv', targets)
    files = [O/'protocol.json', O/scene/'targets.csv', O/scene/'target_rules.txt',
             O/scene/'AC_features_train_validation.csv', O/scene/'strength_diagnostics.json']
    files += [O/scene/f'{s}_inputs.npz' for s in SPLITS] + [O/scene/f'{s}_AC.npz' for s in ['training', 'validation']]
    write_json(O/scene/'AC_ONLY_LOCK.json', dict(stage='targets_fixed_before_fits', smoke=smoke,
               UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               sha256={str(p.relative_to(O)): sha(p) for p in files}))
    event('AC_ONLY_COMPLETE', scene=scene, targets=[(t['output'], t['node'], t['branch'], t['side']) for t in targets])


def check_lock(scene):
    lock = json.loads((O/scene/'AC_ONLY_LOCK.json').read_text())
    for p, digest in lock['sha256'].items():
        assert sha(O/p) == digest, ('locked file changed', p)
    targets = pd.read_csv(O/scene/'targets.csv')
    global PRIMARY
    PRIMARY = np.unique(np.r_[targets.object_index.to_numpy(int), LOSS])
    for i, obj in enumerate(OBJECTS):
        obj['priority'] = bool(i in PRIMARY)
    return targets


def prepare_fitting(scene):
    d = O/scene
    if (d/'fitting_inputs.npz').exists():
        return
    states = {}
    for split in ['training', 'validation']:
        v, _ = load_split(scene, split); states[split] = np.c_[v[:, 1:].real, v[:, 1:].imag]
    center = states['training'].mean(0)
    _, sv, vt = np.linalg.svd(states['training']-center, full_matrices=False)
    rank = int(np.sum(sv > max(states['training'].shape)*np.spacing(sv[0]))); assert rank == 64, rank
    basis = vt.T; scale = sv/np.sqrt(len(states['training'])-1)
    x = (states['training']-center)@basis/scale; z = (states['validation']-center)@basis/scale
    save_npz(d/'fitting_inputs.npz', x=x, validation=z, center=center, basis=basis, scale=scale, singular_values=sv)
    write_json(d/'prepared.json', dict(dimension=64, rank=rank, no_truncation=True, singular_min=float(sv[-1]),
               fitting_inputs_SHA256=sha(d/'fitting_inputs.npz')))


def phase_fit(scene, only_k=None):
    check_lock(scene); prepare_fitting(scene)
    for k in ([only_k] if only_k else KS):
        fits(scene, k)
    event('FIT_COMPLETE', scene=scene, K=only_k or 'all')


def prepare_reference(scene):
    d = O/scene
    if (d/'reference_outputs.npz').exists():
        return
    with np.load(d/'reference_AC.npz') as a:
        ref = outputs(a['voltage'])
    save_npz(d/'reference_outputs.npz', values=ref)
    sd = ref.std(0); edges = np.array([np.linspace(r.min()-4*s, r.max()+4*s, 241) for r, s in zip(ref.T, sd)])
    save_npz(d/'reference_distribution_data.npz', edges=edges,
             counts=np.array([np.histogram(r, e)[0] for r, e in zip(ref.T, edges)]), mean=ref.mean(0), std=sd)


def judge(scene, targets, challenger='TMM', gs=None, qs=None, prefix='effectiveness'):
    if gs is None:
        gs = pd.read_csv(O/'global_repeat_summary.csv')
        qs = pd.read_csv(O/'quantile_repeat_summary.csv')
    gs, qs = gs.query('scene==@scene'), qs.query('scene==@scene')
    key = ['output', 'node', 'branch']
    def pick(table, obj, family, k):
        m = (table.family == family) & (table.K == k)
        for c in key:
            m &= table[c] == obj[c]
        return table[m]
    target_rows, target_ids = [], set()
    for _, t in targets.iterrows():
        target_ids.add(int(t.object_index)); side = t.side
        qset = [.005, .01] if side == 'lower' else [.99, .995]
        same_k = []
        for k in KS:
            g, h = pick(gs, t, 'GMM', k), pick(gs, t, challenger, k)
            if g.empty or h.empty:
                continue
            g, h = g.iloc[0], h.iloc[0]; metric = side+'_tail_cdf_mean_abs'
            beyond = [h[metric+'_max'] < g[metric+'_min']]
            for q in qset:
                gq, hq = pick(qs, t, 'GMM', k), pick(qs, t, challenger, k)
                gq, hq = gq[np.isclose(gq.q, q)].iloc[0], hq[np.isclose(hq.q, q)].iloc[0]
                beyond.append(hq['absolute_error_max'] < gq['absolute_error_min'])
            body = h['ks_mean'] <= CONFIG['ks_body_factor']*g['ks_mean']
            same_k.append(dict(K=k, tail_beyond_repeat_range=bool(all(beyond)), body_KS_within_factor=bool(body),
                               GMM_tail_cdf_mean=g[metric+'_mean'], **{challenger+'_tail_cdf_mean': h[metric+'_mean']},
                               GMM_KS=g['ks_mean'], **{challenger+'_KS': h['ks_mean']}))
        passed = [r['K'] for r in same_k if r['K'] in CONFIG['sameK_set'] and r['tail_beyond_repeat_range'] and r['body_KS_within_factor']]
        minimum = {}
        for family in ['GMM', challenger]:
            ok = [k for k in sorted(KS) if not pick(gs, t, family, k).empty and
                  pick(gs, t, family, k).iloc[0]['ks_mean'] <= CONFIG['full_cdf_level'] and
                  pick(gs, t, family, k).iloc[0][side+'_tail_cdf_sup_abs_mean'] <= CONFIG['tail_cdf_level']]
            minimum[family] = ok[0] if ok else None
        fewer = minimum[challenger] is not None and (minimum['GMM'] is None or minimum[challenger] < minimum['GMM'])
        target_rows.append(dict(object_index=int(t.object_index), output=t.output, node=int(t.node), branch=int(t.branch),
                                side=side, operating_impact=t.operating_impact, sameK_passing_K=json.dumps(passed),
                                sameK_pass=len(passed) >= CONFIG['sameK_required'], GMM_min_K_both_levels=minimum['GMM'],
                                **{challenger+'_min_K_both_levels': minimum[challenger]}, fewer_components_pass=bool(fewer),
                                target_pass=bool(len(passed) >= CONFIG['sameK_required'] or fewer),
                                per_K=json.dumps(same_k, default=float)))
    csv(O/scene/f'{prefix}_targets.csv', target_rows)
    others, listed = [], []
    for k in KS:
        for j, obj in enumerate(OBJECTS):
            if j in target_ids or j == LOSS:
                continue
            g, h = pick(gs, obj, 'GMM', k), pick(gs, obj, challenger, k)
            if g.empty or h.empty:
                continue
            g, h = g.iloc[0], h.iloc[0]; ok = True
            for m in ['ks', 'w1']:
                overlap = not (h[m+'_min'] > g[m+'_max'] or g[m+'_min'] > h[m+'_max'])
                ok &= bool(h[m+'_mean'] <= CONFIG['comparable_factor']*g[m+'_mean'] or overlap)
            others.append(dict(K=k, object_index=j, **{c: obj[c] for c in ['output', 'node', 'branch']}, comparable=ok,
                               **{'KS_ratio_'+challenger+'_over_GMM': h['ks_mean']/g['ks_mean'] if g['ks_mean'] > 0 else np.nan}))
            if g['ks_mean'] > 0 and h['ks_mean'] > CONFIG['listed_factor']*g['ks_mean']:
                listed.append(others[-1])
    csv(O/scene/f'{prefix}_other_objects.csv', others)
    csv(O/scene/f'{prefix}_listed_over_1p5.csv', listed or [dict(none=True)])
    frame = pd.DataFrame(others)
    summary = dict(scene=scene, challenger=challenger, targets=len(target_rows), targets_passed=int(sum(r['target_pass'] for r in target_rows)),
                   other_comparable_fraction_by_K=frame.groupby('K').comparable.mean().to_dict() if len(frame) else {},
                   other_objects_over_1p5_KS=len(listed),
                   criteria='plan section 8.1, fixed before model errors')
    write_json(O/scene/f'{prefix}_summary.json', summary)
    event('JUDGMENT_COMPLETE', **summary)
    return summary


def phase_reference(scene):
    targets = check_lock(scene)
    ac(scene, 'reference'); prepare_reference(scene)
    for k in KS:
        evaluate(scene, k)
    summarize(); judge(scene, targets)


# Block TMM (plan section 12): PV-driven voltage directions u share one random strength
# per component (multivariate t); the remaining directions w are Gaussian given u.
BLOCK_FAMILY = 'BTMM'
SUMMARY_KEYS = ['scene', 'K', 'family', 'output', 'node', 'branch', 'unit', 'priority', 'dimension', 'parameter_count']
SUMMARY_GLOBAL = ['ks', 'w1', 'model_mean', 'reference_mean', 'signed_mean_error', 'mean_absolute_error', 'model_variance',
                  'reference_variance', 'variance_signed_error', 'variance_absolute_error', 'lower_tail_cdf_sup_abs',
                  'lower_tail_cdf_mean_abs', 'lower_tail_cdf_signed_mean', 'upper_tail_cdf_sup_abs',
                  'upper_tail_cdf_mean_abs', 'upper_tail_cdf_signed_mean']
SUMMARY_QUANTILE = ['reference_quantile', 'model_quantile', 'signed_error', 'absolute_error', 'reference_cdf_error']


def pv_voltage_subspace(scene):
    d = O/scene
    if (d/'block_subspace.npz').exists():
        with np.load(d/'block_subspace.npz') as a:
            return a['Q1'], a['Q2']
    from cloud_numeric_core import rect_jacobian
    load_p = load_multiplier(scene)*BUS[1:, 2]; load_q = load_multiplier(scene)*BUS[1:, 3]
    pbus = np.zeros(32); pbus[SITES-2] = CAP*CONFIG['forecast_fraction']
    v, _ = polar_ac(np.r_[(pbus-load_p)/BASE, -load_q/BASE], Y, tolerance=1e-11)
    sensitivity = np.linalg.inv(rect_jacobian(v, Y))[:, SITES-2]
    fit_in = np.load(d/'fitting_inputs.npz')
    whitened = (fit_in['basis'].T@sensitivity)/fit_in['scale'][:, None]
    q1, s, _ = np.linalg.svd(whitened, full_matrices=False)
    rank = int(np.sum(s > s[0]*1e-10))
    Q1 = q1[:, :rank]; Q2 = np.linalg.svd(np.eye(64)-Q1@Q1.T)[0][:, :64-rank]
    assert np.allclose(Q1.T@Q2, 0, atol=1e-10) and np.allclose(np.c_[Q1, Q2].T@np.c_[Q1, Q2], np.eye(64), atol=1e-10)
    x = fit_in['x']; s_train = np.load(d/'training_inputs.npz')['strength']
    if s_train.ndim == 2:
        s_train = s_train[:, 0]
    U, W = x@Q1, x@Q2
    corr = lambda a: [float(np.corrcoef(c**2, s_train**2)[0, 1]) for c in a.T]
    diagnostics = dict(rank=rank, sensitivity_singular_values=s.tolist(),
        u_excess_kurtosis=kurtosis(U, axis=0).tolist(), w_excess_kurtosis_median=float(np.median(kurtosis(W, axis=0))),
        w_excess_kurtosis_max=float(kurtosis(W, axis=0).max()), u_square_strength_correlation=corr(U),
        w_square_strength_correlation_median=float(np.median(corr(W))),
        training_variance_share_u=float(np.sum(((x*fit_in['scale'])@Q1)**2)/np.sum((x*fit_in['scale'])**2)),
        rule='PV active-power sensitivity at the forecast operating point, fixed before any model error')
    save_npz(d/'block_subspace.npz', Q1=Q1, Q2=Q2, sensitivity=sensitivity)
    write_json(d/'block_subspace_diagnostics.json', diagnostics)
    event('block_subspace', scene=scene, rank=rank, u_kurtosis=np.round(diagnostics['u_excess_kurtosis'], 2).tolist())
    return Q1, Q2


def block_log_terms(U, W, model):
    from scipy.special import gammaln
    from scipy.linalg import solve_triangular
    n, r = U.shape; dw = W.shape[1]; k = len(model['weights'])
    logs, delta = np.empty((n, k)), np.empty((n, k))
    for j in range(k):
        L = cholesky(model['scatter'][j], lower=True)
        z = solve_triangular(L, (U-model['means'][j]).T, lower=True).T; dj = np.einsum('ij,ij->i', z, z)
        nu = model['df'][j]
        tlog = -.5*(r*np.log(2*np.pi)+dj) if np.isinf(nu) else \
            gammaln((nu+r)/2)-gammaln(nu/2)-r/2*np.log(nu*np.pi)-(nu+r)/2*np.log1p(dj/nu)
        LO = cholesky(model['residual'][j], lower=True)
        e = solve_triangular(LO, (W-model['intercept'][j]-U@model['coefficient'][j].T).T, lower=True).T
        glog = -.5*(dw*np.log(2*np.pi)+np.einsum('ij,ij->i', e, e))-np.log(LO.diagonal()).sum()
        logs[:, j] = np.log(model['weights'][j])+tlog-np.log(L.diagonal()).sum()+glog
        delta[:, j] = dj
    return logs, delta


def block_density(U, W, model):
    from scipy.special import logsumexp
    logs, delta = block_log_terms(U, W, model)
    density = logsumexp(logs, axis=1)
    return density, np.exp(logs-density[:, None]), delta


def block_from_gmm(gmm, Q1, Q2, nu):
    means, cov = gmm['means'], gmm['scatter']; k = len(gmm['weights'])
    model = dict(weights=np.array(gmm['weights'], float), means=means@Q1, scatter=[], df=np.full(k, float(nu)),
                 intercept=[], coefficient=[], residual=[])
    for j in range(k):
        suu, swu, sww = Q1.T@cov[j]@Q1, Q2.T@cov[j]@Q1, Q2.T@cov[j]@Q2
        c = np.linalg.solve(suu, swu.T).T
        model['scatter'].append(floor_matrix(suu*(nu-2)/nu, FLOOR*(nu-2)/nu)[0])
        model['coefficient'].append(c); model['intercept'].append(means[j]@Q2-c@(means[j]@Q1))
        model['residual'].append(floor_matrix(sww-c@suu@c.T)[0])
    return {key: np.array(v) for key, v in model.items()}


def fit_block(U, W, initial, maximum=300, tolerance=1e-6):
    from scipy.special import digamma
    from scipy.optimize import brentq
    start = time.perf_counter(); model = {key: np.array(v, copy=True) for key, v in initial.items()}
    n, r = U.shape; design = np.c_[np.ones(n), U]
    density, resp, delta = block_density(U, W, model); history = [density.mean()]; status = 'maximum_iterations'
    for iteration in range(1, maximum+1):
        count = resp.sum(0)
        if count.min() <= W.shape[1]+r+2:
            status = 'collapsed_component'; break
        new = {key: v.copy() for key, v in model.items()}; new['weights'] = count/n
        for j, nu in enumerate(model['df']):
            tau = (nu+r)/(nu+delta[:, j]); w = resp[:, j]*tau
            new['means'][j] = (w@U)/w.sum(); res = U-new['means'][j]
            new['scatter'][j] = floor_matrix((res.T@(res*w[:, None]))/count[j], FLOOR*(nu-2)/nu)[0]
            statistic = resp[:, j]@(digamma((nu+r)/2)-np.log((nu+delta[:, j])/2)-tau)/count[j]
            score = lambda v: np.log(v/2)-digamma(v/2)+1+statistic
            new['df'][j] = 4.1 if score(4.1) <= 0 else 1e5 if score(1e5) >= 0 else brentq(score, 4.1, 1e5, xtol=1e-8)
            b = np.linalg.solve(design.T@(design*resp[:, j, None]), design.T@(W*resp[:, j, None]))
            new['intercept'][j], new['coefficient'][j] = b[0], b[1:].T
            e = W-design@b
            new['residual'][j] = floor_matrix((e.T@(e*resp[:, j, None]))/count[j])[0]
        density, new_resp, new_delta = block_density(U, W, new); value = density.mean(); gain = value-history[-1]
        if gain < -1e-8:
            status = 'rejected_decreasing_update'; break
        model, resp, delta = new, new_resp, new_delta; history.append(value)
        if abs(gain) < tolerance*max(1., abs(history[-2])):
            status = 'converged'; break
    return model, dict(status=status, iterations=iteration, train_ll=history[-1], history=np.array(history),
                       df=model['df'], weights=model['weights'], effective_counts=resp.sum(0),
                       seconds=time.perf_counter()-start)


def block_parameter_count(k, r, dw):
    return (k-1)+k*(r+r*(r+1)//2+1+dw+dw*r+dw*(dw+1)//2)


def fit_block_candidates(scene, k, Q1, Q2):
    d = O/scene; c = d/'candidates'; fit_in = np.load(d/'fitting_inputs.npz')
    U, W = fit_in['x']@Q1, fit_in['x']@Q2; Uv, Wv = fit_in['validation']@Q1, fit_in['validation']@Q2
    rows = []
    for p in range(1 if k == 1 else 2):
        gmm = np.load(c/f'K{k}_p{p+1:02d}_GMM.npz')
        for nu in [12, 200]:
            stem = c/f'K{k}_p{p+1:02d}_gmm_warm_df{nu}_BLOCK'
            if not stem.with_suffix('.json').exists():
                event('fitting', scene=scene, K=k, family=BLOCK_FAMILY, partition=p+1, initial_df=nu)
                model, audit = fit_block(U, W, block_from_gmm(gmm, Q1, Q2, nu))
                model.update(Q1=Q1, Q2=Q2, block=np.array(True))
                save_npz(stem.with_suffix('.npz'), **model, history=audit['history'])
                audit.pop('history'); audit.update(scene=scene, K=k, family=BLOCK_FAMILY, partition=p+1, initial_df=nu,
                    validation_ll=float(block_density(Uv, Wv, model)[0].mean()), model_sha256=sha(stem.with_suffix('.npz')))
                write_json(stem.with_suffix('.json'), audit)
            row = json.loads(stem.with_suffix('.json').read_text()); row['candidate'] = stem.name; rows.append(row)
    conv = [r for r in rows if r['status'] == 'converged']; best = max(conv or rows, key=lambda r: r['validation_ll'])
    return dict(scene=scene, K=k, family=BLOCK_FAMILY, candidate=best['candidate'], validation_ll=best['validation_ll'],
                train_ll=best['train_ll'], status=best['status'], iterations=best['iterations'], dimension=64,
                parameter_count=block_parameter_count(k, Q1.shape[1], Q2.shape[1]), df=json.dumps(best['df']),
                weights=json.dumps(best['weights']), model_SHA256=best['model_sha256']), rows


def propagate_block(model, n, seed):
    from scipy.stats import qmc
    from scipy.special import ndtri, gammaincinv
    weights = model['weights']; k = len(weights); r = model['means'].shape[1]; dw = model['intercept'].shape[1]
    desired = n*weights; counts = np.maximum(1, np.floor(desired).astype(int))
    while counts.sum() < n:
        counts[np.argmax(desired-counts)] += 1
    while counts.sum() > n:
        ids = np.flatnonzero(counts > 1); counts[ids[np.argmin((desired-counts)[ids])]] -= 1
    samples, w = [], []
    for j, nj in enumerate(counts):
        nu = model['df'][j]
        sampler = qmc.Sobol(r+dw+1, scramble=True, seed=np.random.Generator(np.random.MT19937(seed+1009*j)))
        uni = np.clip(sampler.random_base2(int(np.ceil(np.log2(nj))))[:nj], np.finfo(float).eps, 1-np.finfo(float).eps)
        tau = np.ones(nj) if np.isinf(nu) else 2*gammaincinv(nu/2, uni[:, -1])/nu
        U = model['means'][j]+(ndtri(uni[:, :r])@cholesky(model['scatter'][j], lower=True).T)/np.sqrt(tau[:, None])
        W = model['intercept'][j]+U@model['coefficient'][j].T+ndtri(uni[:, r:r+dw])@cholesky(model['residual'][j], lower=True).T
        samples.append(U@model['Q1'].T+W@model['Q2'].T); w.append(np.full(nj, weights[j]/nj))
    return np.concatenate(samples), np.concatenate(w)


def propagate_any(model, n, seed):
    keys = getattr(model, 'files', model)
    if 'multiblock' in keys:
        return propagate_multiblock(model, n, seed)
    if 'block' in keys:
        return propagate_block(model, n, seed)
    return propagate(model, n, seed, 'sobol', randomizer='MT19937')


def repeat_summary(frame, metrics, keys):
    a = frame.groupby(keys, dropna=False)[metrics].agg(['mean', 'min', 'max'])
    a.columns = ['_'.join(c) for c in a.columns]
    return a.reset_index()


def phase_block(scene):
    targets = check_lock(scene); d = O/scene
    Q1, Q2 = pv_voltage_subspace(scene)
    selected, attempts = [], []
    for k in KS:
        row, rows = fit_block_candidates(scene, k, Q1, Q2); selected.append(row); attempts += rows
        event('selected', scene=scene, K=k, family=BLOCK_FAMILY, candidate=row['candidate'], df=row['df'])
    csv(d/'selected_models_block.csv', selected); csv(d/'all_fit_attempts_block.csv', attempts)
    archived_functions(SOURCE/'adapted_routines.py', {'evaluate'}, globals(), replace={'evaluate': [
        ('2026101928', str(CONFIG['propagation_seed_base'])), ('def evaluate(scene,k):', 'def evaluate_block(scene,k):'),
        ("'selected_models.csv'", "'selected_models_block.csv'"),
        ("propagate(m,N,seed,'sobol',randomizer='MT19937')", 'propagate_any(m,N,seed)')]})
    for k in KS:
        evaluate_block(scene, k)
    frames = [pd.read_csv(p) for p in sorted((d/'propagation').glob(f'{BLOCK_FAMILY}_*_global.csv'))]
    qframes = [pd.read_csv(p) for p in sorted((d/'propagation').glob(f'{BLOCK_FAMILY}_*_quantiles.csv'))]
    gb = repeat_summary(pd.concat(frames), SUMMARY_GLOBAL, SUMMARY_KEYS)
    qb = repeat_summary(pd.concat(qframes), SUMMARY_QUANTILE, SUMMARY_KEYS+['q'])
    gb.to_csv(O/'block_global_repeat_summary.csv', index=False); qb.to_csv(O/'block_quantile_repeat_summary.csv', index=False)
    gs = pd.concat([pd.read_csv(O/'global_repeat_summary.csv'), gb], ignore_index=True)
    qs = pd.concat([pd.read_csv(O/'quantile_repeat_summary.csv'), qb], ignore_index=True)
    judge(scene, targets, challenger=BLOCK_FAMILY, gs=gs, qs=qs, prefix='block_effectiveness')


# Multi-block TMM (derivation.md section G): each independent weather process drives its own
# group of voltage directions with its own fluctuation strength; remaining directions Gaussian.
MULTI_FAMILY = 'MBTMM'


def multiblock_subspace(scene):
    d = O/scene
    if (d/'multiblock_subspace.npz').exists():
        with np.load(d/'multiblock_subspace.npz') as a:
            return [a[f'Q{g}'] for g in range(int(a['groups']))], a['Qw']
    from cloud_numeric_core import rect_jacobian
    load_p = load_multiplier(scene)*BUS[1:, 2]; load_q = load_multiplier(scene)*BUS[1:, 3]
    pbus = np.zeros(32); pbus[SITES-2] = CAP*CONFIG['forecast_fraction']
    v, _ = polar_ac(np.r_[(pbus-load_p)/BASE, -load_q/BASE], Y, tolerance=1e-11)
    inverse = np.linalg.inv(rect_jacobian(v, Y)); fit_in = np.load(d/'fitting_inputs.npz')
    region = site_region_index(scene); taken = np.zeros((64, 0)); groups = []
    for g in range(region.max()+1):
        h = (fit_in['basis'].T@inverse[:, SITES[region == g]-2])/fit_in['scale'][:, None]
        h = h-taken@(taken.T@h)
        q, s, _ = np.linalg.svd(h, full_matrices=False); q = q[:, :int(np.sum(s > s[0]*1e-10))]
        groups.append(q); taken = np.c_[taken, q]
    Qw = np.linalg.svd(np.eye(64)-taken@taken.T)[0][:, :64-taken.shape[1]]
    full = np.c_[taken, Qw]; assert np.allclose(full.T@full, np.eye(64), atol=1e-10)
    x = fit_in['x']; s_train = np.load(d/'training_inputs.npz')['strength']
    diag = dict(groups=len(groups), ranks=[q.shape[1] for q in groups],
                group_excess_kurtosis=[kurtosis(x@q, axis=0).tolist() for q in groups],
                w_excess_kurtosis_median=float(np.median(kurtosis(x@Qw, axis=0))))
    if s_train.ndim == 2:
        diag['group_square_strength_correlation'] = [[[float(np.corrcoef(c**2, s_train[:, j]**2)[0, 1]) for j in range(s_train.shape[1])]
                                                       for c in (x@q).T] for q in groups]
    save_npz(d/'multiblock_subspace.npz', groups=np.array(len(groups)), Qw=Qw, **{f'Q{g}': q for g, q in enumerate(groups)})
    write_json(d/'multiblock_subspace_diagnostics.json', diag)
    event('multiblock_subspace', scene=scene, ranks=diag['ranks'])
    return groups, Qw


def multiblock_log_terms(U1, U2, W, model):
    from scipy.special import gammaln
    from scipy.linalg import solve_triangular
    n = len(U1); r1, r2, dw = U1.shape[1], U2.shape[1], W.shape[1]; k = len(model['weights'])
    logs, d1, d2 = np.empty((n, k)), np.empty((n, k)), np.empty((n, k))
    def tlog(dist, nu, r, L):
        return gammaln((nu+r)/2)-gammaln(nu/2)-r/2*np.log(nu*np.pi)-(nu+r)/2*np.log1p(dist/nu)-np.log(L.diagonal()).sum()
    for j in range(k):
        L1 = cholesky(model['scatter1'][j], lower=True)
        z1 = solve_triangular(L1, (U1-model['means1'][j]).T, lower=True).T; a1 = np.einsum('ij,ij->i', z1, z1)
        L2 = cholesky(model['scatter2'][j], lower=True)
        e2 = U2-model['means2'][j]-U1@model['coupling'][j].T
        z2 = solve_triangular(L2, e2.T, lower=True).T; a2 = np.einsum('ij,ij->i', z2, z2)
        LO = cholesky(model['residual'][j], lower=True)
        e = solve_triangular(LO, (W-model['intercept'][j]-np.c_[U1, U2]@model['coefficient'][j].T).T, lower=True).T
        glog = -.5*(dw*np.log(2*np.pi)+np.einsum('ij,ij->i', e, e))-np.log(LO.diagonal()).sum()
        logs[:, j] = np.log(model['weights'][j])+tlog(a1, model['df1'][j], r1, L1)+tlog(a2, model['df2'][j], r2, L2)+glog
        d1[:, j], d2[:, j] = a1, a2
    return logs, d1, d2


def multiblock_density(U1, U2, W, model):
    from scipy.special import logsumexp
    logs, d1, d2 = multiblock_log_terms(U1, U2, W, model)
    density = logsumexp(logs, axis=1)
    return density, np.exp(logs-density[:, None]), d1, d2


def multiblock_from_gmm(gmm, Q1, Q2, Qw, nu):
    k = len(gmm['weights']); out = {key: [] for key in ['means1', 'scatter1', 'means2', 'coupling', 'scatter2',
                                                       'intercept', 'coefficient', 'residual']}
    U = np.c_[Q1, Q2]; r1 = Q1.shape[1]
    for j in range(k):
        mu, cov = gmm['means'][j], gmm['scatter'][j]
        s11, s21, s22 = Q1.T@cov@Q1, Q2.T@cov@Q1, Q2.T@cov@Q2
        f = np.linalg.solve(s11, s21.T).T
        out['means1'].append(mu@Q1); out['scatter1'].append(floor_matrix(s11*(nu-2)/nu, FLOOR*(nu-2)/nu)[0])
        out['coupling'].append(f); out['means2'].append(mu@Q2-f@(mu@Q1))
        out['scatter2'].append(floor_matrix((s22-f@s11@f.T)*(nu-2)/nu, FLOOR*(nu-2)/nu)[0])
        suu, swu, sww = U.T@cov@U, Qw.T@cov@U, Qw.T@cov@Qw
        c = np.linalg.solve(suu, swu.T).T
        out['coefficient'].append(c); out['intercept'].append(mu@Qw-c@(mu@U)); out['residual'].append(floor_matrix(sww-c@suu@c.T)[0])
    model = {key: np.array(v) for key, v in out.items()}
    model.update(weights=np.array(gmm['weights'], float), df1=np.full(k, float(nu)), df2=np.full(k, float(nu)))
    return model


def fit_multiblock(U1, U2, W, initial, maximum=300, tolerance=1e-6):
    from scipy.special import digamma
    from scipy.optimize import brentq
    start = time.perf_counter(); model = {key: np.array(v, copy=True) for key, v in initial.items()}
    n = len(U1); r1, r2 = U1.shape[1], U2.shape[1]; X1 = np.c_[np.ones(n), U1]; X = np.c_[np.ones(n), U1, U2]
    density, resp, d1, d2 = multiblock_density(U1, U2, W, model); history = [density.mean()]; status = 'maximum_iterations'
    def df_root(nu, r, dist, g, count):
        tau = (nu+r)/(nu+dist); stat = g@(digamma((nu+r)/2)-np.log((nu+dist)/2)-tau)/count
        score = lambda v: np.log(v/2)-digamma(v/2)+1+stat
        return 4.1 if score(4.1) <= 0 else 1e5 if score(1e5) >= 0 else brentq(score, 4.1, 1e5, xtol=1e-8)
    for iteration in range(1, maximum+1):
        count = resp.sum(0)
        if count.min() <= W.shape[1]+r1+r2+2:
            status = 'collapsed_component'; break
        new = {key: v.copy() for key, v in model.items()}; new['weights'] = count/n
        for j in range(len(count)):
            g = resp[:, j]
            nu1 = model['df1'][j]; t1 = (nu1+r1)/(nu1+d1[:, j]); w1 = g*t1
            new['means1'][j] = (w1@U1)/w1.sum(); res = U1-new['means1'][j]
            new['scatter1'][j] = floor_matrix((res.T@(res*w1[:, None]))/count[j], FLOOR*(nu1-2)/nu1)[0]
            new['df1'][j] = df_root(nu1, r1, d1[:, j], g, count[j])
            nu2 = model['df2'][j]; t2 = (nu2+r2)/(nu2+d2[:, j]); w2 = g*t2
            b = np.linalg.solve(X1.T@(X1*w2[:, None]), X1.T@(U2*w2[:, None]))
            new['means2'][j], new['coupling'][j] = b[0], b[1:].T; e2 = U2-X1@b
            new['scatter2'][j] = floor_matrix((e2.T@(e2*w2[:, None]))/count[j], FLOOR*(nu2-2)/nu2)[0]
            new['df2'][j] = df_root(nu2, r2, d2[:, j], g, count[j])
            b = np.linalg.solve(X.T@(X*g[:, None]), X.T@(W*g[:, None]))
            new['intercept'][j], new['coefficient'][j] = b[0], b[1:].T; e = W-X@b
            new['residual'][j] = floor_matrix((e.T@(e*g[:, None]))/count[j])[0]
        density, new_resp, nd1, nd2 = multiblock_density(U1, U2, W, new); value = density.mean(); gain = value-history[-1]
        if gain < -1e-8:
            status = 'rejected_decreasing_update'; break
        model, resp, d1, d2 = new, new_resp, nd1, nd2; history.append(value)
        if abs(gain) < tolerance*max(1., abs(history[-2])):
            status = 'converged'; break
    return model, dict(status=status, iterations=iteration, train_ll=history[-1], history=np.array(history),
                       df1=model['df1'], df2=model['df2'], weights=model['weights'], seconds=time.perf_counter()-start)


def propagate_multiblock(model, n, seed):
    from scipy.stats import qmc
    from scipy.special import ndtri, gammaincinv
    weights = model['weights']; r1 = model['means1'].shape[1]; r2 = model['means2'].shape[1]; dw = model['intercept'].shape[1]
    desired = n*weights; counts = np.maximum(1, np.floor(desired).astype(int))
    while counts.sum() < n:
        counts[np.argmax(desired-counts)] += 1
    while counts.sum() > n:
        ids = np.flatnonzero(counts > 1); counts[ids[np.argmin((desired-counts)[ids])]] -= 1
    samples, w = [], []
    for j, nj in enumerate(counts):
        sampler = qmc.Sobol(r1+r2+dw+2, scramble=True, seed=np.random.Generator(np.random.MT19937(seed+1009*j)))
        uni = np.clip(sampler.random_base2(int(np.ceil(np.log2(nj))))[:nj], np.finfo(float).eps, 1-np.finfo(float).eps)
        t1 = 2*gammaincinv(model['df1'][j]/2, uni[:, -2])/model['df1'][j]; t2 = 2*gammaincinv(model['df2'][j]/2, uni[:, -1])/model['df2'][j]
        U1 = model['means1'][j]+(ndtri(uni[:, :r1])@cholesky(model['scatter1'][j], lower=True).T)/np.sqrt(t1[:, None])
        U2 = model['means2'][j]+U1@model['coupling'][j].T+(ndtri(uni[:, r1:r1+r2])@cholesky(model['scatter2'][j], lower=True).T)/np.sqrt(t2[:, None])
        Wv = model['intercept'][j]+np.c_[U1, U2]@model['coefficient'][j].T+ndtri(uni[:, r1+r2:r1+r2+dw])@cholesky(model['residual'][j], lower=True).T
        samples.append(U1@model['Q1'].T+U2@model['Q2'].T+Wv@model['Qw'].T); w.append(np.full(nj, weights[j]/nj))
    return np.concatenate(samples), np.concatenate(w)


def phase_multiblock(scene):
    targets = check_lock(scene); d = O/scene; c = d/'candidates'
    groups, Qw = multiblock_subspace(scene); assert len(groups) == 2, 'two weather processes are required'
    Q1, Q2 = groups; fit_in = np.load(d/'fitting_inputs.npz')
    split = lambda z: (z@Q1, z@Q2, z@Qw)
    (U1, U2, W), (V1, V2, Vw) = split(fit_in['x']), split(fit_in['validation'])
    selected, attempts = [], []
    for k in KS:
        rows = []
        for p in range(1 if k == 1 else 2):
            gmm = np.load(c/f'K{k}_p{p+1:02d}_GMM.npz')
            for nu in [12, 200]:
                stem = c/f'K{k}_p{p+1:02d}_gmm_warm_df{nu}_MULTIBLOCK'
                if not stem.with_suffix('.json').exists():
                    event('fitting', scene=scene, K=k, family=MULTI_FAMILY, partition=p+1, initial_df=nu)
                    model, audit = fit_multiblock(U1, U2, W, multiblock_from_gmm(gmm, Q1, Q2, Qw, nu))
                    model.update(Q1=Q1, Q2=Q2, Qw=Qw, multiblock=np.array(True))
                    save_npz(stem.with_suffix('.npz'), **model, history=audit['history']); audit.pop('history')
                    audit.update(scene=scene, K=k, family=MULTI_FAMILY, partition=p+1, initial_df=nu,
                                 validation_ll=float(multiblock_density(V1, V2, Vw, model)[0].mean()), model_sha256=sha(stem.with_suffix('.npz')))
                    write_json(stem.with_suffix('.json'), audit)
                row = json.loads(stem.with_suffix('.json').read_text()); row['candidate'] = stem.name; rows.append(row)
        conv = [r for r in rows if r['status'] == 'converged']; best = max(conv or rows, key=lambda r: r['validation_ll'])
        r1, r2, dw = Q1.shape[1], Q2.shape[1], Qw.shape[1]
        params = (k-1)+k*(r1+r1*(r1+1)//2+1+r2+r2*r1+r2*(r2+1)//2+1+dw+dw*(r1+r2)+dw*(dw+1)//2)
        selected.append(dict(scene=scene, K=k, family=MULTI_FAMILY, candidate=best['candidate'], validation_ll=best['validation_ll'],
                             train_ll=best['train_ll'], status=best['status'], iterations=best['iterations'], dimension=64,
                             parameter_count=params, df=json.dumps([best['df1'], best['df2']]), weights=json.dumps(best['weights']),
                             model_SHA256=best['model_sha256']))
        attempts += rows; event('selected', scene=scene, K=k, family=MULTI_FAMILY, df1=best['df1'], df2=best['df2'])
    csv(d/'selected_models_multiblock.csv', selected); csv(d/'all_fit_attempts_multiblock.csv', attempts)
    archived_functions(SOURCE/'adapted_routines.py', {'evaluate'}, globals(), replace={'evaluate': [
        ('2026101928', str(CONFIG['propagation_seed_base'])), ('def evaluate(scene,k):', 'def evaluate_multiblock(scene,k):'),
        ("'selected_models.csv'", "'selected_models_multiblock.csv'"),
        ("propagate(m,N,seed,'sobol',randomizer='MT19937')", 'propagate_any(m,N,seed)')]})
    for k in KS:
        evaluate_multiblock(scene, k)
    frames = [pd.read_csv(p) for p in sorted((d/'propagation').glob(f'{MULTI_FAMILY}_*_global.csv'))]
    qframes = [pd.read_csv(p) for p in sorted((d/'propagation').glob(f'{MULTI_FAMILY}_*_quantiles.csv'))]
    gb = repeat_summary(pd.concat(frames), SUMMARY_GLOBAL, SUMMARY_KEYS); qb = repeat_summary(pd.concat(qframes), SUMMARY_QUANTILE, SUMMARY_KEYS+['q'])
    gb.to_csv(O/'multiblock_global_repeat_summary.csv', index=False); qb.to_csv(O/'multiblock_quantile_repeat_summary.csv', index=False)
    gs = pd.concat([pd.read_csv(O/'global_repeat_summary.csv'), gb], ignore_index=True)
    qs = pd.concat([pd.read_csv(O/'quantile_repeat_summary.csv'), qb], ignore_index=True)
    judge(scene, targets, challenger=MULTI_FAMILY, gs=gs, qs=qs, prefix='multiblock_effectiveness')


def main():
    global O
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['ac_only', 'fit', 'reference', 'block', 'multiblock'])
    parser.add_argument('--scene', choices=SCENES, default='common_strength')
    parser.add_argument('--out', default=str(REPO/'result/case33bw/pv_fixed_forecast_case33_20261021'))
    parser.add_argument('--smoke', action='store_true', help='tiny counts to test the code path only')
    parser.add_argument('--K', type=int, help='fit one K; finish with a plain fit run to collect all candidates')
    parser.add_argument('--fixed-targets', help='targets.csv locked by the main run; the own-rule selection is kept aside')
    parser.add_argument('--training-seed', type=int); parser.add_argument('--validation-seed', type=int)
    args = parser.parse_args()
    global FIXED_TARGETS
    O = Path(args.out); O.mkdir(parents=True, exist_ok=True)
    FIXED_TARGETS = Path(args.fixed_targets) if args.fixed_targets else None
    if args.training_seed:
        CONFIG['seeds']['training'] = args.training_seed
    if args.validation_seed:
        CONFIG['seeds']['validation'] = args.validation_seed
    apply_mode(args.smoke)
    if args.scene == 'gaussian_control' and FIXED_TARGETS is None:
        raise RuntimeError('Gaussian control is judged on the targets fixed by the main scene')
    try:
        {'ac_only': lambda: phase_ac_only(args.scene, args.smoke),
         'fit': lambda: phase_fit(args.scene, args.K),
         'reference': lambda: phase_reference(args.scene),
         'block': lambda: phase_block(args.scene),
         'multiblock': lambda: phase_multiblock(args.scene)}[args.phase]()
    except Exception:
        write_json(O/f'failure_{args.phase}_{os.getpid()}.json',
                   dict(error=traceback.format_exc(), UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        raise


if __name__ == '__main__':
    main()
