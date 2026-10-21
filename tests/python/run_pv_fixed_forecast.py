"""PV case at a fixed forecast interval with event-varying broken-cloud strength, any radial case.

General version of run_pv_fixed_forecast_case33.py (--case case33bw|case141). Inputs follow
sections 11 and 16 of 01_IDEA/pv_fixed_forecast_common_scale_plan_20261021.md. AC, joint-voltage
fitting, propagation and error statistics execute the archived cloud routines; only the
33-bus dimensions inside the archived AC routine are substituted by the case dimensions.

Phases (run in order; each checks the files locked by the previous one):
  ac_only       inputs for all scenario groups, AC of the parameter-estimation and component-choice
                scenarios, operating-point check, strength diagnostics and target objects,
                locked before any model error
  fit           joint d=2(n-1) voltage GMM/TMM fits and choice by component-choice likelihood
  block_fit     block TMM fits of plan section 12 (warm start from the GMM fits)
  reference_ac  AC of the exact-comparison scenarios (after the target lock, never read by fits)
  reference     voltage-to-quantity distributions of GMM/TMM, all-object errors, section 8.1 judgment
  block         block TMM distributions and section 8.1 judgment against GMM
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

CASES = dict(
    case33bw=dict(
        version='pv-fixed-forecast-literature-v1',
        plan_section='11 and 8.1',
        case_file='data/cases/case33bw_modify.m',
        sites=[18, 17, 16, 15, 14, 13, 33, 22],
        site_xy_km={'13': [0.0, 0.0], '14': [0.4, 0.0], '15': [0.8, 0.0], '16': [1.2, 0.0],
                    '17': [1.6, 0.0], '18': [2.0, 0.0], '33': [1.0, 2.0], '22': [-1.5, 1.5]},
        site_capacity_MW=0.58046875,
        sameK_set=[1, 3, 5, 6],
        default_out='result/case33bw/pv_fixed_forecast_case33_20261021'),
    case141=dict(
        version='pv-fixed-forecast-literature-case141-v1',
        plan_section='16, 16.1, 11, 12, 14 and 8.1',
        case_file='data/cases/case141_modify.m',
        sites='site_regions',
        site_distance='feeder electrical path length, line length = |z| (ohm) / 0.4 ohm/km',
        line_ohm_per_km=0.4,
        site_capacity_rule='1.25 x total base active load / 8',
        power_factor=0.85,
        K_order=[1, 3, 6],
        sameK_set=[1, 3, 6],
        default_out='result/case141/pv_fixed_forecast_case141_20261021_{scene}'),
)
CONFIG = dict(
    forecast_fraction=0.50,
    clear_sky_fraction=(.85-.025*(12.375-12.5)**2)*(1-.0037*(28-25)),
    error_rms_fraction=0.10,
    strength_log_sd={'common_strength': 0.383, 'gaussian_control': 0.0, 'strong_weather': 0.489,
                     'strong_weather_nominal_load': 0.489},
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
CASE_DEFAULTS = {k: CONFIG[k] for k in ['K_order', 'sameK_set']}
SCENES = ['common_strength', 'gaussian_control', 'strong_weather', 'strong_weather_nominal_load']
SMOKE = dict(splits={'training': 600, 'validation': 300, 'reference': 600},
             K_order=[1, 3], propagation_points=4096, sameK_set=[1, 3], sameK_required=1,
             stop_resample_fraction=1.0, stop_above_clear_sky_fraction=1.0,
             scene_stop_resample_fraction={})
# d = 280 needs more than d+2 scenarios per component for the program check to reach K = 3.
SMOKE_SPLITS = dict(case141={'training': 1500, 'validation': 600, 'reference': 1000})
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
CASE = None


def read_matpower_tables(path):
    text = re.sub(r'%[^\n]*', '', Path(path).read_text()); tables = {}
    for name in ['bus', 'branch']:
        block = re.search(r'mpc\.'+name+r'\s*=\s*\[(.*?)\];', text, re.S)[1]
        tables[name] = np.array([[float(v) for v in row.split()] for row in block.split(';') if row.strip()])
    base = float(re.search(r'mpc.baseMVA\s*=\s*([\d.]+)', text)[1])
    return base, tables['bus'], tables['branch']


def network_case141():
    # Same conversion as the closing block of case141_modify.m: ohm -> p.u., kVA -> MVA,
    # then MVA at 0.85 power factor -> MW and MVAr.
    base, bus, branch = read_matpower_tables(REPO/CASES['case141']['case_file'])
    z_base = (bus[0, 9]*1e3)**2/(base*1e6)
    branch[:, 2:4] /= z_base
    bus[:, 2:4] /= 1e3
    pf = CASES['case141']['power_factor']
    bus[:, 3] = bus[:, 2]*np.sin(np.arccos(pf))
    bus[:, 2] = bus[:, 2]*pf
    return base, bus, branch


def voltage(state):
    return np.column_stack((np.ones(len(state)), state[:, :NB-1]+1j*state[:, NB-1:]))


def feeder_path_km(bus, branch):
    from scipy.sparse.csgraph import shortest_path
    z_base = (bus[0, 9]*1e3)**2/(BASE*1e6)
    f, t = branch[:, 0].astype(int)-1, branch[:, 1].astype(int)-1
    graph = np.zeros((len(bus), len(bus)))
    graph[f, t] = graph[t, f] = np.hypot(branch[:, 2], branch[:, 3])*z_base/CONFIG['line_ohm_per_km']
    return shortest_path(graph, directed=False)


def setup_case(name):
    """Network, PV sites, load regions and object list of one case; module globals as in the 33-bus script."""
    global CASE, BASE, BUS, BRANCH, Y, NB, D, ACTIVE, SITES, CAP, REGIONS, N_REGIONS, OBJECTS, LOSS, NV, NBR, PRIMARY
    CASE = name
    for key in {k for c in CASES.values() for k in c}:
        CONFIG.pop(key, None)
    CONFIG.update(CASE_DEFAULTS); CONFIG.update(CASES[name])
    if name == 'case33bw':
        scope = {}
        archived_functions(SOURCE/'run_typhoon_comparison_K136.py', {'network'}, scope)
        scope.update(ROOT=REPO, re=re, np=np); BASE, BUS, BRANCH = scope['network']()
    else:
        BASE, BUS, BRANCH = network_case141()
    Y, _, _ = admittance(BASE, BUS, BRANCH)
    NB = len(BUS); D = 2*(NB-1)
    assert np.array_equal(BUS[:, 0], np.arange(1, NB+1)) and BUS[0, 1] == 3
    ACTIVE = np.flatnonzero(BRANCH[:, 10] > 0); NBR = len(ACTIVE)
    sites, regions, region_centres = site_regions(BUS, BRANCH)
    if CONFIG['sites'] == 'site_regions':
        SITES = np.array(sites)+1
        CAP = 1.25*np.maximum(BUS[:, 2], 0).sum()/8
        CONFIG['site_capacity_MW'] = CAP
    else:
        SITES = np.array(CONFIG['sites']); CAP = CONFIG['site_capacity_MW']
    CONFIG['sites_used'] = SITES.tolist(); CONFIG['load_region_centres'] = (np.array(region_centres)+1).tolist()
    REGIONS = regions[1:]; N_REGIONS = len(region_centres)
    OBJECTS = [dict(output='V', node=i, branch=0, unit='p.u.', priority=False) for i in range(2, NB+1)]
    OBJECTS += [dict(output=q, node=0, branch=int(i)+1, unit='MW' if q == 'P' else 'MVAr', priority=False)
                for q in ['P', 'Q'] for i in ACTIVE]
    OBJECTS += [dict(output='loss', node=0, branch=0, unit='MW', priority=True)]
    LOSS = len(OBJECTS)-1; NV = NB-1
    PRIMARY = np.array([LOSS])


def outputs(v):
    return np.column_stack((abs(v[:, 1:]), evaluate_outputs(v, BASE, BUS, BRANCH)))


archived_functions(SOURCE/'run_cloud_study.py', {'save_fit', 'weighted_sorted', 'distribution_metrics'}, globals())
archived_functions(FEATURE_SOURCE, {'feature'}, globals())


def load_case_ac():
    archived_functions(SOURCE/'run_cloud_passage_case33.py', {'ac'}, globals(), replace={'ac': [
        ('(n,33)', f'(n,{NB})'), ('(n,65)', f'(n,{2*NBR+1})'), ('target[i,:32]', f'target[i,:{NB-1}]')]})
Q = np.array([.005, .01, .025, .05, .25, .5, .75, .95, .975, .99, .995])
QN = ['q005', 'q01', 'q025', 'q05', 'q25', 'q50', 'q75', 'q95', 'q975', 'q99', 'q995']


def apply_mode(smoke, with_k12=False):
    global SPLITS, KS, N
    if with_k12 and 12 not in CONFIG['K_order']:
        CONFIG['K_order'] = CONFIG['K_order']+[12]
    if smoke:
        CONFIG.update({k: v for k, v in SMOKE.items()})
        CONFIG['splits'] = SMOKE_SPLITS.get(CASE, CONFIG['splits'])
    SPLITS, KS, N = CONFIG['splits'], CONFIG['K_order'], CONFIG['propagation_points']
    archived_functions(SOURCE/'adapted_routines.py', {'fits', 'evaluate', 'summarize'}, globals(),
                       replace={'evaluate': [('2026101928', str(CONFIG['propagation_seed_base']))]})


def load_multiplier(scene):
    return CONFIG['scene_load_multiplier'].get(scene, CONFIG['load_multiplier'])


def site_distance_m():
    if 'site_xy_km' in CONFIG:
        xy = np.array([CONFIG['site_xy_km'][str(b)] for b in SITES])
        return 1000*np.linalg.norm(xy[:, None]-xy[None], axis=2)
    return 1000*feeder_path_km(BUS, BRANCH)[np.ix_(SITES-1, SITES-1)]


def site_correlation():
    return np.exp(-site_distance_m()/(CONFIG['wvm_A_mps']*CONFIG['timescale_s']))


def save_protocol(scene):
    path = O/'protocol.json'
    record = dict(CONFIG, case=CASE, scene=scene, plan_SHA256=sha(PLAN), script_SHA256=sha(Path(__file__)),
                  case_file_SHA256=sha(REPO/CONFIG['case_file']), buses=NB, in_service_branches=NBR,
                  voltage_state_dimension=D, total_base_load_MW=float(BUS[:, 2].sum()),
                  total_base_load_MVAr=float(BUS[:, 3].sum()),
                  archived_source_SHA256={p.name: sha(p) for p in sorted(SOURCE.glob('*.py'))},
                  feature_source_SHA256=sha(FEATURE_SOURCE), site_distance_m=site_distance_m(),
                  site_correlation=site_correlation(),
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
    lower = np.linalg.cholesky(site_correlation())
    forecast, rms = CONFIG['forecast_fraction'], CONFIG['error_rms_fraction']
    kept_s, kept_p, rejected_s = [], [], []
    while sum(len(a) for a in kept_s) < n:
        m = n-sum(len(a) for a in kept_s)+64
        strength = np.exp(rng.normal(-sigma**2, sigma, m)) if sigma > 0 else np.ones(m)
        z = rng.standard_normal((m, len(SITES)))
        p = forecast+rms*strength[:, None]*(z@lower.T)
        valid = np.all((p >= 0) & (p <= 1), axis=1)
        kept_s.append(strength[valid]); kept_p.append(p[valid]); rejected_s.append(strength[~valid])
    strength = np.concatenate(kept_s)[:n]; pv_pu = np.concatenate(kept_p)[:n]
    rejected = np.concatenate(rejected_s)
    load_rng = np.random.Generator(np.random.MT19937(seed+CONFIG['load_stream_offset']))
    rho_g, rho_a, rho_n = CONFIG['load_PQ_correlation']
    wp, wq = np.array(CONFIG['load_P_weights'], float), np.array(CONFIG['load_Q_weights'], float)
    wp *= CONFIG['load_relative_sd']/np.linalg.norm(wp); wq *= CONFIG['load_relative_sd']/np.linalg.norm(wq)
    def paired(width, rho):
        a = load_rng.standard_normal((n, width))
        return a, rho*a+np.sqrt(1-rho*rho)*load_rng.standard_normal((n, width))
    gp, gq = paired(1, rho_g); ap, aq = paired(N_REGIONS, rho_a); zp, zq = paired(NB-1, rho_n)
    fp = 1+wp[0]*gp+wp[1]*ap[:, REGIONS]+wp[2]*zp
    fq = 1+wq[0]*gq+wq[1]*aq[:, REGIONS]+wq[2]*zq
    load_p = load_multiplier(scene)*BUS[None, 1:, 2]*fp
    load_q = load_multiplier(scene)*BUS[None, 1:, 3]*fq
    pv = CAP*pv_pu; pbus = np.zeros((n, NB-1)); pbus[:, SITES-2] = pv
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
        resampled_strength_quantiles=np.quantile(rejected, [.5, .9, .99]).tolist() if len(rejected) else [],
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
    pbus = np.zeros(NB-1); pbus[SITES-2] = CAP*CONFIG['forecast_fraction']
    v, iterations = polar_ac(np.r_[(pbus-load_p)/BASE, -load_q/BASE], Y, tolerance=1e-11)
    out = outputs(v[None])[0]
    rows = [dict(**o, value=float(out[j])) for j, o in enumerate(OBJECTS)]
    csv(d/'forecast_operating_point.csv', rows)
    event('forecast_operating_point', scene=scene, iterations=iterations,
          min_voltage_pu=float(out[:NV].min()), min_voltage_node=int(np.argmin(out[:NV]))+2,
          max_voltage_pu=float(out[:NV].max()), max_voltage_node=int(np.argmax(out[:NV]))+2, loss_MW=float(out[LOSS]))


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


def limit_risk_events(scene, x, targets):
    """Limit-violation events of plan 16.1, fixed from the parameter-estimation and component-choice AC only."""
    median = np.median(x[:, :NV], axis=0)
    low, high = int(np.argmin(median)), int(np.argmax(median))
    rows = []
    def add(name, j, threshold, side, source):
        o = OBJECTS[j]
        rows.append(dict(event=name, object_index=j, output=o['output'], node=o['node'], branch=o['branch'],
                         unit=o['unit'], threshold=float(threshold), side=side, source=source))
    voltage_objects = [(low, 'lowest median voltage bus'), (high, 'highest median voltage bus')]
    voltage_objects += [(int(t['object_index']), 'target') for t in targets if t['output'] == 'V']
    seen = set()
    for j, source in voltage_objects:
        if j in seen:
            continue
        seen.add(j); n = OBJECTS[j]['node']
        add(f'V{n} < 0.95', j, .95, 'below', source); add(f'V{n} > 1.05', j, 1.05, 'above', source)
    for t in targets:
        j, o = int(t['object_index']), OBJECTS[int(t['object_index'])]
        if o['output'] == 'V':
            if t['side'] == 'lower':
                add(f"V{o['node']} < {CONFIG['target_voltage_lower_pu']}", j, CONFIG['target_voltage_lower_pu'], 'below', 'target-specific')
            else:
                add(f"V{o['node']} > {CONFIG['target_voltage_upper_pu']}", j, CONFIG['target_voltage_upper_pu'], 'above', 'target-specific')
        elif o['output'] == 'P':
            impact = t['operating_impact']
            if impact == 'reversal_reverse_to_import':
                add(f"P{o['branch']} > 0 MW", j, 0., 'above', 'target-specific: reverse flow turns to import')
            elif impact == 'reversal_import_to_reverse':
                add(f"P{o['branch']} < 0 MW", j, 0., 'below', 'target-specific: import turns to reverse flow')
            elif impact == 'reverse_flow_increase':
                level = (1+CONFIG['target_reverse_increase'])*float(np.median(x[:, j]))
                add(f"P{o['branch']} < {level:.4f} MW", j, level, 'below', 'target-specific: reverse flow 20% above median')
    csv(O/scene/'limit_risk_events.csv', rows)


def strength_diagnostics(scene, x, strength, voltage_state):
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
        # The all-object comparison and the extreme-voltage limit risk still run (plan 16.1).
        write_json(O/scene/'NO_TARGET.json', dict(stage='ac_only', reason='no object met plan 8.1; criteria not relaxed'))
        event('NO_TARGET', scene=scene)
    if not targets and smoke:
        j = int(np.argmax(x[:, :NV].std(0)))
        targets = [dict(object_index=j, output='V', node=OBJECTS[j]['node'], branch=0, unit='p.u.', side='lower',
                        operating_impact='smoke_only')]
        csv(O/scene/'targets.csv', targets)
    limit_risk_events(scene, x, targets)
    files = [O/'protocol.json', O/scene/'targets.csv', O/scene/'target_rules.txt',
             O/scene/'AC_features_train_validation.csv', O/scene/'strength_diagnostics.json',
             O/scene/'limit_risk_events.csv']
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
    if 'object_index' not in targets:
        targets = pd.DataFrame(columns=['object_index', 'output', 'node', 'branch', 'unit', 'side', 'operating_impact'])
    global PRIMARY
    events = pd.read_csv(O/scene/'limit_risk_events.csv')
    PRIMARY = np.unique(np.r_[targets.object_index.to_numpy(int), events.object_index.to_numpy(int), LOSS])
    for i, obj in enumerate(OBJECTS):
        obj['priority'] = bool(i in PRIMARY)
    return targets


def zero_injection_constraints():
    """Rows of Re/Im(Y V) = 0 at non-slack buses with neither load nor PV, in x = [Re V_2.., Im V_2..]."""
    zero = np.flatnonzero((BUS[1:, 2] == 0) & (BUS[1:, 3] == 0) & ~np.isin(np.arange(2, NB+1), SITES))+1
    G, B = Y.real[zero], Y.imag[zero]
    C = np.r_[np.c_[G[:, 1:], -B[:, 1:]], np.c_[B[:, 1:], G[:, 1:]]]
    c0 = -np.r_[G[:, 0], B[:, 0]]
    return zero, C, c0


def prepare_fitting(scene):
    d = O/scene
    if (d/'fitting_inputs.npz').exists():
        return
    states = {}
    for split in ['training', 'validation']:
        v, _ = load_split(scene, split); states[split] = np.c_[v[:, 1:].real, v[:, 1:].imag]
    center = states['training'].mean(0)
    _, sv_full, _ = np.linalg.svd(states['training']-center, full_matrices=False)
    rank_full = int(np.sum(sv_full > max(states['training'].shape)*np.spacing(sv_full[0])))
    zero, C, c0 = zero_injection_constraints()
    if len(C):
        _, sc, vc = np.linalg.svd(C); rank_c = int(np.sum(sc > sc[0]*1e-12))
        null = vc[rank_c:].T
    else:
        rank_c, null = 0, np.eye(D)
    residual = {s: float(abs(states[s]@C.T-c0).max()) if len(C) else 0. for s in states}
    deviation = states['training']-center
    outside = float(np.linalg.norm(deviation-deviation@null@null.T)/np.linalg.norm(deviation))
    _, sv, vt = np.linalg.svd(deviation@null, full_matrices=False)
    rank = int(np.sum(sv > max(deviation.shape)*np.spacing(sv[0])))
    write_json(d/'voltage_covariance_spectrum.json', dict(
        voltage_state_dimension=D, sample_rank_in_voltage_state=rank_full, singular_values_voltage_state=sv_full,
        zero_injection_non_slack_buses=(zero+1).tolist(), zero_injection_constraint_rank=rank_c,
        constraint_max_residual_pu=residual, relative_variation_outside_constraint_space=outside,
        fitting_dimension=null.shape[1], sample_rank_in_fitting_coordinates=rank, singular_values_fitting=sv,
        std_ratio_strongest_to_weakest=float(sv[0]/sv[-1]),
        rule='plan 16.1: Kirchhoff current law at zero-injection buses is an exact linear relation of the '
             'rectangular voltage; the voltage distribution is fitted in the null space of these relations '
             'and the full voltage is recovered exactly; no direction carrying source-load variation is removed'))
    assert rank == null.shape[1] == D-rank_c and outside < 1e-8, (rank, rank_full, null.shape, rank_c, outside)
    basis = null@vt.T; scale = sv/np.sqrt(len(states['training'])-1)
    x = deviation@basis/scale; z = (states['validation']-center)@basis/scale
    save_npz(d/'fitting_inputs.npz', x=x, validation=z, center=center, basis=basis, scale=scale, singular_values=sv)
    write_json(d/'prepared.json', dict(dimension=null.shape[1], voltage_state_dimension=D, rank=rank,
               zero_injection_constraints_removed=rank_c, no_truncation_of_source_load_variation=True,
               singular_min=float(sv[-1]), fitting_inputs_SHA256=sha(d/'fitting_inputs.npz')))


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


def phase_reference_ac(scene):
    check_lock(scene); ac(scene, 'reference'); prepare_reference(scene)
    event('REFERENCE_AC_COMPLETE', scene=scene)


def phase_reference(scene, only_k=None):
    targets = check_lock(scene)
    ac(scene, 'reference'); prepare_reference(scene)
    for k in ([only_k] if only_k else KS):
        evaluate(scene, k)
    if not only_k:
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
    pbus = np.zeros(NB-1); pbus[SITES-2] = CAP*CONFIG['forecast_fraction']
    v, _ = polar_ac(np.r_[(pbus-load_p)/BASE, -load_q/BASE], Y, tolerance=1e-11)
    sensitivity = np.linalg.inv(rect_jacobian(v, Y))[:, SITES-2]
    fit_in = np.load(d/'fitting_inputs.npz')
    whitened = (fit_in['basis'].T@sensitivity)/fit_in['scale'][:, None]
    q1, s, _ = np.linalg.svd(whitened, full_matrices=False)
    rank = int(np.sum(s > s[0]*1e-10)); dfit = whitened.shape[0]
    Q1 = q1[:, :rank]; Q2 = np.linalg.svd(np.eye(dfit)-Q1@Q1.T)[0][:, :dfit-rank]
    assert np.allclose(Q1.T@Q2, 0, atol=1e-10) and np.allclose(np.c_[Q1, Q2].T@np.c_[Q1, Q2], np.eye(dfit), atol=1e-10)
    # PV sensitivities keep zero current at zero-injection buses, so they lie inside the fitting coordinates.
    represented = fit_in['basis']@(fit_in['basis'].T@sensitivity)
    outside = float(np.linalg.norm(sensitivity-represented)/np.linalg.norm(sensitivity))
    x = fit_in['x']; s_train = np.load(d/'training_inputs.npz')['strength']
    U, W = x@Q1, x@Q2
    corr = lambda a: [float(np.corrcoef(c**2, s_train**2)[0, 1]) for c in a.T]
    diagnostics = dict(rank=rank, sensitivity_singular_values=s.tolist(),
        u_excess_kurtosis=kurtosis(U, axis=0).tolist(), w_excess_kurtosis_median=float(np.median(kurtosis(W, axis=0))),
        w_excess_kurtosis_max=float(kurtosis(W, axis=0).max()), u_square_strength_correlation=corr(U),
        w_square_strength_correlation_median=float(np.median(corr(W))),
        training_variance_share_u=float(np.sum(((x*fit_in['scale'])@Q1)**2)/np.sum((x*fit_in['scale'])**2)),
        sensitivity_outside_fitting_coordinates=outside,
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
                train_ll=best['train_ll'], status=best['status'], iterations=best['iterations'], dimension=Q1.shape[0],
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
    if 'block' in getattr(model, 'files', model):
        return propagate_block(model, n, seed)
    return propagate(model, n, seed, 'sobol', randomizer='MT19937')


def repeat_summary(frame, metrics, keys):
    a = frame.groupby(keys, dropna=False)[metrics].agg(['mean', 'min', 'max'])
    a.columns = ['_'.join(c) for c in a.columns]
    return a.reset_index()


def phase_block_fit(scene, only_k=None):
    check_lock(scene); d = O/scene
    Q1, Q2 = pv_voltage_subspace(scene)
    selected, attempts = [], []
    for k in ([only_k] if only_k else KS):
        row, rows = fit_block_candidates(scene, k, Q1, Q2); selected.append(row); attempts += rows
        event('selected', scene=scene, K=k, family=BLOCK_FAMILY, candidate=row['candidate'], df=row['df'])
    if not only_k:
        csv(d/'selected_models_block.csv', selected); csv(d/'all_fit_attempts_block.csv', attempts)
    event('BLOCK_FIT_COMPLETE', scene=scene, K=only_k or 'all')


def phase_block(scene, only_k=None):
    targets = check_lock(scene); d = O/scene
    phase_block_fit(scene)
    archived_functions(SOURCE/'adapted_routines.py', {'evaluate'}, globals(), replace={'evaluate': [
        ('2026101928', str(CONFIG['propagation_seed_base'])), ('def evaluate(scene,k):', 'def evaluate_block(scene,k):'),
        ("'selected_models.csv'", "'selected_models_block.csv'"),
        ("propagate(m,N,seed,'sobol',randomizer='MT19937')", 'propagate_any(m,N,seed)')]})
    for k in ([only_k] if only_k else KS):
        evaluate_block(scene, k)
    if only_k:
        return
    frames = [pd.read_csv(p) for p in sorted((d/'propagation').glob(f'{BLOCK_FAMILY}_*_global.csv'))]
    qframes = [pd.read_csv(p) for p in sorted((d/'propagation').glob(f'{BLOCK_FAMILY}_*_quantiles.csv'))]
    gb = repeat_summary(pd.concat(frames), SUMMARY_GLOBAL, SUMMARY_KEYS)
    qb = repeat_summary(pd.concat(qframes), SUMMARY_QUANTILE, SUMMARY_KEYS+['q'])
    gb.to_csv(O/'block_global_repeat_summary.csv', index=False); qb.to_csv(O/'block_quantile_repeat_summary.csv', index=False)
    gs = pd.concat([pd.read_csv(O/'global_repeat_summary.csv'), gb], ignore_index=True)
    qs = pd.concat([pd.read_csv(O/'quantile_repeat_summary.csv'), qb], ignore_index=True)
    judge(scene, targets, challenger=BLOCK_FAMILY, gs=gs, qs=qs, prefix='block_effectiveness')


def main():
    global O
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['ac_only', 'fit', 'block_fit', 'reference_ac', 'reference', 'block'])
    parser.add_argument('--case', choices=sorted(CASES), default='case33bw')
    parser.add_argument('--scene', choices=SCENES, default='common_strength')
    parser.add_argument('--out', help='result directory; default from the case table')
    parser.add_argument('--smoke', action='store_true', help='small scenario counts to check the program only')
    parser.add_argument('--with-K12', action='store_true', help='add K = 12 (plan 16: only if time allows)')
    parser.add_argument('--K', type=int, help='one K only; finish with a plain run to collect all K')
    parser.add_argument('--fixed-targets', help='targets.csv locked by the main run; the own-rule selection is kept aside')
    parser.add_argument('--training-seed', type=int); parser.add_argument('--validation-seed', type=int)
    args = parser.parse_args()
    global FIXED_TARGETS
    setup_case(args.case); load_case_ac()
    O = Path(args.out or REPO/CONFIG['default_out'].format(scene=args.scene)); O.mkdir(parents=True, exist_ok=True)
    FIXED_TARGETS = Path(args.fixed_targets) if args.fixed_targets else None
    if args.training_seed:
        CONFIG['seeds']['training'] = args.training_seed
    if args.validation_seed:
        CONFIG['seeds']['validation'] = args.validation_seed
    apply_mode(args.smoke, args.with_K12)
    if args.scene == 'gaussian_control' and FIXED_TARGETS is None:
        raise RuntimeError('Gaussian control is judged on the targets fixed by the main scene')
    try:
        {'ac_only': lambda: phase_ac_only(args.scene, args.smoke),
         'fit': lambda: phase_fit(args.scene, args.K),
         'block_fit': lambda: phase_block_fit(args.scene, args.K),
         'reference_ac': lambda: phase_reference_ac(args.scene),
         'reference': lambda: phase_reference(args.scene, args.K),
         'block': lambda: phase_block(args.scene, args.K)}[args.phase]()
    except Exception:
        write_json(O/f'failure_{args.phase}_{os.getpid()}.json',
                   dict(error=traceback.format_exc(), UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        raise


if __name__ == '__main__':
    main()
