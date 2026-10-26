"""Cloud-only independent Python implementation; original MATLAB is retained read-only.

One Gamma precision per standard multivariate t component. Scatter floor matches
the historical bounded likelihood. Python random streams do not replay MATLAB.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import re
import time
import sys
import os
import platform
import shutil
import numpy as np
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.linalg import solve_triangular, cholesky
from scipy.special import gammaln, digamma, logsumexp, ndtri, gammaincinv
from scipy.optimize import brentq
from scipy.stats import qmc, truncnorm, norm, multivariate_t, multivariate_normal
from scipy.sparse.csgraph import shortest_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'result/case33bw/cloud_audit'
OUT.mkdir(parents=True, exist_ok=True)
REF = '0d5c999db4cbee2ea7055844d31b729d9694d747'
FLOOR = 1e-7
QUANTILES = np.array([.005, .01, .025, .05, .95, .975, .99, .995])
PROP_SEEDS = [2026101901, 2026101902, 2026101903, 2026101904]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path = Path(path)
    temporary = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    default=convert, allow_nan=True))
    os.replace(temporary, path)


def environment():
    import scipy
    from threadpoolctl import threadpool_info
    return dict(location='OpenAI hosted conversation workspace shell',
                platform=platform.platform(), python=sys.version,
                numpy=np.__version__, scipy=scipy.__version__,
                hostname=platform.node(), cwd=os.getcwd(), cpu_count=os.cpu_count(),
                matlab_on_path=shutil.which('matlab'), octave_on_path=shutil.which('octave'),
                blas=threadpool_info(), user_mac_used=False, github_actions_used=False,
                cloud_provider_physical_ownership='not independently observable from shell',
                timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))


def read_sources():
    import h5py
    text = (ROOT / 'data/cases/case33bw_modify.m').read_text()
    clean = re.sub(r'%[^\n]*', '', text)
    tables = {}
    for name in ('bus', 'branch'):
        match = re.search(r'mpc\.' + name + r'\s*=\s*\[(.*?)\];', clean, re.S)
        tables[name] = np.array([[float(v) for v in row.split()]
                                for row in match[1].split(';') if row.strip()])
    base = float(re.search(r'mpc.baseMVA\s*=\s*([\d.]+)', clean)[1])
    bus, branch = tables['bus'], tables['branch']
    bus[:, 2:4] /= 1000
    branch[:, 2:4] /= (bus[0, 9] * 1000)**2 / (base * 1e6)
    raw = loadmat(ROOT / 'data/Load_Data.mat')['__function_workspace__'].tobytes()
    header = b'MATLAB 5.0 MAT-file'.ljust(116, b' ') + b'\0'*8 + b'\0\1IM'
    reader = MatFile5Reader(BytesIO(header + raw[8:]), struct_as_record=False,
                            squeeze_me=True)
    reader.initialize_read()
    reader.read_file_header()
    hdr, _ = reader.read_var_header()
    obj = reader.read_var_array(hdr, True)
    cells = obj.MCOS['arr'][0]
    profiles = np.column_stack(list(cells[2]))
    load_names = list(cells[7])
    assert profiles.shape == (8760, 24)
    assert len(load_names) == 24 and float(cells[4]) == 8760
    assert np.isfinite(profiles).all() and np.min(profiles) >= 0
    with h5py.File(ROOT / 'data/luan_pv_8760_pu_timetable.mat') as f:
        pv = np.column_stack([f[r][()].ravel() for r in f['#refs#/e/data'][()].ravel()])
        assert int(f['#refs#/e/numRows'][0, 0]) == 8760
        pv_names = [''.join(chr(int(x)) for x in f[r][()].ravel())
                    for r in f['#refs#/e/varNames'][()].ravel()]
    assert pv.shape == (8760, 12) and np.isfinite(pv).all()
    pv = np.clip(pv, 0, 1)
    profiles /= profiles.mean(axis=0)
    return base, bus, branch, profiles, pv, load_names, pv_names


def admittance(base, bus, branch):
    n = len(bus)
    f, t = branch[:, 0].astype(int)-1, branch[:, 1].astype(int)-1
    status = (branch[:, 10] > 0).astype(float)
    ys = status / (branch[:, 2] + 1j*branch[:, 3])
    bc = status * branch[:, 4]
    tap = np.where(branch[:, 8] == 0, 1., branch[:, 8]) * np.exp(1j*np.deg2rad(branch[:, 9]))
    ytt = ys + .5j*bc
    yff = ytt / abs(tap)**2
    yft, ytf = -ys/np.conj(tap), -ys/tap
    yf, yt = np.zeros((len(branch), n), complex), np.zeros((len(branch), n), complex)
    for l, (a, b) in enumerate(zip(f, t)):
        yf[l, a], yf[l, b] = yff[l], yft[l]
        yt[l, a], yt[l, b] = ytf[l], ytt[l]
    cf, ct = np.eye(n)[f], np.eye(n)[t]
    y = cf.T @ yf + ct.T @ yt + np.diag((bus[:, 4]+1j*bus[:, 5])/base)
    return y, yf, yt


def evaluate_outputs(v, base, bus, branch):
    _, yf, yt = admittance(base, bus, branch)
    f, t = branch[:, 0].astype(int)-1, branch[:, 1].astype(int)-1
    sf = base*v[:, f]*np.conj(v @ yf.T)
    st = base*v[:, t]*np.conj(v @ yt.T)
    active = branch[:, 10] > 0
    return np.column_stack((sf[:, active].real, sf[:, active].imag,
                            (sf[:, active]+st[:, active]).real.sum(axis=1)))


def independent_outputs(v, base, branch):
    f, t = branch[:, 0].astype(int)-1, branch[:, 1].astype(int)-1
    status = branch[:, 10] > 0
    tap = np.where(branch[:, 8] == 0, 1., branch[:, 8]) * np.exp(1j*np.deg2rad(branch[:, 9]))
    a, b = v[:, f]/tap, v[:, t]
    current = (a-b)/(branch[:, 2]+1j*branch[:, 3])
    sf = base*a*np.conj(current + .5j*branch[:, 4]*a)*status
    st = base*b*np.conj(-current + .5j*branch[:, 4]*b)*status
    loss = base*np.sum(abs(a-b)**2 * (1/(branch[:, 2]+1j*branch[:, 3])).real*status, axis=1)
    return np.column_stack((sf[:, status].real, sf[:, status].imag, loss))


def reduced_power(x, y):
    d = x.shape[1]//2
    v = np.column_stack((np.ones(len(x)), x[:, :d]+1j*x[:, d:]))
    s = v*np.conj(v @ y.T)
    return np.column_stack((s[:, 1:].real, s[:, 1:].imag))


def rect_jacobian(v, y):
    g, b = y.real, y.imag
    e, f = v.real, v.imag
    ir, ii = (y @ v).real, (y @ v).imag
    j = np.block([[np.diag(ir)+e[:, None]*g+f[:, None]*b,
                   np.diag(ii)-e[:, None]*b+f[:, None]*g],
                  [f[:, None]*g-np.diag(ii)-e[:, None]*b,
                   np.diag(ir)-f[:, None]*b-e[:, None]*g]])
    ix = np.r_[np.arange(1, len(v)), np.arange(len(v)+1, 2*len(v))]
    return j[np.ix_(ix, ix)]


def polar_ac(target, y, tolerance=1e-11):
    # Independent Newton AC solve, with a polar Jacobian and a flat start.
    v = np.ones(len(y), complex)
    for it in range(40):
        current = y @ v
        s = v*np.conj(current)
        residual = target - np.r_[s[1:].real, s[1:].imag]
        if max(abs(residual)) < tolerance:
            return v, it
        unit = v/abs(v)
        da = 1j*np.diag(v) @ np.conj(np.diag(current)-y @ np.diag(v))
        dm = np.diag(v) @ np.conj(y @ np.diag(unit)) + np.conj(np.diag(current)) @ np.diag(unit)
        j = np.block([[da[1:, 1:].real, dm[1:, 1:].real],
                      [da[1:, 1:].imag, dm[1:, 1:].imag]])
        delta = np.linalg.solve(j, residual)
        angle, mag = np.angle(v), abs(v)
        angle[1:] += delta[:len(y)-1]
        mag[1:] += delta[len(y)-1:]
        v = mag*np.exp(1j*angle)
    raise RuntimeError('Independent polar AC did not converge')


def ptov(target, y):
    start = time.perf_counter()
    n, d = target.shape
    center = target.mean(axis=0)
    next_index = np.argmin(np.sum((target-center)**2, axis=1))
    state = np.full_like(target, np.nan)
    accepted = np.zeros(n, bool)
    best_score = np.full(n, np.inf)
    assignments = np.full(n, -1)
    total_corrections = 0
    for anchor_number in range(12):
        v, _ = polar_ac(target[next_index], y)
        anchor = np.r_[v[1:].real, v[1:].imag]
        power = reduced_power(anchor[None, :], y)[0]
        inverse = np.linalg.inv(rect_jacobian(v, y))
        inverse_norm = np.linalg.norm(inverse, 2)
        idx = np.flatnonzero(~accepted)
        x = anchor + (target[idx]-power) @ inverse.T
        x -= (reduced_power(x, y)-target[idx]) @ inverse.T
        x += (target[idx]-reduced_power(x, y)) @ inverse.T
        if next_index in idx:
            x[np.flatnonzero(idx == next_index)[0]] = anchor
        residual = target[idx]-reduced_power(x, y)
        stalled = np.zeros(len(idx), bool)
        for _ in range(10):
            vm = np.hypot(x[:, :d//2], x[:, d//2:])
            inside = np.all((vm >= .5) & (vm <= 1.5), axis=1)
            passed = inside & (inverse_norm*np.linalg.norm(residual, axis=1) <= 1e-7) & (np.max(abs(residual), axis=1) <= 1e-8)
            active = np.flatnonzero(~passed & inside & ~stalled)
            if not len(active):
                break
            delta = residual[active] @ inverse.T
            old_norm = np.linalg.norm(residual[active], axis=1)
            pending = np.arange(len(active))
            moved = np.zeros(len(active), bool)
            for reduction in range(9):
                trial = x[active[pending]] + delta[pending]*(.5**reduction)
                tr = target[idx[active[pending]]] - reduced_power(trial, y)
                vm = np.hypot(trial[:, :d//2], trial[:, d//2:])
                good = np.all((vm >= .5) & (vm <= 1.5), axis=1) & (np.linalg.norm(tr, axis=1) < old_norm[pending])
                chosen = pending[good]
                x[active[chosen]], residual[active[chosen]] = trial[good], tr[good]
                moved[chosen] = True
                total_corrections += len(chosen)
                pending = pending[~good]
                if not len(pending):
                    break
            stalled[active[~moved]] = True
        vm = np.hypot(x[:, :d//2], x[:, d//2:])
        score = inverse_norm*np.linalg.norm(residual, axis=1)
        inside = np.all((vm >= .5) & (vm <= 1.5), axis=1)
        score[~inside] = np.inf
        passed = inside & (score <= 1e-7) & (np.max(abs(residual), axis=1) <= 1e-8)
        improved = (score < best_score[idx]) | passed
        state[idx[improved]] = x[improved]
        best_score[idx[improved]] = score[improved]
        assignments[idx[improved]] = anchor_number
        accepted[idx[improved]] = passed[improved]
        if accepted.all():
            break
        rank_score = best_score.copy()
        rank_score[accepted] = -np.inf
        next_index = np.argmax(rank_score)
    failed = np.flatnonzero(~accepted)
    for i in failed:
        v, _ = polar_ac(target[i], y)
        state[i] = np.r_[v[1:].real, v[1:].imag]
    residual = target-reduced_power(state, y)
    maximum = np.max(abs(residual))
    assert maximum <= 1e-8 and np.isfinite(state).all()
    return state, dict(anchor_count=anchor_number+1, fallback_count=len(failed),
                       maximum_power_residual_pu=maximum, total_corrections=total_corrections,
                       seconds=time.perf_counter()-start, assigned_anchor=assignments)


def site_regions(bus, branch):
    n = len(bus)
    f, t = branch[:, 0].astype(int)-1, branch[:, 1].astype(int)-1
    graph = np.zeros((n, n))
    graph[f, t] = graph[t, f] = np.hypot(branch[:, 2], branch[:, 3])
    distance = shortest_path(graph, directed=False)[0]
    degree = (graph > 0).sum(axis=0)
    load = np.maximum(bus[:, 2], 0)
    threshold = np.sort(load[load > 0])[int(np.ceil(.75*np.sum(load > 0)))-1]
    eligible = (load <= threshold+1e-12) & (np.arange(n) != 0)
    def ranking(ids, a, b):
        score = a*distance[ids]/max(distance)+b*(1-load[ids]/max(load))
        return ids[np.lexsort((ids, load[ids], -distance[ids], -score))]
    sites = list(ranking(np.flatnonzero(eligible & (degree == 1)), .6, .4)[:8])
    remaining = np.array([i for i in np.flatnonzero(eligible) if i not in sites])
    sites += list(ranking(remaining, .7, .3)[:8-len(sites)])
    sites = np.array(sorted(sites, key=lambda i: (-distance[i], i)))
    topo = shortest_path((graph > 0).astype(float), directed=False)
    seeds = [int(np.argmax(topo[0]))]
    for _ in range(3):
        seeds.append(int(np.argmax(np.min(topo[seeds], axis=0))))
    regions = np.argmin(topo[seeds], axis=0)
    return sites, regions, seeds


def amplitude_second_moment():
    # Exact second moment of clip(exp(.7Z), .3, 3), Z standard normal.
    a, b, sigma = .3, 3., .7
    lo, hi = np.log(a)/sigma, np.log(b)/sigma
    return a*a*norm.cdf(lo)+np.exp(2*sigma*sigma)*(norm.cdf(hi-2*sigma)-norm.cdf(lo-2*sigma))+b*b*norm.sf(hi)


def generate(scene, count, seed, source):
    base, bus, branch, profiles, pv, _, _ = source
    sites, regions, region_seeds = site_regions(bus, branch)
    rng = np.random.Generator(np.random.MT19937(seed))
    weather_rng = np.random.Generator(np.random.MT19937(seed+700001))
    family = 'heavy' if scene == 'A_heavy' else 'light'
    def innovation(shape):
        if family == 'heavy':
            return rng.laplace(size=shape)/np.sqrt(2)
        return rng.standard_normal(shape)
    weather = scene in ['N0', 'H', 'C']
    pool = np.flatnonzero(np.all(pv[:, :8] >= .10, axis=1)) if weather else np.arange(8760)
    hours = pool[rng.integers(len(pool), size=count)]
    loads = np.flatnonzero((bus[:, 2] > 0) | (abs(bus[:, 3]) > 0))
    loads = loads[loads != 0]
    cols = np.arange(len(loads)) % profiles.shape[1]
    qcols = (cols+1) % profiles.shape[1]
    gp = innovation((count, 1))
    gq = .35*gp+np.sqrt(1-.35**2)*innovation((count, 1))
    ap = innovation((count, 4))
    aq = .35*ap+np.sqrt(1-.35**2)*innovation((count, 4))
    if scene == 'H':
        amp = np.clip(np.exp(.7*weather_rng.standard_normal((count, 4))), .3, 3)/np.sqrt(amplitude_second_moment())
        ap, aq = ap*amp, aq*amp
    np_ = innovation((count, len(loads)))
    nq = .25*np_+np.sqrt(1-.25**2)*innovation((count, len(loads)))
    fp = 1+.03*gp+.08*ap[:, regions[loads]]+.15*np_
    fq = 1+.04*gq+.10*aq[:, regions[loads]]+.20*nq
    clip_p = np.mean((fp < .5) | (fp > 1.6))
    clip_q = np.mean((fq < .4) | (fq > 1.8))
    fp, fq = np.clip(fp, .5, 1.6), np.clip(fq, .4, 1.8)
    history = profiles[hours[:, None], cols]
    qhistory = .75*history+.25*profiles[hours[:, None], qcols]
    pd = np.broadcast_to(bus[:, 2], (count, len(bus))).copy()
    qd = np.broadcast_to(bus[:, 3], (count, len(bus))).copy()
    pd[:, loads] *= history*fp
    qd[:, loads] *= qhistory*fq
    temp = np.full((count, 4), 25.)
    if scene in ['H', 'C']:
        mean, std, lower, upper = (38, 1.5, 35, 42) if scene == 'H' else (28, 1, 25, 32)
        temp = truncnorm.rvs((lower-mean)/std, (upper-mean)/std, loc=mean,
                             scale=std, size=(count, 4), random_state=weather_rng)
    if scene == 'H':
        pd[:, loads] *= 1+.02*np.maximum(temp[:, regions[loads]]-30, 0)
        qd[:, loads] *= 1+.025*np.maximum(temp[:, regions[loads]]-30, 0)
    cloud_area, cloud_node = innovation((count, 4)), innovation((count, 8))
    availability = pv[hours[:, None], np.arange(8)]
    if scene == 'C':
        amp = np.clip(np.exp(.7*weather_rng.standard_normal((count, 1))), .3, 3)
        tau = np.clip(.75+.20*amp*cloud_area[:, regions[sites]]+.05*cloud_node, .05, 1)
        availability = availability*tau
    else:
        availability = availability*np.maximum(0, 1+.08*cloud_area[:, regions[sites]]+.10*cloud_node)
    if scene in ['H', 'C']:
        availability *= 1-.0037*(temp[:, regions[sites]]-25)
    capacity = 1.25*sum(np.maximum(bus[:, 2], 0))/8
    generation = capacity*np.clip(availability, 0, 1)
    da, dn = innovation((count, 4)), innovation((count, 8))
    reactive_generation = .20*np.tanh(.6*da[:, regions[sites]]+dn)*generation
    assert np.max(np.hypot(generation, reactive_generation)) <= 1.05*capacity+1e-12
    pd[:, sites] -= generation
    qd[:, sites] -= reactive_generation
    target = np.column_stack((-pd[:, 1:]/base, -qd[:, 1:]/base))
    metadata = dict(scene=scene, seed=seed, count=count, hours=hours+1,
                    day_pool=pool+1, pv_buses=sites+1, regions=regions+1,
                    region_seeds=np.array(region_seeds)+1, capacity_per_site_mw=capacity,
                    clipping_p=clip_p, clipping_q=clip_q, temperature_c=temp,
                    amplitude_second_moment=amplitude_second_moment(),
                    load_p_mw=pd.sum(axis=1)+generation.sum(axis=1),
                    load_q_mvar=qd.sum(axis=1)+reactive_generation.sum(axis=1),
                    pv_p_mw=generation.sum(axis=1))
    return target, metadata


def floor_matrix(a, floor=FLOOR):
    a = (a+a.T)/2
    try:
        cholesky(a-np.eye(len(a))*(floor+64*len(a)*np.spacing(np.linalg.norm(a, 1))), lower=True)
        return a, 0
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(a)
        return (vectors*np.maximum(values, floor)) @ vectors.T, int(np.sum(values < floor))


def px_scale(a, mean_precision, nu, floor=FLOOR):
    a = (a+a.T)/2
    trial = a/mean_precision
    try:
        cholesky(trial-np.eye(len(a))*(floor+64*len(a)*np.spacing(np.linalg.norm(trial, 1))), lower=True)
        return trial, mean_precision, 0
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(a)
        ratios = np.sort(values/floor)
        alpha = mean_precision
        active = 0
        while active < len(a) and ratios[active] < alpha:
            active += 1
            alpha += (ratios[active-1]-alpha)/(nu+active)
        result = (vectors*np.maximum(values/alpha, floor)) @ vectors.T
        assert abs(nu*(alpha-mean_precision)+np.maximum(0, alpha-ratios).sum()) <= 1e-8*max(1, nu*mean_precision)
        return result, alpha, active


def log_density(x, model):
    n, d = x.shape
    k = len(model['weights'])
    logs, distances = np.empty((n, k)), np.empty((n, k))
    for j in range(k):
        l = cholesky(model['scatter'][j], lower=True, check_finite=False)
        z = solve_triangular(l, (x-model['means'][j]).T, lower=True, check_finite=False).T
        delta = np.einsum('ij,ij->i', z, z)
        nu = model['df'][j]
        if np.isinf(nu):
            value = -.5*(d*np.log(2*np.pi)+delta)
        else:
            value = gammaln((nu+d)/2)-gammaln(nu/2)-d/2*np.log(nu*np.pi)-(nu+d)/2*np.log1p(delta/nu)
        logs[:, j] = np.log(model['weights'][j])+value-np.log(l.diagonal()).sum()
        distances[:, j] = delta
    density = logsumexp(logs, axis=1)
    return density, np.exp(logs-density[:, None]), distances


def initialize(x, k, partition, scale, population_covariance=False):
    n, d = x.shape
    seed = 202610190+partition
    rng = np.random.default_rng(seed)
    if partition < 2:
        score = x[:, partition]*scale[partition]
        labels = np.empty(n, int)
        labels[np.argsort(score, kind='stable')] = np.minimum(k-1, np.arange(n)*k//n)
        rule = 'PC'+str(partition+1)+' deterministic balanced'
    elif partition % 5 == 4:
        labels = np.empty(n, int)
        labels[rng.permutation(n)] = np.minimum(k-1, np.arange(n)*k//n)
        rule = 'random balanced membership'
    else:
        direction = np.zeros(d)
        direction[:min(d, 8)] = rng.standard_normal(min(d, 8))
        direction /= np.linalg.norm(direction)
        score = x @ direction
        labels = np.empty(n, int)
        labels[np.argsort(score, kind='stable')] = np.minimum(k-1, np.arange(n)*k//n)
        rule = 'seeded random direction balanced split'
    means, covs, weights = [], [], []
    for j in range(k):
        group = x[labels == j]
        means.append(group.mean(axis=0))
        covs.append(floor_matrix(np.cov(group, rowvar=False, bias=population_covariance))[0])
        weights.append(len(group)/n)
    model = dict(weights=np.array(weights), means=np.array(means), scatter=np.array(covs),
                 df=np.full(k, np.inf))
    return model, labels, dict(partition=partition+1, seed=seed, rule=rule)


def fit(x, initial, acceleration='EM', maximum=300, tolerance=1e-6,
        actual_covariance_floor=False, relative_tolerance=False):
    start = time.perf_counter()
    model = {k: np.array(initial[k], copy=True) for k in ['weights', 'means', 'scatter', 'df']}
    density, r, delta = log_density(x, model)
    history = [density.mean()]
    floor_history = []
    status = 'maximum_iterations'
    n, d = x.shape
    for iteration in range(1, maximum+1):
        count = r.sum(axis=0)
        if count.min() <= d+2:
            status = 'collapsed_component'
            break
        updated = dict(weights=count/n, means=model['means'].copy(),
                       scatter=model['scatter'].copy(), df=model['df'].copy())
        active = []
        for j, nu in enumerate(model['df']):
            scatter_floor = FLOOR if not actual_covariance_floor or np.isinf(nu) else FLOOR*(nu-2)/nu
            u = np.ones(n) if np.isinf(nu) else (nu+d)/(nu+delta[:, j])
            w = r[:, j]*u
            updated['means'][j] = (w @ x)/sum(w)
            residual = x-updated['means'][j]
            a = (residual.T @ (residual*w[:, None]))/count[j]
            if np.isfinite(nu):
                elog = digamma((nu+d)/2)-np.log((nu+delta[:, j])/2)
                mean_u = w.sum()/count[j]
                statistic = (r[:, j] @ (elog-u))/count[j]
                if acceleration == 'PX':
                    updated['scatter'][j], alpha, floor_count = px_scale(a, mean_u, nu, scatter_floor)
                    statistic = (r[:, j] @ elog)/count[j]-np.log(alpha)-mean_u/alpha
                else:
                    updated['scatter'][j], floor_count = floor_matrix(a, scatter_floor)
                def score(v):
                    return np.log(v/2)-digamma(v/2)+1+statistic
                lower, upper = 4.1, 1e5
                if actual_covariance_floor:
                    smallest = np.linalg.eigvalsh(updated['scatter'][j])[0]
                    if smallest < FLOOR:
                        upper = min(upper, max(lower, 2/(1-smallest/FLOOR)))
                updated['df'][j] = lower if score(lower) <= 0 else upper if score(upper) >= 0 else brentq(score, lower, upper, xtol=1e-8)
            else:
                updated['scatter'][j], floor_count = floor_matrix(a)
            active.append(floor_count)
        density, new_r, new_delta = log_density(x, updated)
        value = density.mean()
        gain = value-history[-1]
        if gain < -1e-8:
            status = 'rejected_decreasing_update'
            break
        model = updated
        r, delta = new_r, new_delta
        history.append(value)
        floor_history.append(active)
        scale = max(1., abs(history[-2])) if relative_tolerance else 1.
        if abs(gain) < tolerance*scale:
            status = 'converged'
            break
    eig = np.linalg.eigvalsh(model['scatter'])
    audit = dict(status=status, iterations=iteration, history=np.array(history),
                 train_ll=history[-1], minimum_effective_count=r.sum(axis=0).min(),
                 effective_counts=r.sum(axis=0), df=model['df'], weights=model['weights'],
                 scatter_eigen_min=eig[:, 0], scatter_condition=eig[:, -1]/eig[:, 0],
                 floor_active_max=int(np.max(floor_history)) if floor_history else 0,
                 floor_history=np.array(floor_history), seconds=time.perf_counter()-start,
                 actual_covariance_floor=actual_covariance_floor,
                 relative_tolerance=relative_tolerance)
    return model, audit


def propagate(model, n, seed, method, randomizer='default'):
    k, d = model['means'].shape
    rng = np.random.Generator(np.random.MT19937(seed)) if randomizer=='MT19937' else np.random.default_rng(seed)
    if method == 'pseudo':
        labels = rng.choice(k, size=n, p=model['weights'])
        counts = np.bincount(labels, minlength=k)
    else:
        desired = n*model['weights']
        counts = np.maximum(1, np.floor(desired).astype(int))
        while counts.sum() < n:
            counts[np.argmax(desired-counts)] += 1
        while counts.sum() > n:
            candidates = np.flatnonzero(counts > 1)
            counts[candidates[np.argmin((desired-counts)[candidates])]] -= 1
    samples, weights = [], []
    for j, nj in enumerate(counts):
        if nj == 0:
            continue
        nu = model['df'][j]
        if method == 'pseudo':
            z = rng.standard_normal((nj, d))
            u = np.ones(nj) if np.isinf(nu) else rng.gamma(nu/2, 2/nu, nj)
        else:
            scramble_seed = np.random.Generator(np.random.MT19937(seed+1009*j)) if randomizer=='MT19937' else seed+1009*j
            sampler = qmc.Sobol(d+int(np.isfinite(nu)), scramble=True, seed=scramble_seed)
            uniform = sampler.random_base2(int(np.ceil(np.log2(nj))))[:nj]
            uniform = np.clip(uniform, np.finfo(float).eps, 1-np.finfo(float).eps)
            z = ndtri(uniform[:, :d])
            u = np.ones(nj) if np.isinf(nu) else 2*gammaincinv(nu/2, uniform[:, d])/nu
        l = cholesky(model['scatter'][j], lower=True)
        samples.append(model['means'][j]+(z @ l.T)/np.sqrt(u[:, None]))
        weights.append(np.full(nj, 1/n if method == 'pseudo' else model['weights'][j]/nj))
    return np.concatenate(samples), np.concatenate(weights)


def implementation_checks(source):
    base, bus, branch, _, _, _, _ = source
    rng = np.random.default_rng(48193)
    d = 5
    a = rng.normal(size=(d, d))
    scatter = a @ a.T+np.eye(d)*.3
    mean = rng.normal(size=d)
    x = rng.normal(size=(1000, d))
    checks = {}
    for nu in [4.1, 5., 12., 30., 200., 1e5, np.inf]:
        model = dict(weights=np.array([1.]), means=mean[None], scatter=scatter[None], df=np.array([nu]))
        calculated = log_density(x, model)[0]
        reference = multivariate_normal.logpdf(x, mean, scatter) if np.isinf(nu) else multivariate_t.logpdf(x, mean, scatter, nu)
        err = max(abs(calculated-reference))
        checks['logpdf_max_error_df_'+str(nu)] = err
        assert err < 1e-9
    normal_lp = multivariate_normal.logpdf(x, mean, scatter)
    model['df'][:] = 1e5
    checks['gaussian_limit_logpdf_error_df_1e5'] = max(abs(log_density(x, model)[0]-normal_lp))
    for nu in [8., np.inf]:
        known = dict(weights=np.array([.4, .6]), means=np.array([mean-1, mean+1]),
                     scatter=np.array([scatter, scatter*.6]), df=np.array([nu, nu]))
        draws, w = propagate(known, 65536, 9161, 'sobol')
        expected_mean = known['weights'] @ known['means']
        covs = known['scatter']*(1 if np.isinf(nu) else nu/(nu-2))
        expected_cov = sum(known['weights'][j]*(covs[j]+np.outer(known['means'][j]-expected_mean, known['means'][j]-expected_mean)) for j in range(2))
        actual_mean = w @ draws
        actual_cov = (draws-actual_mean).T @ ((draws-actual_mean)*w[:, None])
        checks['sampling_mean_max_error_df_'+str(nu)] = max(abs(actual_mean-expected_mean))
        checks['sampling_cov_relative_error_df_'+str(nu)] = np.linalg.norm(actual_cov-expected_cov)/np.linalg.norm(expected_cov)
        assert np.linalg.norm(actual_cov-expected_cov)/np.linalg.norm(expected_cov) < .03
        shuffled = rng.permutation(len(draws))[:4096]
        fitted, audit = fit(draws[shuffled], known, 'PX', maximum=400)
        checks['known_model_fit_df_'+str(nu)] = {k: v for k, v in audit.items() if k not in ['history', 'floor_history']}
        assert np.min(np.diff(audit['history'])) >= -1e-8
        assert audit['status'] == 'converged'
        checks['known_model_fit_mean_max_error_df_'+str(nu)] = np.max(abs(fitted['means']-known['means']))
    for variant in [False, True]:
        bb = branch.copy()
        bu = bus.copy()
        if variant:
            bb[0, 8], bb[0, 9], bb[1, 4], bb[2, 10] = 1.04, 9, .03, 0
            bu[3, 4] = .03
        v = (1+.03*rng.normal(size=(100, 33))) * np.exp(.1j*rng.normal(size=(100, 33)))
        yy, _, _ = admittance(base, bu, bb)
        out1, out2 = evaluate_outputs(v, base, bu, bb), independent_outputs(v, base, bb)
        err = np.max(abs(out1-out2))
        checks['output_formula_max_error_modified_'+str(variant)] = err
        assert err < 1e-10
        loss = base*np.sum((v*np.conj(v @ yy.T)).real, axis=1)-np.sum(abs(v)**2*bu[:, 4], axis=1)
        checks['loss_ybus_quadratic_error_modified_'+str(variant)] = max(abs(loss-out1[:, -1]))
        assert max(abs(loss-out1[:, -1])) < 1e-10
    y, _, _ = admittance(base, bus, branch)
    target, _ = generate('A_light', 128, 20261008+11, source)
    mapped, audit = ptov(target, y)
    ac = np.array([polar_ac(t, y)[0] for t in target])
    check_v = np.column_stack((mapped[:, :32], mapped[:, 32:]))
    ac_x = np.column_stack((ac[:, 1:].real, ac[:, 1:].imag))
    checks['ptov_polar_voltage_max_error'] = np.max(abs(check_v-ac_x))
    checks['ptov_audit'] = audit
    assert checks['ptov_polar_voltage_max_error'] < 1e-7
    write_json(OUT / 'implementation_checks.json', checks)
    return checks
