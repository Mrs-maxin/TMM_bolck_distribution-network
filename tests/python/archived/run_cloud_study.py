"""Reproducible cloud entry: prepare, fit, propagate, summarize.

Run with PYTHONPATH pointing to the task's cloud_deps and one BLAS thread.
All intermediate candidates remain available; no source repository writes.
"""
import argparse
import csv
import json
import time
import traceback
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.stats import binom, wasserstein_distance, skew, kurtosis
from cloud_numeric_core import (ROOT, OUT, REF, FLOOR, QUANTILES, PROP_SEEDS,
    sha, write_json, environment, read_sources, admittance, generate, ptov,
    polar_ac, evaluate_outputs, log_density, initialize, fit, propagate,
    floor_matrix, implementation_checks)


def append_row(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def prepare(scene):
    directory = OUT/scene
    directory.mkdir(exist_ok=True)
    if (directory/'prepared.json').exists():
        return
    source = read_sources()
    base, bus, branch = source[:3]
    y, _, _ = admittance(base, bus, branch)
    count_val = 4096 if scene.startswith('A_') else 8192
    count_ac = 32768 if scene.startswith('A_') else 65536
    counts = [8192, count_val, count_ac]
    names = ['training', 'validation', 'test']
    audits, times = {}, {}
    states = []
    for name, count, offset in zip(names, counts, [11, 104729, 209759]):
        start = time.perf_counter()
        target, metadata = generate(scene, count, 20261008+offset, source)
        times[name+'_input_seconds'] = time.perf_counter()-start
        np.savez_compressed(directory/(name+'_inputs.npz'), target=target, **metadata)
        if name != 'test':
            state, audit = ptov(target, y)
            states.append(state)
            audits[name] = audit
        else:
            start = time.perf_counter()
            voltages = np.empty((count, 33), complex)
            iterations = np.empty(count, int)
            for i, t in enumerate(target):
                voltages[i], iterations[i] = polar_ac(t, y)
                if (i+1) % 4096 == 0:
                    print(f'AC {scene} [{i+1}/{count}]', flush=True)
            ac_seconds = time.perf_counter()-start
            output = evaluate_outputs(voltages, base, bus, branch)
            np.savez_compressed(directory/'ac_reference.npz', voltage=voltages,
                                output=output, iterations=iterations)
            audits['test'] = dict(count=count, ac_seconds=ac_seconds,
                                  maximum_iterations=int(max(iterations)))
    center = states[0].mean(axis=0)
    _, sv, vt = np.linalg.svd(states[0]-center, full_matrices=False)
    rank_tol = max(states[0].shape)*np.spacing(sv[0])
    rank = int(sum(sv > rank_tol))
    assert rank == 64
    basis = vt.T
    scale = sv/np.sqrt(counts[0]-1)
    x = ((states[0]-center) @ basis)/scale
    validation = ((states[1]-center) @ basis)/scale
    np.savez_compressed(directory/'fitting_inputs.npz', x=x, validation=validation,
                        center=center, basis=basis, scale=scale,
                        training_voltage=states[0], validation_voltage=states[1],
                        singular_values=sv)
    # Independent AC spot check on the exact input rows used by the mapper.
    sample = np.load(directory/'training_inputs.npz')['target'][:128]
    independent = np.array([polar_ac(t, y)[0] for t in sample])
    independent_state = np.column_stack((independent[:, 1:].real, independent[:, 1:].imag))
    error = np.max(abs(independent_state-states[0][:128]))
    assert error < 1e-7
    write_json(directory/'prepared.json', dict(scene=scene, counts=counts,
        full_voltage_dimension=rank, numerical_rank_tolerance=rank_tol,
        maximum_reconstruction_error=np.max(abs((x*scale) @ basis.T+center-states[0])),
        independent_ac_spot_voltage_error=error, audits=audits, times=times,
        fitting_input_sha256=sha(directory/'fitting_inputs.npz'),
        old_model_replay=False, rng='NumPy MT19937; no MATLAB stream replay'))
    print('PREPARED', scene, 'rank', rank, 'AC spot error', error, flush=True)


def candidate_file(directory, k, partition, route, nu, acceleration):
    return directory/f'K{k}_p{partition+1:02d}_{route}_df{nu}_{acceleration}'


def save_fit(prefix, model, audit, validation, metadata):
    validation_ll = log_density(validation, model)[0].mean()
    temporary = str(prefix)+'.'+str(os.getpid())+'.tmp.npz'
    np.savez_compressed(temporary, **model, history=audit['history'],
                        floor_history=audit['floor_history'])
    os.replace(temporary, str(prefix)+'.npz')
    audit = {k: v for k, v in audit.items() if k not in ['history', 'floor_history']}
    audit.update(metadata)
    audit['validation_ll'] = validation_ll
    audit['model_sha256'] = sha(str(prefix)+'.npz')
    audit['covariance_eigen_min'] = np.asarray(audit['scatter_eigen_min'])*np.where(
        np.isfinite(model['df']), np.divide(model['df'], model['df']-2,
            out=np.ones_like(model['df']), where=np.isfinite(model['df'])), 1)
    audit['df_lower_boundary'] = bool(np.any(np.isfinite(model['df']) & (model['df'] <= 4.1*(1+1e-8))))
    audit['df_upper_boundary'] = bool(np.any(np.isfinite(model['df']) & (model['df'] >= 1e5*(1-1e-8))))
    write_json(str(prefix)+'.json', audit)
    return audit


def fit_partition(scene, k, partition):
    directory = OUT/scene/'candidates'
    directory.mkdir(exist_ok=True)
    write_json(directory/f'worker_K{k}_p{partition+1:02d}.state.json',
               dict(pid=os.getpid(), parent_pid=os.getppid(), scene=scene,
                    K=k, partition=partition+1, status='fitting'))
    data = np.load(OUT/scene/'fitting_inputs.npz')
    x, validation, scale = data['x'], data['validation'], data['scale']
    initial, labels, metadata = initialize(x, k, partition, scale)
    np.savez_compressed(directory/f'K{k}_p{partition+1:02d}_initial.npz',
                        **initial, labels=labels)
    records = []
    # GMM has no Gamma expansion; EM and PX coincide exactly.
    prefix = candidate_file(directory, k, partition, 'gaussian', 'inf', 'EM')
    if Path(str(prefix)+'.json').exists():
        gmm = {name: np.load(str(prefix)+'.npz')[name] for name in initial}
        gaudit = json.loads(Path(str(prefix)+'.json').read_text())
    else:
        gmm, gaudit = fit(x, initial)
        gaudit = save_fit(prefix, gmm, gaudit, validation,
                         dict(scene=scene, K=k, family='GMM', route='independent',
                              initial_df='inf', acceleration='EM=PX', **metadata))
    records.append(gaudit)
    for route in ['independent', 'gmm_warm']:
        for nu in [5, 12, 30, 200]:
            base_model = initial if route == 'independent' else gmm
            start_model = {name: np.array(base_model[name], copy=True) for name in initial}
            start_model['df'][:] = nu
            # Same actual initial covariance; the historical objective floors scatter.
            start_model['scatter'] *= (nu-2)/nu
            for j in range(k):
                start_model['scatter'][j] = floor_matrix(start_model['scatter'][j])[0]
            for acceleration in ['EM', 'PX']:
                prefix = candidate_file(directory, k, partition, route, nu, acceleration)
                if Path(str(prefix)+'.json').exists():
                    records.append(json.loads(Path(str(prefix)+'.json').read_text()))
                    continue
                try:
                    model, audit = fit(x, start_model, acceleration)
                    record = save_fit(prefix, model, audit, validation,
                        dict(scene=scene, K=k, family='TMM', route=route, initial_df=nu,
                             acceleration=acceleration, **metadata))
                except Exception:
                    record = dict(scene=scene, K=k, family='TMM', route=route,
                                  initial_df=nu, acceleration=acceleration, **metadata,
                                  status='exception', error=traceback.format_exc(),
                                  seconds=None)
                    write_json(str(prefix)+'.json', record)
                records.append(record)
    write_json(directory/f'worker_K{k}_p{partition+1:02d}.state.json',
               dict(pid=os.getpid(), parent_pid=os.getppid(), scene=scene,
                    K=k, partition=partition+1, status='completed', attempts=len(records)))
    return scene, k, partition+1, len(records)


def collect_fits(scene):
    directory = OUT/scene
    records, trajectories, components = [], [], []
    for file in sorted((directory/'candidates').glob('*.json')):
        if file.name.endswith('.state.json'):
            continue
        record = json.loads(file.read_text())
        record['candidate'] = file.stem
        if 'model_sha256' in record:
            data = np.load(file.with_suffix('.npz'))
            for it, value in enumerate(data['history']):
                trajectories.append(dict(candidate=file.stem, iteration=it, train_ll=value))
            for j, nu in enumerate(data['df']):
                components.append(dict(candidate=file.stem, component=j+1,
                    weight=data['weights'][j], df=nu,
                    effective_count=record['effective_counts'][j],
                    scatter_eigen_min=record['scatter_eigen_min'][j],
                    covariance_eigen_min=record['covariance_eigen_min'][j],
                    scatter_condition=record['scatter_condition'][j]))
        records.append({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                        for k, v in record.items()})
    pd.DataFrame(records).to_csv(directory/'fit_attempts.csv', index=False)
    pd.DataFrame(trajectories).to_csv(directory/'fit_trajectories.csv', index=False)
    pd.DataFrame(components).to_csv(directory/'fit_components.csv', index=False)
    return pd.DataFrame(records)


def run_fits(scenes, workers=6, partitions=20):
    jobs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for scene in scenes:
            for k in [1, 3, 6]:
                # K=1 initial partitions all give the same model; record aliases later.
                for partition in range(1 if k == 1 else partitions):
                    jobs.append(pool.submit(fit_partition, scene, k, partition))
        for index, job in enumerate(as_completed(jobs), 1):
            result = job.result()
            print('FIT_PROGRESS', index, '/', len(jobs), result, flush=True)
            append_row(OUT/'fit_progress_log.csv', dict(timestamp=time.time(),
                parent_pid=os.getpid(), completed_tasks=index, total_tasks=len(jobs),
                scene=result[0], K=result[1], partition=result[2], attempts=result[3]))
    for scene in scenes:
        collect_fits(scene)


def refine_select(scene, ks=(1, 3, 6)):
    directory = OUT/scene
    data = np.load(directory/'fitting_inputs.npz')
    x, validation = data['x'], data['validation']
    table = collect_fits(scene)
    selection = []
    for k in ks:
        for family, acceleration in [('GMM', 'EM=PX'), ('TMM', 'EM'), ('TMM', 'PX')]:
            group = table[(table.K == k) & (table.family == family) & (table.acceleration == acceleration)]
            group = group[group.route != 'continuation']
            group = group.dropna(subset=['validation_ll']).sort_values('validation_ll', ascending=False)
            assert len(group), (scene, k, family, acceleration)
            for budget in [10, 20]:
                eligible = group[group.partition <= budget]
                chosen = eligible.iloc[0]
                prefix = directory/'candidates'/str(chosen.candidate)
                initial = {name: np.load(str(prefix)+'.npz')[name]
                           for name in ['weights', 'means', 'scatter', 'df']}
                new_prefix = directory/'candidates'/(str(chosen.candidate)+f'_refined_budget{budget}')
                if not Path(str(new_prefix)+'.json').exists():
                    model, audit = fit(x, initial, 'PX' if acceleration == 'PX' else 'EM', maximum=900, tolerance=1e-8)
                    save_fit(new_prefix, model, audit, validation,
                        dict(scene=scene, K=k, family=family, acceleration=acceleration,
                             route='continuation', initial_df=chosen.initial_df,
                             partition=int(chosen.partition), parent_candidate=chosen.candidate,
                             budget=budget, seed=int(chosen.seed)))
                refined = json.loads(Path(str(new_prefix)+'.json').read_text())
                if refined['validation_ll'] > float(chosen.validation_ll):
                    name, score = new_prefix.name, refined['validation_ll']
                else:
                    name, score = chosen.candidate, float(chosen.validation_ll)
                selection.append(dict(scene=scene, K=k, family=family,
                    acceleration=acceleration, budget=budget, candidate=name,
                    validation_ll=score, original_candidate=chosen.candidate,
                    original_validation_ll=float(chosen.validation_ll),
                    original_status=chosen.status, continuation_status=refined['status'],
                    continuation_train_gain=refined['train_ll']-float(chosen.train_ll),
                    continuation_validation_gain=refined['validation_ll']-float(chosen.validation_ll)))
    pd.DataFrame(selection).to_csv(directory/'selection_stability.csv', index=False)
    collect_fits(scene)
    return pd.DataFrame(selection)


def weighted_sorted(values, weights):
    ix = np.argsort(values)
    return values[ix], np.cumsum(weights[ix])


def distribution_metrics(reference, predicted, weights):
    rs = np.sort(reference)
    ps, cw = weighted_sorted(predicted, weights)
    cw /= cw[-1]
    grid = np.unique(np.r_[rs, ps])
    ri, pi = np.searchsorted(rs, grid, side='right'), np.searchsorted(ps, grid, side='right')
    rf = ri/len(rs)
    pf = np.r_[0., cw][pi]
    diff = pf-rf
    ks = np.max(abs(diff))
    w1 = np.sum(abs(diff[:-1])*np.diff(grid))
    # Right-continuous empirical inverse CDF for both sample laws.
    rq = rs[np.clip(np.ceil(QUANTILES*len(rs)).astype(int)-1, 0, len(rs)-1)]
    pq = ps[np.clip(np.searchsorted(cw, QUANTILES), 0, len(ps)-1)]
    f_at_rq = np.r_[0., cw][np.searchsorted(ps, rq, side='right')]
    mean = np.dot(weights, predicted)
    variance = np.dot(weights, (predicted-mean)**2)
    rows = []
    for index, q in enumerate(QUANTILES):
        lo_index = int(np.clip(binom.ppf(.025, len(rs), q)-1, 0, len(rs)-1))
        hi_index = int(np.clip(binom.ppf(.975, len(rs), q), 0, len(rs)-1))
        rows.append(dict(q=q, reference_quantile=rq[index], model_quantile=pq[index],
            signed_error=pq[index]-rq[index], absolute_error=abs(pq[index]-rq[index]),
            reference_cdf_error=f_at_rq[index]-q,
            reference_quantile_ci95_lower=rs[lo_index], reference_quantile_ci95_upper=rs[hi_index],
            reference_cdf_pointwise_se=np.sqrt(q*(1-q)/len(rs))))
    global_row = dict(ks=ks, w1=w1, reference_mean=reference.mean(), model_mean=mean,
        signed_mean_error=mean-reference.mean(), reference_variance=reference.var(),
        model_variance=variance, variance_ratio=variance/reference.var())
    for side, mask in [('lower', rf <= .05), ('upper', rf >= .95)]:
        global_row[side+'_tail_cdf_sup_abs'] = np.max(abs(diff[mask]))
        global_row[side+'_tail_cdf_signed_mean'] = np.mean(diff[mask])
    # Quantile-indexed CDF grid preserves local detail in both tails.
    qq = np.unique(np.r_[np.linspace(0, 1, 1001), np.geomspace(1e-5, .05, 100),
                         1-np.geomspace(1e-5, .05, 100), QUANTILES])
    values = np.quantile(reference, qq)
    model_cdf = np.r_[0., cw][np.searchsorted(ps, values, side='right')]
    return global_row, rows, (values, model_cdf, np.searchsorted(rs, values, side='right')/len(rs))


def run_propagation(scene):
    directory = OUT/scene
    selection = pd.read_csv(directory/'selection_stability.csv')
    final = selection[selection.budget == 20].sort_values('validation_ll', ascending=False).drop_duplicates(['K', 'family'])
    final.to_csv(directory/'selected_models.csv', index=False)
    data = np.load(directory/'fitting_inputs.npz')
    source = read_sources()
    base, bus, branch = source[:3]
    reference = np.load(directory/'ac_reference.npz')['output']
    names = ['P']*32+['Q']*32+['loss']
    objects = list(range(1, 33))*2+[0]
    propagation_dir = directory/'propagation'
    propagation_dir.mkdir(exist_ok=True)
    for _, selected in final.iterrows():
        k, family = int(selected.K), selected.family
        model = {name: np.load(directory/'candidates'/(selected.candidate+'.npz'))[name]
                 for name in ['weights', 'means', 'scatter', 'df']}
        for n in ([16384, 65536] if scene.startswith('A_') else [65536]):
            for method in ['pseudo', 'sobol']:
                for repeat, seed in enumerate(PROP_SEEDS, 1):
                    stem = f'{family}_K{k}_n{n}_{method}_r{repeat}'
                    if (propagation_dir/(stem+'_global.csv')).exists():
                        continue
                    start = time.perf_counter()
                    z, weights = propagate(model, n, seed, method)
                    state = (z*data['scale']) @ data['basis'].T+data['center']
                    voltage = np.column_stack((np.ones(n), state[:, :32]+1j*state[:, 32:]))
                    output = evaluate_outputs(voltage, base, bus, branch)
                    propagation_seconds = time.perf_counter()-start
                    import hashlib
                    write_json(propagation_dir/(stem+'_sample_hashes.json'), dict(
                        output_float64_c_order_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
                        weights_float64_c_order_sha256=hashlib.sha256(weights.tobytes()).hexdigest(),
                        shape=output.shape,
                        reconstruction='Regenerate from selected candidate, transform, n, method, seed'))
                    global_rows, quantile_rows, cdf_grids = [], [], []
                    start = time.perf_counter()
                    for j in range(65):
                        meta = dict(scene=scene, family=family, K=k, n=n,
                            method=method, repeat=repeat, seed=seed, output=names[j],
                            branch=objects[j], candidate=selected.candidate)
                        global_row, rows, cdf = distribution_metrics(reference[:, j], output[:, j], weights)
                        global_rows.append({**meta, **global_row})
                        quantile_rows += [{**meta, **r} for r in rows]
                        cdf_grids.append(cdf)
                    pd.DataFrame(global_rows).to_csv(propagation_dir/(stem+'_global.csv'), index=False)
                    pd.DataFrame(quantile_rows).to_csv(propagation_dir/(stem+'_quantiles.csv'), index=False)
                    np.savez_compressed(propagation_dir/(stem+'_cdf.npz'), grids=np.array(cdf_grids))
                    append_row(directory/'cost_propagation.csv', dict(stem=stem,
                        propagation_seconds=propagation_seconds, processing_seconds=time.perf_counter()-start))
                    print('PROP', scene, stem, 'seconds', round(propagation_seconds, 3), flush=True)
    pd.concat([pd.read_csv(f) for f in propagation_dir.glob('*_global.csv')]).to_csv(directory/'global_metrics.csv', index=False)
    pd.concat([pd.read_csv(f) for f in propagation_dir.glob('*_quantiles.csv')]).to_csv(directory/'quantile_metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['check', 'prepare', 'fit', 'select', 'propagate'])
    parser.add_argument('--scenes', nargs='+', default=['A_light'])
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--partitions', type=int, default=20)
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 3, 6])
    args = parser.parse_args()
    write_json(OUT/f'process_{os.getpid()}.json', dict(pid=os.getpid(),
               parent_pid=os.getppid(), stage=args.stage, scenes=args.scenes,
               command=' '.join(__import__('sys').argv), started_unix=time.time()))
    write_json(OUT/'environment.json', environment())
    if args.stage == 'check':
        implementation_checks(read_sources())
    elif args.stage == 'prepare':
        for scene in args.scenes:
            prepare(scene)
    elif args.stage == 'fit':
        run_fits(args.scenes, args.workers, args.partitions)
    elif args.stage == 'select':
        for scene in args.scenes:
            refine_select(scene, args.ks)
    elif args.stage == 'propagate':
        for scene in args.scenes:
            run_propagation(scene)


if __name__ == '__main__':
    main()
