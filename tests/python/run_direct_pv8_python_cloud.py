"""Cloud-only Python diagnostic for direct PV output; see the registered plan.

Run from the project root in an OpenAI cloud environment:
  python tests/python/run_direct_pv8_python_cloud.py --confirm-openai-cloud
The switch is an operator declaration, not independent proof of cloud execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'result/case33bw/direct_pv8_python_cloud_20261026'
DATA = ROOT / 'data/nrel_sind/pv8_direct_2006/pv8_2006_1000_1400_pu.csv'
CAPACITY_MW = 0.60
KS = (1, 2, 4)
RNG_STATE = 2026102607


def write_json(path, value):
    def convert(v):
        if hasattr(v, 'tolist'):
            return v.tolist()
        return str(v)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure():
    # Legacy imports require these empty result parents; no cached results are read.
    (ROOT / 'result/case33bw/pv_nrel_case33_newsplit_20261022/nrel_ha4_40_60').mkdir(parents=True, exist_ok=True)
    os.environ['MAIN_SCENE'] = 'direct_pv8'
    os.environ['MAIN_SHIFT'] = '1'
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    sys.argv = sys.argv[:1]
    import numpy as np
    import pandas as pd
    from scipy import stats
    import scipy
    import mainline_comparison_case33 as mc
    import analytic_grouped_t_case33 as an
    import cloud_numeric_core as core
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from easyplot_py import publication_context, save_figure
    return np, pd, stats, scipy, mc, an, core, plt, publication_context, save_figure


def main():
    np, pd, stats, scipy, mc, an, core, plt, publication_context, save_figure = configure()
    base = mc.base
    figures = OUT / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    manifest = dict(status='running', execution_location='operator_declared_OpenAI_cloud',
                    host=platform.node(), platform=platform.platform(), cwd=str(ROOT),
                    python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                    capacity_MW_per_site=CAPACITY_MW, K=KS, random_state=RNG_STATE,
                    input_sha256=sha(DATA), entry_sha256=sha(Path(__file__)),
                    plan='01_IDEA/plans/direct_pv8_python_cloud_20261026.md',
                    claim_state='No result until cloud run completes', figure_audit=[])
    write_json(OUT / 'run_manifest.json', manifest)

    def stage(name, **values):
        manifest.update(stage=name, elapsed_seconds=time.perf_counter()-t0)
        manifest.update(values)
        write_json(OUT / 'run_manifest.json', manifest)
        print(json.dumps(dict(stage=name, seconds=manifest['elapsed_seconds']), ensure_ascii=False), flush=True)

    def export(fig, name, table, provenance):
        table.to_csv(figures / f'{name}.csv', index=False)
        fig.set_size_inches(100/25.4, 76/25.4)
        fig.tight_layout()
        for extension in ('png', 'svg'):
            save_figure(fig, figures / f'{name}.{extension}', dpi=220, overwrite=True)
        plt.close(fig)
        write_json(figures / f'{name}.json', dict(provenance, width_mm=100, height_mm=76,
                   font_pt=10, status='Python diagnostic', smoothing='none'))

    def histogram(values, name, xlabel, unit):
        density, edges = np.histogram(values, bins='fd', density=True)
        fig, ax = plt.subplots()
        ax.stairs(density, edges, fill=True, color='#D8DDE2', edgecolor='#566169', linewidth=.6)
        ax.set(xlabel=xlabel, ylabel=f'PDF (1/{unit})')
        export(fig, name, pd.DataFrame(dict(left=edges[:-1], right=edges[1:], pdf=density)),
               dict(source='all original source-load records', zero_share=float(np.mean(values == 0)),
                    records=len(values), unit=unit, histogram='Freedman-Diaconis'))

    frame = pd.read_csv(DATA)
    stamp = pd.to_datetime(frame.LocalTime, format='%m/%d/%y %H:%M')
    z = frame[[f'PV{i}_pu' for i in range(1, 9)]].to_numpy()
    assert z.shape == (17520, 8) and np.isfinite(z).all() and z.min() >= 0
    assert len(set(stamp)) == len(stamp)
    rng = np.random.default_rng(RNG_STATE)
    from_days = np.unique(stamp.dt.strftime('%Y-%m-%d'))
    order = rng.permutation(len(from_days))
    groups = {'estimation': from_days[order[:219]], 'selection': from_days[order[219:292]],
              'reference': from_days[order[292:]]}
    split = {key: np.flatnonzero(stamp.dt.strftime('%Y-%m-%d').isin(days)) for key, days in groups.items()}
    assert set(split['estimation']).isdisjoint(split['reference'])
    assert set(split['selection']).isdisjoint(split['reference'])
    write_json(OUT / 'date_split.json', dict(days=groups, row_indices=split))
    wp, wq = np.array(base.CONFIG['load_P_weights'], float), np.array(base.CONFIG['load_Q_weights'], float)
    wp *= .01 / np.linalg.norm(wp)
    wq *= .01 / np.linalg.norm(wq)
    lp_factor, lq_factor = np.ones((len(z), 32)), np.ones((len(z), 32))
    for width, rho, ap, aq, mapping in zip(
            (1, 4, 32), base.CONFIG['load_PQ_correlation'], wp, wq,
            (np.zeros(32, int), base.REGIONS, np.arange(32))):
        gp = rng.standard_normal((len(z), width))
        gq = rho*gp + np.sqrt(1-rho*rho)*rng.standard_normal(gp.shape)
        lp_factor += ap*gp[:, mapping]
        lq_factor += aq*gq[:, mapping]
    lp = .8*base.BUS[None, 1:, 2]*lp_factor
    lq = .8*base.BUS[None, 1:, 3]*lq_factor
    pv = CAPACITY_MW*z
    p = -lp
    p[:, base.SITES-2] += pv
    target = np.c_[p, -lq]/base.BASE
    np.savez_compressed(OUT / 'inputs.npz', target=target, PV_MW=pv, load_P_MW=lp,
                        load_Q_MVAr=lq, timestamps=frame.LocalTime.to_numpy(dtype=str))
    with publication_context(font_family='DejaVu Sans', base_size=10):
        for j in range(8):
            histogram(pv[:, j], f'PV{j+1}_active_output', f'PV{j+1} active output (MW)', 'MW')
        for j in range(32):
            histogram(p[:, j], f'bus{j+2}_net_P', f'Bus {j+2} net active injection (MW)', 'MW')
            histogram(-lq[:, j], f'bus{j+2}_net_Q', f'Bus {j+2} net reactive injection (MVAr)', 'MVAr')
        histogram(pv.sum(1), 'total_PV_output', 'Total PV active output (MW)', 'MW')
    stage('injection_pdfs_saved', input_rows=len(z), split_rows={k: len(v) for k, v in split.items()})

    iv = np.r_[split['estimation'], split['selection']]
    states, mapping = core.ptov(target[iv], base.Y)
    vfit = base.voltage(states)
    probes = np.linspace(0, len(iv)-1, 16, dtype=int)
    exact_probes = np.array([core.polar_ac(target[iv[i]], base.Y)[0] for i in probes])
    map_error = float(np.max(abs(vfit[probes]-exact_probes)))
    assert map_error < 1e-6, f'Python mapping error exceeds 1e-6 p.u.: {map_error}'
    tr, va = split['estimation'], split['selection']
    st, sv = states[:len(tr)], states[len(tr):]
    center = st.mean(0)
    _, singular, basis_t = np.linalg.svd(st-center, full_matrices=False)
    scale, basis = singular/np.sqrt(len(st)-1), basis_t.T
    assert scale.min() > 0, 'Voltage coordinates are rank deficient'
    x, xv = (st-center)@basis/scale, (sv-center)@basis/scale
    data = dict(x=x, xv=xv, train=target[tr], every=target[iv], center=center, basis=basis, scale=scale)
    vref = np.array([core.polar_ac(t, base.Y)[0] for t in target[split['reference']]])
    ref = base.outputs(vref)
    independent = core.independent_outputs(vref, base.BASE, base.BRANCH)
    assert np.max(abs(independent-ref[:, 32:])) < 1e-9
    np.savez_compressed(OUT / 'ac_results.npz', reference=ref, reference_voltage=vref,
                        fitting_states=states, fitting_row_indices=iv)
    stage('ac_complete', mapping=mapping, maximum_mapping_error_pu=map_error,
          minimum_reference_voltage_pu=float(abs(vref[:, 1:]).min()),
          maximum_reference_voltage_pu=float(abs(vref[:, 1:]).max()),
          reference_fraction_any_voltage_above_1p07=float(np.mean((abs(vref[:, 1:]) > 1.07).any(1))))

    rows = []
    for j, obj in enumerate(base.OBJECTS):
        values = ref[:, j]
        q01, q25, median, q75, q99 = np.quantile(values, [.01, .25, .5, .75, .99])
        sd = values.std()
        core_sd = max((q75-q25)/1.349, 1e-14)
        lower, upper = (median-q01)/core_sd, (q99-median)/core_sd
        left, right = (median-q01)/max(sd, 1e-14), (q99-median)/max(sd, 1e-14)
        kurt = float(stats.kurtosis(values))
        near_constant = sd < (1e-8 if obj['output'] == 'V' else 1e-7)
        heavy = not near_constant and kurt > .5 and max(lower, upper) > 3
        long_tail = not near_constant and max(left, right) > 2.326 and max(left, right)/max(min(left, right), 1e-14) >= 1.5
        rows.append(dict(index=j, object=mc.object_name(j), kind=obj['output'],
                         from_bus=int(base.BRANCH[obj['branch']-1, 0]) if obj['branch'] else 0,
                         to_bus=int(base.BRANCH[obj['branch']-1, 1]) if obj['branch'] else 0,
                         mean=float(values.mean()), sd=sd, skewness=float(stats.skew(values)),
                         excess_kurtosis=kurt, lower_core_extension=lower, upper_core_extension=upper,
                         q01=q01, median=median, q99=q99, heavier_side='lower' if lower >= upper else 'upper',
                         heavy_tail_indication=heavy, long_tail_indication=long_tail,
                         near_constant=near_constant, negative_fraction=float(np.mean(values < 0))))
    inventory = pd.DataFrame(rows)
    inventory.to_csv(OUT / 'tail_inventory_all97.csv', index=False)
    selected = []
    for kind in ('V', 'P', 'Q'):
        pool = [r for r in rows if r['kind'] == kind and not r['near_constant']]
        hits = [r for r in pool if r['heavy_tail_indication'] or r['long_tail_indication']]
        chosen = sorted(hits or pool, key=lambda r: -max(r['lower_core_extension'], r['upper_core_extension']))[:2 if hits else 1]
        selected.extend(chosen)
    selected += [rows[-1]]
    write_json(OUT / 'selected_objects_before_fit.json', selected)
    stage('tail_objects_locked', selected_objects=[r['object'] for r in selected])

    anchor = int(np.argmin(((target[tr]-target[tr].mean(0))**2).sum(1)))
    grouping, Q1, Q2 = mc.chk.split(target[tr], x, basis, scale, rng, 'direct_pv8',
                                   group_target=target[iv], return_q=True, v0=vfit[anchor])
    strength_caps = mc.ts.strength_cap(target[iv], grouping['groups'])
    weather_groups = [np.array(g) for g in grouping['group_cols']]
    write_json(OUT / 'voltage_groups.json', dict(grouping, strength_caps=strength_caps))
    forms = an.quadratic_forms(data)
    assert forms['check'] < 1e-8, 'Voltage quadratic-form check failed'
    ids = [r['index'] for r in selected]
    forms = dict(A=forms['A'][ids], b=forms['b'][ids], c=forms['c'][ids])
    audit, metrics = [], []
    styles = {'AQGMM': ('#356D9A', '--'), 'Standard TMM': ('#98671F', '-.'),
              'Grouped TMM': ('#AB4B53', '-')}
    for k in KS:
        best, fits = None, []
        for start in range(1 if k == 1 else 2):
            ini, _, rule = core.initialize(x, k, start, scale, population_covariance=True)
            model, info = core.fit(x, ini, actual_covariance_floor=True, relative_tolerance=True)
            if info['status'] == 'collapsed_component':
                raise RuntimeError(f'AQGMM component collapse at K={k}, start={start}')
            likelihood = float(core.log_density(xv, model)[0].mean())
            fits.append(dict(start=start, likelihood=likelihood, rule=rule, audit=info))
            if best is None or likelihood > best[0]:
                best = likelihood, model
        gmm = best[1]
        _, resp, _ = core.log_density(x, gmm)
        standard = dict(weights=resp.sum(0)/len(x), means=[], scatter=[], df=[])
        for j in range(k):
            w = resp[:, j]
            mu = w@x/w.sum()
            error = x-mu
            covariance = error.T@(error*w[:, None])/w.sum()
            delta = np.einsum('ij,ij->i', error@np.linalg.inv(covariance), error)
            nu = mc.mm.nu_from_kurtosis(delta, w, x.shape[1])
            standard['means'].append(mu)
            standard['scatter'].append(covariance*((nu-2)/nu if np.isfinite(nu) else 1))
            standard['df'].append(nu)
        standard = {key: np.asarray(value) for key, value in standard.items()}
        grouped = mc.sh.shift_grouped_model(x, gmm, Q1, Q2, weather_groups, strength_caps) if weather_groups else gmm
        models = [('AQGMM', 'GMM', gmm), ('Standard TMM', 'TMM', standard),
                  ('Grouped TMM', 'SHIFT' if weather_groups else 'GMM', grouped)]
        audit.append(dict(K=k, fits=fits, grouped_degenerates_to_gmm=not bool(weather_groups)))
        curves = {}
        for label, kind, model in models:
            terms = an.terms(model, kind, forms)
            curves[label] = []
            for pos, row in enumerate(selected):
                values = ref[:, row['index']]
                lo, hi = values.min(), values.max()
                pad = .05*max(hi-lo, 1e-8)
                grid = np.linspace(lo-pad, hi+pad, 401)
                if row['kind'] == 'V':
                    grid = grid[grid > 0]
                qgrid = grid**2 if row['kind'] == 'V' else grid
                pdf, cdf, integral_info = inverse_pdf(terms[pos], qgrid, np)
                if row['kind'] == 'V':
                    pdf *= 2*grid
                curves[label].append((grid, pdf, cdf))
                manifest['figure_audit'].append(dict(K=k, model=label, object=row['object'], **integral_info))
                levels = np.linspace(.01, .05, 25)
                quantiles = np.quantile(values, np.r_[levels, 1-levels])
                F, cf_end = an.cdf_at(terms[pos], quantiles**2 if row['kind'] == 'V' else quantiles)
                metrics.append(dict(K=k, model=label, object=row['object'],
                                    lower_tail_cdf_error=float(np.mean(abs(F[:25]-levels))),
                                    upper_tail_cdf_error=float(np.mean(abs(1-F[25:]-levels))),
                                    full_cdf_error_on_grid=float(np.max(abs(cdf-np.searchsorted(np.sort(values), grid, side='right')/len(values)))),
                                    cf_endpoint=cf_end))
        with publication_context(font_family='DejaVu Sans', base_size=10):
            for pos, row in enumerate(selected):
                values = ref[:, row['index']]
                density, edges = np.histogram(values, bins='fd', density=True)
                unit = 'p.u.' if row['kind'] == 'V' else 'MVAr' if row['kind'] == 'Q' else 'MW'
                xlabel = f"{row['object']} ({unit})"
                table = None
                for extent in ('full', 'lower', 'upper'):
                    fig, ax = plt.subplots()
                    ax.stairs(density, edges, fill=True, color='#D8DDE2', edgecolor='#566169', linewidth=.6, label='AC records')
                    for label in styles:
                        grid, pdf, cdf = curves[label][pos]
                        color, ls = styles[label]
                        ax.plot(grid, pdf, color=color, linestyle=ls, linewidth=1.1, label=label)
                        if table is None:
                            table = pd.DataFrame(dict(value=grid))
                        table[label+'_pdf'], table[label+'_cdf'] = pdf, cdf
                    if extent != 'full':
                        limits = np.quantile(values, [.01, .05] if extent == 'lower' else [.95, .99])
                        ax.set_xlim(*limits)
                        masks = [(g >= limits[0]) & (g <= limits[1]) for g, _, _ in (curves[l][pos] for l in styles)]
                        peak = max([float(np.max(curves[l][pos][1][mask])) for l, mask in zip(styles, masks) if mask.any()] + [0.])
                        bins_in = (edges[:-1] <= limits[1]) & (edges[1:] >= limits[0])
                        peak = max(peak, float(density[bins_in].max()) if bins_in.any() else 0.)
                        ax.set_ylim(0, 1.12*max(peak, 1e-8))
                    else:
                        ax.set_xlim(curves['AQGMM'][pos][0][[0, -1]])
                    ax.set(xlabel=xlabel, ylabel=f'PDF (1/{unit})')
                    ax.legend(fontsize=6.5, frameon=False)
                    export(fig, f"{row['object']}_K{k}_{extent}", table,
                           dict(object=row, K=k, extent=extent, reference='independent AC dates',
                                density='analytic Fourier inversion; reference histogram',
                                pdf_nonnegative_display='negative inversion values retained on full plot; tail ordinate starts at zero'))
                pd.DataFrame(dict(left=edges[:-1], right=edges[1:], pdf=density)).to_csv(
                    figures / f"{row['object']}_K{k}_AC_histogram.csv", index=False)
        stage(f'K{k}_complete')
        pd.DataFrame(metrics).to_csv(OUT / 'per_object_comparison.csv', index=False)
        write_json(OUT / 'fit_audit.json', audit)
    manifest['artifacts'] = [dict(path=str(p.relative_to(OUT)), sha256=sha(p))
                             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'run_manifest.json']
    stage('complete', status='complete', claim_state='Basic Python diagnostic; inspect results and rendered figures before paper use')


def inverse_pdf(terms, thresholds, np):
    """PDF and CDF from the existing conditional quadratic-form characteristic function."""
    from numpy.polynomial.legendre import leggauss
    w = np.array([q[0] for q in terms])
    m0 = np.array([q[1] for q in terms])
    lam = np.array([q[2] for q in terms])
    beta = np.array([q[3] for q in terms])
    mk = m0+lam.sum(1)
    vk = 2*(lam**2).sum(1)+(beta**2).sum(1)
    mean = w@mk
    sd = np.sqrt(max(w@(vk+mk**2)-mean**2, 1e-300))
    keep = w > 1e-14
    tmax = 12/np.sqrt(vk[keep]/sd**2).min()
    if not np.isfinite(tmax) or tmax > 2000:
        raise RuntimeError('PDF inversion requires more than the basic-test frequency budget; preserve partial results')
    nodes, weights = leggauss(64)
    edges = np.linspace(0, tmax, int(np.ceil(tmax*4))+2)
    tau = ((edges[1:]-edges[:-1])[:, None]*(nodes+1)/2+edges[:-1, None]).ravel()
    wt = ((edges[1:]-edges[:-1])[:, None]*weights/2).ravel()
    t = tau/sd
    phi = np.zeros(len(t), complex)
    for j in np.flatnonzero(keep):
        z = 1-2j*np.outer(t, lam[j])
        phi += w[j]*np.exp(1j*t*(m0[j]-mean)-.5*np.log(z).sum(1)
                           -(np.outer(t**2, beta[j]**2)/(2*z)).sum(1))
    pdf, cdf = [], []
    for points in np.array_split((np.asarray(thresholds)-mean)/sd, 16):
        value = np.exp(-1j*np.outer(points, tau))*phi
        pdf.extend(value.real@wt/(np.pi*sd))
        cdf.extend(.5-(value.imag/tau@wt)/np.pi)
    pdf, cdf = np.array(pdf), np.array(cdf)
    return pdf, cdf, dict(frequency_max=float(tmax), cf_endpoint=float(abs(phi[-1])),
                          minimum_pdf=float(pdf.min()), raw_cdf_min=float(cdf.min()), raw_cdf_max=float(cdf.max()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-openai-cloud', action='store_true')
    args = parser.parse_args()
    if not args.confirm_openai_cloud or platform.system() != 'Linux':
        parser.error('Numerical work is authorized only inside an OpenAI cloud Linux environment; local execution is disabled')
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'failure_traceback.txt').write_text(traceback.format_exc())
        path = OUT / 'run_manifest.json'
        manifest = json.loads(path.read_text()) if path.exists() else {}
        manifest.update(status='failed', error=traceback.format_exc())
        write_json(path, manifest)
        raise
