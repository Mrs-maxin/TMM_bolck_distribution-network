"""Case 1 summary: proposed method against the tail-oriented and GMM comparison methods.

Reads the outputs of run_pv_nrel_case33.py, analytic_limit_risk_pv_nrel_case33.py and
tail_baselines_pv_nrel_case33.py. Writes, under <out>/nrel_ha4_40_60/case1_summary/:
  limit_probabilities.csv   locked events: AC, cross-entropy importance sampling, every method at its selected K
  tail_probability.csv      model probability beyond the AC 1%/0.5%/0.1% quantiles of the 8 target objects
  other_objects_ks.csv      KS of the non-target objects
  computation.csv           AC load flows and seconds
"""
import os, sys, json, time
from pathlib import Path
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
O = Path(os.environ.setdefault('PV_RISK_DIR', str(REPO/'result/case33bw/pv_nrel_case33_20261022')))
os.environ.setdefault('PV_RISK_SCENE', 'nrel_ha4_40_60')
import numpy as np
import pandas as pd
from scipy import stats
import analytic_limit_risk_pv_fixed_forecast as risk

D = risk.D; TB = D/'tail_baselines'; OUT = D/'case1_summary'; OUT.mkdir(exist_ok=True)
LABEL = {'SBTMM': '本文方法', 'GMM': '前作GMM', 'Gao2024': 'Gao2024', 'Williams2012': 'Williams2012',
         'Mohammadi2017': 'Mohammadi2017', 'LeiteDaSilva2018': 'LeiteDaSilva2018'}
LEVELS = [0.01, 0.005, 0.001]


def main():
    events = json.loads((D/'limit_events_locked.json').read_text())['events']
    best = json.loads((D/'compare_summary.json').read_text())['best_K_by_validation']
    gao_k = json.loads((TB/'gao_protocol.json').read_text())['best_K_by_validation']
    ref = np.load(D/'reference_outputs.npz')['values']; n_ref = len(ref)
    risk_table = pd.read_csv(D/'analytic_limit_risk.csv')
    gao = pd.read_csv(TB/'gao_limits.csv'); gao = gao[gao.K == gao_k]
    lin = pd.read_csv(TB/'analytic_baseline_limits.csv')
    rows = []
    for i, e in enumerate(events):
        j = e['object_index']; hit = ref[:, j] < e['value'] if e['op'] == '<' else ref[:, j] > e['value']
        ce = json.loads((TB/f'ceis_event{i:02d}.json').read_text())
        row = dict(event=e['name'], AC=float(hit.mean()), AC_count=int(hit.sum()),
                   AC_95_low=float(stats.beta.ppf(.025, hit.sum(), n_ref-hit.sum()+1)) if hit.sum() else 0.,
                   AC_95_high=float(stats.beta.ppf(.975, hit.sum()+1, n_ref-hit.sum())),
                   LeiteDaSilva2018=ce['probability'], LeiteDaSilva2018_cv=ce['coefficient_of_variation'])
        for fam in ['SBTMM', 'GMM']:
            row[fam] = float(risk_table[(risk_table.family == fam) & (risk_table.K.astype(str) == str(best[fam]))].iloc[0][e['name']])
        row['Gao2024'] = float(gao[gao.event == e['name']].probability.iloc[0])
        for m in ['Williams2012', 'Mohammadi2017']:
            row[m] = float(lin[(lin.method == m) & (lin.event == e['name'])].probability.iloc[0])
        for m in ['SBTMM', 'GMM', 'Gao2024', 'Williams2012', 'Mohammadi2017']:
            row[m+'_rel_err_vs_CEIS'] = (row[m]-row['LeiteDaSilva2018'])/row['LeiteDaSilva2018']
        rows.append(row)
    limits = pd.DataFrame(rows); limits.to_csv(OUT/'limit_probabilities.csv', index=False)

    targets = pd.read_csv(D/'targets.csv'); selected = pd.concat([pd.read_csv(D/f) for f in ['selected_models.csv', 'selected_models_shared.csv']])
    cdfs = np.load(TB/'analytic_baseline_cdfs.npz'); tail_rows = []
    models = {f: dict(np.load(D/'candidates'/(selected[(selected.family == f) & (selected.K == best[f])].candidate.iloc[0]+'.npz')))
              for f in ['SBTMM', 'GMM']}
    g = np.load(TB/f'gao_K{gao_k}.npz')
    for _, t in targets.iterrows():
        j = int(t.object_index); r = np.sort(ref[:, j]); grid = np.linspace(r[0]-3*r.std(), r[-1]+3*r.std(), 1601)
        for lv in LEVELS:
            x = np.quantile(ref[:, j], lv if t.side == 'lower' else 1-lv)
            def beyond(F):
                return float(F if t.side == 'lower' else 1-F)
            row = dict(object=f"{t.output}{int(t.node) if t.output == 'V' else int(t.branch)}", side=t.side, level=lv, AC_value=float(x))
            for fam, model in models.items():
                th = x**2 if j < 32 else x
                F, _ = risk.cdf_at(risk.component_terms(model, 'BTMM' if fam == 'SBTMM' else 'GMM', j), [th]); row[fam] = beyond(F[0])
            row['Gao2024'] = beyond(g['weights']@stats.norm.cdf((x-g['means'][:, j])/g['sds'][:, j]))
            for m in ['Williams2012', 'Mohammadi2017']:
                row[m] = beyond(np.interp(x, grid, cdfs[f'{m}_{j}']))
            for m in ['SBTMM', 'GMM', 'Gao2024', 'Williams2012', 'Mohammadi2017']:
                row[m+'_rel_err'] = (row[m]-lv)/lv
            tail_rows.append(row)
    tails = pd.DataFrame(tail_rows); tails.to_csv(OUT/'tail_probability.csv', index=False)

    gs = pd.concat([pd.read_csv(O/'global_repeat_summary.csv'), pd.read_csv(O/'shared_global_repeat_summary.csv')])
    target_ids = set(targets.object_index.astype(int)); ks_rows = []
    gao_obj = pd.read_csv(TB/'gao_objects.csv'); gao_obj = gao_obj[gao_obj.K == gao_k]; lin_obj = pd.read_csv(TB/'analytic_baseline_objects.csv')
    for j, obj in enumerate(risk.run.OBJECTS):
        if j in target_ids:
            continue
        row = dict(object_index=j, output=obj['output'], node=obj['node'], branch=obj['branch'])
        for fam in ['SBTMM', 'GMM']:
            m = gs[(gs.family == fam) & (gs.K == best[fam]) & (gs.output == obj['output']) & (gs.node == obj['node']) & (gs.branch == obj['branch'])]
            row[fam] = float(m.ks_mean.iloc[0])
        row['Gao2024'] = float(gao_obj[gao_obj.object_index == j].ks.iloc[0])
        for m in ['Williams2012', 'Mohammadi2017']:
            row[m] = float(lin_obj[(lin_obj.method == m) & (lin_obj.object_index == j)].ks.iloc[0])
        ks_rows.append(row)
    ks = pd.DataFrame(ks_rows); ks.to_csv(OUT/'other_objects_ks.csv', index=False)

    fits = pd.read_csv(D/'all_fit_attempts_shared.csv'); gfits = pd.read_csv(D/'all_fit_attempts.csv')
    clock = time.perf_counter()
    for e in events:
        th = e['value']**2 if e['object_index'] < 32 else e['value']
        risk.cdf_at(risk.component_terms(models['SBTMM'], 'BTMM', e['object_index']), [th])
    analytic_seconds = (time.perf_counter()-clock)/len(events)
    ac_sec = sum(json.loads((D/f'{s}_AC_summary.json').read_text())['seconds'] for s in ['training', 'validation'])
    ce = [json.loads((TB/f'ceis_event{i:02d}.json').read_text()) for i in range(len(events))]
    comp = [dict(method='本文方法', AC_load_flows=12288, AC_seconds=ac_sec,
                 fitting_seconds_all_K=float(fits[fits.family == 'SBTMM'].seconds.sum()+gfits[gfits.family == 'GMM'].seconds.sum()),
                 per_event_seconds=analytic_seconds, note='one fit serves all 97 quantities and every limit'),
            dict(method='LeiteDaSilva2018', AC_load_flows=int(sum(c['AC_load_flows'] for c in ce)),
                 per_event_seconds=float(np.mean([c['seconds'] for c in ce])), note='one cross-entropy run per event')]
    pd.DataFrame(comp).to_csv(OUT/'computation.csv', index=False)
    print(limits[['event', 'AC', 'AC_count', 'LeiteDaSilva2018', 'SBTMM', 'GMM', 'Gao2024', 'Williams2012', 'Mohammadi2017']].to_string())
    err = limits[[c for c in limits if c.endswith('_rel_err_vs_CEIS')]].abs()
    deep = limits.event.index % 2 == 1
    print('\nmedian |relative error| vs CE-IS, 1% events / 0.1% events');
    print(pd.DataFrame({'1%': err[~deep].median(), '0.1%': err[deep].median(), 'max 0.1%': err[deep].max()}).round(3))
    print('\ntail probability median |relative error| by level');
    print(tails.groupby('level')[[c for c in tails if c.endswith('_rel_err')]].apply(lambda x: x.abs().median()).round(3))
    print('\nnon-target KS median / max'); print(ks[['SBTMM', 'GMM', 'Gao2024', 'Williams2012', 'Mohammadi2017']].agg(['median', 'max']).round(4))
    print('\nKS SBTMM <= 1.2 x GMM share', float(np.mean(ks.SBTMM <= 1.2*ks.GMM)))
    print(pd.DataFrame(comp).to_string())


if __name__ == '__main__':
    main()
