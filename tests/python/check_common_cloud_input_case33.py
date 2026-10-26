"""Common-cloud PV deviation input for case33 (all stations driven by one weather process): statistics only, no power flow.

Each scenario takes one recorded 15-min interval (Denver 8 sites, forecast 60%-80%) and uses its 8-site mean deviation
as the common cloud effect of the feeder, scaled so that the single-site deviation variance equals the record; every
site adds an independent Gaussian deviation carrying a share f of the single-site variance; output = clip(70% + e,
0, observed max output). Reports the deviation statistics for several f next to the records.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import fit_nrel_pv_forecast_error_case33 as one

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'result'/'case33bw'/'common_cloud_input_check_20261026'
FORECAST, BAND, SHARES = 0.7, (0.6, 0.8), [0.1, 0.2, 0.3]


def describe(dev):
    s = dev.mean(1); sd = s.std(); d = dev-dev.mean(0); big = (np.abs(d) > 2*d.std(0)).sum(1)
    return dict(single_site_kurtosis=float(np.median(stats.kurtosis(dev, 0))), single_site_skew=float(np.median(stats.skew(dev, 0))),
                mean_kurtosis=float(stats.kurtosis(s)), mean_skew=float(stats.skew(s)),
                mean_q001_sd=float((np.quantile(s, .001)-s.mean())/sd), mean_q999_sd=float((np.quantile(s, .999)-s.mean())/sd),
                site_corr=float(np.corrcoef(d.T)[np.triu_indices(8, 1)].mean()), sq_corr=float(np.corrcoef((d**2).T)[np.triu_indices(8, 1)].mean()),
                half_big=float((big >= 4).mean()), site_rms=float(np.sqrt((d**2).mean())))


def common_cloud(e, upper, share, n, rng):
    common = e.mean(1); var_site = e.var(0).mean()
    a = np.sqrt((1-share)*var_site/common.var()); local_sd = np.sqrt(share*var_site)
    pick = common[rng.integers(0, len(common), n)]
    dev = a*(pick-common.mean())[:, None]+e.mean()+local_sd*rng.standard_normal((n, 8))
    return np.clip(FORECAST+dev, 0, upper)-FORECAST, a, local_sd


def main():
    one.BAND = BAND; _, act, fc = one.load_records(); e = act-fc; upper = act.max(0)
    rows = {'records': describe(np.clip(FORECAST+e, 0, upper)-FORECAST)}; info = {}
    rng = np.random.default_rng(20261026)
    for f in SHARES:
        dev, a, sd = common_cloud(e, upper, f, 400000, rng); rows[f'local share {f:.0%}'] = describe(dev); info[f] = dict(scale=a, local_sd=sd)
    t = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); t.to_csv(OUT/'common_cloud_statistics.csv')
    (OUT/'common_cloud_parameters.json').write_text(json.dumps(dict(records=len(e), band=BAND, forecast=FORECAST, parameters=info), indent=1))
    pd.set_option('display.width', 200); print(f'records {len(e)}'); print(t.round(3).to_string())


if __name__ == '__main__':
    main()
