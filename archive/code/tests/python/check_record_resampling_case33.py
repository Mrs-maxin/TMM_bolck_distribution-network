"""Check record-based PV deviation scenarios (scenario_lock_20261024.md section 4).

Each scenario takes the 8-site forecast-error vector of one recorded 15-min interval, adds h * L eps
(L L^T = record covariance), adds it to the 50% forecast and clips to [0, observed max output].
Reports the record statistics of the resulting deviations for several h, and the number of intervals
available for the extreme periods of S2.
"""
import json
from pathlib import Path

import numpy as np

import fit_nrel_pv_forecast_error_case33 as one
import fit_nrel_two_region_case33 as two

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'result'/'case33bw'/'record_resampling_check_20261024'
H = [0.0, 0.05, 0.1, 0.2]
FORECAST = 0.5


def resample(e, upper, h, n, rng):
    L = np.linalg.cholesky(np.cov(e.T))
    pick = e[rng.integers(0, len(e), n)]+h*(rng.standard_normal((n, e.shape[1]))@L.T)
    return np.clip(FORECAST+pick, 0, upper)-FORECAST


def stats_row(e, ia=None, ib=None):
    row = one.describe(e, None)
    if ia is not None:
        row.update(two.cross_coamplification(e, ia, ib))
    return row


def extreme_pool():
    one.BAND = (0.3, 0.7)
    _, act, fc = one.load_records()
    e = act-fc; rms = np.sqrt(((e-e.mean(0))**2).mean(1))
    return {f'top{q}%': int(np.sum(rms >= np.quantile(rms, 1-q/100))) for q in [1, 2, 5]}, len(e)


def main():
    rng = np.random.default_rng(20261024); n = 200000; result = {}
    _, act, fc = one.load_records(); e1 = act-fc; up1 = act.max(0)
    rec1 = np.clip(FORECAST+e1, 0, up1)-FORECAST
    result['S1'] = {'records (shifted to 50% forecast)': stats_row(rec1), **{f'h={h}': stats_row(resample(e1, up1, h, n, rng)) for h in H}}
    sites, act3, fc3 = two.load_records(); e3 = act3-fc3; up3 = act3.max(0)
    ia = np.array([i for i, s in enumerate(sites) if s['region'] == 'A']); ib = np.array([i for i, s in enumerate(sites) if s['region'] == 'B'])
    rec3 = np.clip(FORECAST+e3, 0, up3)-FORECAST
    result['S3'] = {'records (shifted to 50% forecast)': stats_row(rec3, ia, ib), **{f'h={h}': stats_row(resample(e3, up3, h, n, rng), ia, ib) for h in H}}
    pool, total = extreme_pool(); result['S2_extreme_pool'] = dict(intervals_30_70=total, **pool)
    for scene in ['S1', 'S3']:
        rows = result[scene]; keys = list(next(iter(rows.values())))
        print(scene); print('quantity'.ljust(16)+''.join(k[:14].rjust(15) for k in rows))
        for k in keys:
            print(k.ljust(16)+''.join(f'{rows[r][k]:15.3f}' for r in rows))
    print('S2 extreme pool', result['S2_extreme_pool'])
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'record_resampling_check.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
