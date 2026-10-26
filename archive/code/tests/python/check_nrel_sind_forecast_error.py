"""Check PV forecast-error statistics in the NREL Solar Power Data for Integration Studies.

Reads data/nrel_sind/<state>/ (Actual 5-min output, DA and HA4 hourly forecasts),
selects the distributed-PV sites nearest to a chosen centre, and reports, for daytime
hours grouped by forecast level, the per-site error kurtosis, the inter-site error
correlation and whether large errors occur at several sites in the same hour.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATE = os.environ.get('SIND_STATE', 'co')
CENTRE = tuple(float(v) for v in os.environ.get('SIND_CENTRE', '39.75,-104.95').split(','))
N_SITES = int(os.environ.get('SIND_NSITES', '8'))
PV_TYPE = os.environ.get('SIND_TYPE', 'DPV')
DATA = ROOT/'data'/'nrel_sind'/STATE


def site_table():
    rows = []
    for f in os.listdir(DATA):
        if not f.startswith('Actual_'):
            continue
        p = f.split('_')
        rows.append(dict(lat=float(p[1]), lon=float(p[2]), typ=p[4], cap=float(p[5][:-2]), key='_'.join(p[1:6])))
    return rows


def distance_km(lat1, lon1, lat2, lon2):
    a, b, c, d = map(np.radians, (lat1, lon1, lat2, lon2))
    return 6371*np.arccos(np.clip(np.sin(a)*np.sin(c)+np.cos(a)*np.cos(c)*np.cos(b-d), -1, 1))


def read_series(prefix, key, minutes):
    f = DATA/f'{prefix}_{key}_{minutes}_Min.csv'
    return np.loadtxt(f, delimiter=',', skiprows=1, usecols=1)


def excess_kurtosis(x):
    x = x-x.mean()
    return float(np.mean(x**4)/np.mean(x**2)**2-3)


def main():
    sites = [s for s in site_table() if s['typ'] == PV_TYPE]
    for s in sites:
        s['dist'] = distance_km(CENTRE[0], CENTRE[1], s['lat'], s['lon'])
    sites = sorted(sites, key=lambda s: s['dist'])[:N_SITES]
    print('sites', [(s['lat'], s['lon'], s['cap'], round(s['dist'], 1)) for s in sites])
    lat = np.array([s['lat'] for s in sites]); lon = np.array([s['lon'] for s in sites])
    dmat = distance_km(lat[:, None], lon[:, None], lat[None], lon[None])
    print('max inter-site distance km', round(float(dmat.max()), 1))

    cap = np.array([s['cap'] for s in sites])
    actual5 = np.stack([read_series('Actual', s['key'], 5) for s in sites], 1)/cap
    hours = actual5.shape[0]//12
    actual_hour = actual5[:hours*12].reshape(hours, 12, -1).mean(1)
    for label in ['DA', 'HA4']:
        fc = np.stack([read_series(label, s['key'], 60) for s in sites], 1)[:hours]/cap
        err_hour = actual_hour-fc
        err5 = actual5[:hours*12].reshape(hours, 12, -1)-fc[:, None, :]
        level = fc.mean(1)
        print(f'\n=== {label} forecast; daytime hours with mean forecast in band')
        for lo, hi in [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.4, 0.6), (0.7, 1.0)]:
            m = (level >= lo) & (level < hi)
            if m.sum() < 30:
                print(f'band {lo:.1f}-{hi:.1f}: {m.sum()} hours, skipped'); continue
            for name, e in [('hourly', err_hour[m]), ('5-min', err5[m].reshape(-1, len(sites)))]:
                rms = np.sqrt(np.mean(e**2, 0))
                kurt = [excess_kurtosis(e[:, i]) for i in range(e.shape[1])]
                c = np.corrcoef(e.T); iu = np.triu_indices(len(sites), 1)
                c2 = np.corrcoef((e-e.mean(0)).T**2)
                mean_err = e.mean(1)
                print(f'band {lo:.1f}-{hi:.1f} {name:6s} records {e.shape[0]:6d} | RMS/cap {rms.mean():.3f} | '
                      f'site excess kurtosis median {np.median(kurt):5.2f} range {min(kurt):5.2f}..{max(kurt):5.2f} | '
                      f'error corr {c[iu].mean():.2f} | squared-error corr {c2[iu].mean():.2f} vs Gaussian {np.mean(c[iu]**2):.2f} | '
                      f'site-mean error kurtosis {excess_kurtosis(mean_err):5.2f}')
        m = (level >= 0.4) & (level < 0.6)
        e = err_hour[m]
        spread = np.sqrt(np.mean((e-e.mean(0))**2, 1))
        q = np.quantile(spread, [0.1, 0.5, 0.9, 0.99])
        print('hour-by-hour cross-site RMS error (band 0.4-0.6) quantiles 10/50/90/99%:', np.round(q, 3),
              '| ratio 90%/10%', round(q[2]/q[0], 1))
        big = np.abs(e-e.mean(0)) > 2*e.std(0)
        k = big.sum(1)
        print('hours with |error|>2 sd at >=1, >=half, all sites:', int((k >= 1).sum()), int((k >= len(sites)//2).sum()),
              int((k == len(sites)).sum()), 'of', len(k))


if __name__ == '__main__':
    sys.exit(main())
