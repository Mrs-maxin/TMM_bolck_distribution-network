"""Tail heaviness of NREL PV forecast errors under record-based selection choices (no scaling of deviations).

Choices: time resolution of actual output (15-min mean or 5-min), forecast horizon (HA4 or DA), months,
and sites (8 nearest Denver sites or the 8 smallest-capacity Denver-area sites). Deviations are shifted to
the 50% forecast and clipped to [0, observed max output] as in the scenario rule.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

import fit_nrel_pv_forecast_error_case33 as one

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'result'/'case33bw'/'record_tail_options_20261024'


def sites(rule):
    out = []
    for f in sorted(one.DATA.iterdir()):
        if f.name.startswith('Actual_') and '_DPV_' in f.name:
            p = f.name.split('_'); s = dict(lat=float(p[1]), lon=float(p[2]), cap=float(p[5][:-2]), key='_'.join(p[1:6]))
            s['dist'] = float(one.distance_km(39.75, -104.95, s['lat'], s['lon'])); out.append(s)
    if rule == 'nearest':
        return sorted(out, key=lambda s: s['dist'])[:8]
    near = [s for s in out if s['dist'] < 60]
    return sorted(near, key=lambda s: (s['cap'], s['dist']))[:8]


def records(site_rule='nearest', resolution=15, forecast='HA4', months=None, band=(0.4, 0.6)):
    ss = sites(site_rule); cap = np.array([s['cap'] for s in ss])
    act = np.stack([np.loadtxt(one.DATA/f"Actual_{s['key']}_5_Min.csv", delimiter=',', skiprows=1, usecols=1) for s in ss], 1)/cap
    fc = np.stack([np.loadtxt(one.DATA/f"{forecast}_{s['key']}_60_Min.csv", delimiter=',', skiprows=1, usecols=1) for s in ss], 1)/cap
    stamp = np.loadtxt(one.DATA/f"Actual_{ss[0]['key']}_5_Min.csv", delimiter=',', skiprows=1, usecols=0, dtype=str)
    month = np.array([datetime.strptime(t, '%m/%d/%y %H:%M').month for t in stamp])
    if resolution == 15:
        n = len(act)//3; act = act[:n*3].reshape(n, 3, -1).mean(1); month = month[:n*3:3]; fc = np.repeat(fc, 4, 0)[:n]
    else:
        fc = np.repeat(fc, 12, 0)[:len(act)]
    level = fc.mean(1); keep = (level >= band[0]) & (level < band[1])
    if months is not None:
        keep &= np.isin(month, months)
    return act[keep]-fc[keep], act[keep].max(0), [s['cap'] for s in ss]


def describe(e, upper):
    dev = np.clip(0.5+e, 0, upper)-0.5
    row = one.describe(dev, None)
    row['station_q001_median'] = float(np.median(np.quantile(dev, 0.001, axis=0)))
    row['station_q999_median'] = float(np.median(np.quantile(dev, 0.999, axis=0)))
    row['station_skew_median'] = float(np.median(stats.skew(dev, axis=0)))
    row['intervals'] = len(e)
    return row


OPTIONS = {
    'A 15min HA4 (current S1)': dict(),
    'B 5min HA4': dict(resolution=5),
    'C 15min DA': dict(forecast='DA'),
    'D 5min DA': dict(resolution=5, forecast='DA'),
    'E 15min HA4 May-Sep': dict(months=[5, 6, 7, 8, 9]),
    'F 5min HA4 May-Sep': dict(resolution=5, months=[5, 6, 7, 8, 9]),
    'G 15min HA4 smallest 8 sites': dict(site_rule='smallest'),
    'H 5min HA4 smallest 8 sites': dict(site_rule='smallest', resolution=5),
}


def main():
    rows = {}
    for name, kw in OPTIONS.items():
        e, up, caps = records(**kw); rows[name] = dict(describe(e, up), capacities=caps)
    keys = ['intervals', 'rms', 'kurt_median', 'station_skew_median', 'station_q001_median', 'station_q999_median', 'sq_corr', 'share_half_big', 'corr', 'q001', 'q999']
    print('option'.ljust(30)+''.join(k[:12].rjust(13) for k in keys))
    for name, r in rows.items():
        print(name.ljust(30)+''.join(f'{r[k]:13.3f}' for k in keys))
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'tail_options.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
