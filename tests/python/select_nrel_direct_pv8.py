"""Select eight synchronous NREL output records without adding power variation.

Selection: 01_IDEA/plans/nrel_direct_output_selection_20261026.md.
Only native 5-minute Actual output is read. No forecast files are used.
"""
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data' / 'nrel_sind' / 'co'
OUT = ROOT / 'data' / 'nrel_sind' / 'pv8_direct_2006'
PLAN = ROOT / '01_IDEA/plans/nrel_direct_output_selection_20261026.md'
FIRST_SLOT, LAST_SLOT = 120, 168  # 10:00 <= LocalTime < 14:00.


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sites = []
    for path in sorted(SOURCE.glob('Actual_*_5_Min.csv')):
        fields = path.name.split('_')
        lat, lon, capacity = float(fields[1]), float(fields[2]), float(fields[5][:-2])
        if not (39.55 <= lat <= 39.95 and -105.15 <= lon <= -104.75):
            continue
        power = np.loadtxt(path, delimiter=',', skiprows=1, usecols=1)
        assert power.shape == (365 * 288,), f'Unexpected record count: {path}'
        assert np.isfinite(power).all() and power.min() >= 0, f'Invalid output: {path}'
        daytime = (power / capacity).reshape(365, 288)[:, FIRST_SLOT:LAST_SLOT]
        changes = np.diff(daytime, axis=1).ravel()
        sites.append(dict(
            path=path, latitude=lat, longitude=lon, capacity_MW=capacity,
            q95_abs_5min_change_pu=float(np.quantile(np.abs(changes), .95)),
            rms_5min_change_pu=float(np.sqrt(np.mean(changes ** 2))),
            daytime_sd_pu=float(daytime.std()),
        ))
    # Ignore floating-point roundoff when determining ties in the quantile.
    sites.sort(key=lambda s: (
        -round(s['q95_abs_5min_change_pu'], 12),
        -round(s['rms_5min_change_pu'], 12), s['path'].name,
    ))
    assert len(sites) >= 8, 'Fewer than eight eligible stations'
    selected = sites[:8]
    times, values = None, []
    for site in selected:
        with site['path'].open(newline='') as handle:
            reader = csv.reader(handle)
            assert next(reader) == ['LocalTime', 'Power(MW)']
            rows = list(reader)
        stamp = [row[0] for row in rows]
        if times is None:
            times = stamp
            parsed = [datetime.strptime(t, '%m/%d/%y %H:%M') for t in times]
            assert parsed[0] == datetime(2006, 1, 1)
            assert parsed[-1] == datetime(2006, 12, 31, 23, 55)
            assert all(b - a == timedelta(minutes=5) for a, b in zip(parsed, parsed[1:]))
        else:
            assert stamp == times, f'Timestamps differ: {site["path"]}'
        values.append([row[1] for row in rows])
    matrix = np.array(values, dtype=float).T
    capacities = np.array([s['capacity_MW'] for s in selected])
    keep = [i for i, t in enumerate(parsed) if 10 <= t.hour < 14]
    assert len(keep) == 365 * 48
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / 'pv8_2006_5min_MW.csv'
    pu_path = OUT / 'pv8_2006_1000_1400_pu.csv'
    with raw_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['LocalTime'] + [f'PV{i + 1}_MW' for i in range(8)])
        for i, stamp in enumerate(times):
            writer.writerow([stamp] + [column[i] for column in values])
    with pu_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['LocalTime'] + [f'PV{i + 1}_pu' for i in range(8)])
        for i in keep:
            writer.writerow([times[i]] + [format(v, '.15g') for v in matrix[i] / capacities])
    table_path = OUT / 'sites.csv'
    table = [dict(
        station=f'PV{i + 1}', latitude=s['latitude'], longitude=s['longitude'],
        capacity_MW=s['capacity_MW'], q95_abs_5min_change_pu=s['q95_abs_5min_change_pu'],
        rms_5min_change_pu=s['rms_5min_change_pu'],
        source=str(s['path'].relative_to(ROOT)), source_sha256=sha256(s['path']),
    ) for i, s in enumerate(selected)]
    with table_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    # Verify saved data against source values, including the retained time rows.
    with raw_path.open(newline='') as handle:
        reader = csv.reader(handle)
        next(reader)
        for i, row in enumerate(reader):
            assert row == [times[i]] + [column[i] for column in values]
    with pu_path.open(newline='') as handle:
        reader = csv.reader(handle)
        next(reader)
        rows = list(reader)
    assert [r[0] for r in rows] == [times[i] for i in keep]
    saved_pu = np.array([r[1:] for r in rows], dtype=float)
    assert np.allclose(saved_pu, matrix[keep] / capacities, rtol=0, atol=1e-14)
    corr = np.corrcoef(saved_pu.T)[np.triu_indices(8, 1)]
    manifest = dict(
        date='2026-10-26', source_type='NREL weather-driven simulated PV power output',
        source_url='https://www.nlr.gov/grid/solar-power-data', time_basis='LocalTime as supplied',
        eligible_sites=len(sites), selected_sites=table,
        selection='Top eight daytime q95 absolute 5-minute per-unit changes; RMS breaks ties',
        plan=str(PLAN.relative_to(ROOT)), plan_sha256=sha256(PLAN),
        script_sha256=sha256(Path(__file__)), numpy_version=np.__version__,
        full_year_rows=len(times), daytime_rows=len(keep), interval_minutes=5,
        daytime_window='10:00 <= LocalTime < 14:00; all 365 dates retained',
        raw_values_preserved=True, timestamps_aligned=True, added_power_variation=False,
        pv_pu_min_per_site=saved_pu.min(0).tolist(), pv_pu_max_per_site=saved_pu.max(0).tolist(),
        daytime_pairwise_correlation=dict(min=float(corr.min()), median=float(np.median(corr)), max=float(corr.max())),
        eligible_station_ranking=[{k: (str(v.relative_to(ROOT)) if k == 'path' else v)
                                   for k, v in s.items()} for s in sites],
        artifacts=[dict(path=str(p.relative_to(ROOT)), sha256=sha256(p))
                   for p in (raw_path, pu_path, table_path)],
        evidence='Data integrity checked; no AC power flow or TMM/GMM comparison performed',
    )
    (OUT / 'selection_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(full_year_rows=len(times), daytime_rows=len(keep),
                         stations=table, correlation=manifest['daytime_pairwise_correlation']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
