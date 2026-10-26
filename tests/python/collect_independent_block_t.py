"""Collect completed object files without touching running workers or old outputs."""
import json
from pathlib import Path
import pandas as pd
from independent_block_t_fit import OUT


def collect(partial=False):
 records=[]
 for p in sorted((OUT/'distributions').glob('K*/*/*.json')):
  try:r=json.loads(p.read_text())
  except json.JSONDecodeError:continue
  if r.get('status')=='complete':records.append(r)
 if not partial:assert len(records)==873,len(records)
 rows=[];quants=[];diagnostics=[]
 for r in records:
  for metric in r['metrics']:
   row=dict(metric);row.pop('W1_diagnostics');row['unit']=r['unit'];row['mean_absolute_relative_error']=abs(row['mean_relative_error']) if row.get('mean_relative_error') is not None else None;rows.append(row)
  for q in r['quantiles']:quants.append(dict(q,unit=r['unit']))
  diagnostics.append(dict(K=r['K'],model=r['family'],object=r['object'],seconds=r['seconds'],strength_level=r['strength_level'],strength_CDF_change=r['strength_history'][-1]['maximum_CDF_change'],PDF_mass=r['PDF_mass'],PDF_CDF_mass_difference=r['PDF_CDF_mass_difference'],CDF_roundoff_correction=r['CDF_roundoff_monotonic_correction'],quantile_grid_sensitivity=r['model_quantile_grid_sensitivity'],maximum_inverse_residual=r['maximum_CDF_inverse_residual'],finite_strength_variance_relative_difference=r['quadrature_variance_relative_difference']))
 prefix='partial_' if partial else ''
 pd.DataFrame(rows).to_csv(OUT/(prefix+'per_object_errors_and_moments.csv'),index=False);pd.DataFrame(quants).to_csv(OUT/(prefix+'extreme_quantiles.csv'),index=False);pd.DataFrame(diagnostics).to_csv(OUT/(prefix+'integration_audit.csv'),index=False)
 print('COLLECTED',len(records),'objects',len(rows),'reference rows')

if __name__=='__main__':
 import sys
 collect('--partial' in sys.argv)
