"""Known distributions for the new controlled FFT interpolation/period refinement."""
import numpy as np,json
from scipy import stats
from independent_cf_inversion import inverse_pdf
from independent_block_t_numeric import scale_nodes
from independent_block_t_fit import OUT


def run():
 rows=[]
 for label,tm,x,pdf,cdf in [
  ('Gaussian linear',[(1.,0.,np.zeros(1),np.array([2.]))],np.linspace(-10,10,201),lambda x:stats.norm.pdf(x,scale=2),lambda x:stats.norm.cdf(x,scale=2)),
  ('Gaussian 64-d quadratic chi-square',[(1.,0.,np.ones(64),np.zeros(64))],np.linspace(0,150,301),lambda x:stats.chi2.pdf(x,64),lambda x:stats.chi2.cdf(x,64))]:
  p,f,info=inverse_pdf(tm,x);ep=float(abs(p-pdf(x)).max());ef=float(abs(f-cdf(x)).max());assert ep<2e-6 and ef<2e-6;rows.append(dict(check=label,PDF_error=ep,CDF_error=ef))
 nu=4.1;s,w=scale_nodes(nu,8);tm=[(float(wi),0.,np.zeros(1),np.array([np.sqrt(si)])) for wi,si in zip(w,s)];x=np.linspace(-8,8,101);p,f,info=inverse_pdf(tm,x);ep=float(abs(p-stats.t.pdf(x,nu)).max());ef=float(abs(f-stats.t.cdf(x,nu)).max());assert ep<2e-5 and ef<2e-5;rows.append(dict(check='nu=4.1 standard t distribution using untruncated scale',scale_level=8,PDF_error=ep,CDF_error=ef))
 (OUT/'CF_known_distribution_checks.json').write_text(json.dumps(dict(status='passed',checks=rows),indent=2)+'\n');print(rows)

if __name__=='__main__':run()
