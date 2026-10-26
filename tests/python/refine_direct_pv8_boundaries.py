"""Extend analytic plot intervals until omitted boundary mass is below 1e-8."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from direct_pv8_inversion import inverse_pdf,moments
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'result/case33bw/direct_pv8_python_cloud_20261026/analytic_only'


def run():
    records=[]
    for path in sorted(OUT.glob('K*/*/*.json')):
        r=json.loads(path.read_text())
        if r.get('status')!='passed':continue
        if r['cdf_start']>1e-8 or r['cdf_end']<1-1e-8:
            source=path.with_suffix('.csv');old=pd.read_csv(source)
            a=np.load(path.with_name(path.stem+'_characteristic_terms.npz'))
            tm=list(zip(a['weights'],a['m0'],a['lam'],a['beta']))
            mean,sd,_=moments(tm)
            q=old.value.to_numpy()**(2 if r['kind']=='V' else 1)
            lo,hi=q[0],q[-1]
            initial=(lo,hi,r['cdf_start'],r['cdf_end'])
            for _ in range(6):
                lo=mean-2*(mean-lo);hi=mean+2*(hi-mean)
                if r['kind']=='V':lo=max(lo,1e-12)
                _,boundary,_=inverse_pdf(tm,np.array([lo,hi]))
                if boundary[0]<1e-8 and boundary[-1]>1-1e-8:break
            else:raise RuntimeError((str(path),boundary))
            q=np.unique(np.r_[q,np.linspace(lo,q[0],1201),np.linspace(q[-1],hi,1201)])
            pdf,cdf,info=inverse_pdf(tm,q)
            grid=np.sqrt(q) if r['kind']=='V' else q
            if r['kind']=='V':pdf*=2*grid
            history=OUT/'boundary_refinement_history'/path.parent.name;history.mkdir(parents=True,exist_ok=True)
            source.rename(history/source.name);path.rename(history/path.name)
            pd.DataFrame(dict(value=grid,pdf=pdf,cdf=cdf)).to_csv(source,index=False)
            mass=float(np.trapezoid(pdf,grid));error=abs(mass-float(cdf[-1]-cdf[0]))
            assert error<3e-4 and np.diff(cdf).min()>-2e-7
            r.update(boundary_refined_from=initial,mass_on_grid=mass,mass_grid_error=error,
                cdf_start=float(cdf[0]),cdf_end=float(cdf[-1]),minimum_density=float(pdf.min()),
                minimum_cdf_increment=float(np.diff(cdf).min()),inversion=info)
            path.write_text(json.dumps(r,indent=2)+'\n')
            print('Extended full PDF/CDF interval',r['model'],r['object'],flush=True)
        records.append(r)
    (OUT/'integration_diagnostics.json').write_text(json.dumps(records,indent=2)+'\n')
    pd.DataFrame([{key:r[key] for key in ('K','model','object','kind','unit','mass_on_grid','mass_grid_error','minimum_density','cdf_start','cdf_end','minimum_cdf_increment')} for r in records]).to_csv(OUT/'numerical_self_checks.csv',index=False)


if __name__=='__main__':run()
