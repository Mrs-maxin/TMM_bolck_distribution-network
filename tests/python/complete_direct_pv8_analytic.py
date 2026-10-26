"""Complete saved joint-voltage models' analytic power flow, without method comparisons."""
import os
import sys
from pathlib import Path
import json
import pickle
import time
import hashlib
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'result/case33bw/direct_pv8_python_cloud_20261026'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')


def run():
    os.environ['MAIN_SCENE']='direct_pv8'
    os.environ['MAIN_SHIFT']='1'
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
    sys.argv=sys.argv[:1]
    import numpy as np
    import pandas as pd
    import scipy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import mainline_comparison_case33 as mc
    import analytic_grouped_t_case33 as an
    from direct_pv8_inversion import inverse_pdf,moments
    from easyplot_py import publication_context,save_figure
    destination=OUT/'analytic_only'
    destination.mkdir(exist_ok=True)
    figures=destination/'figures';figures.mkdir(exist_ok=True)
    diagnostics=[]
    started=time.perf_counter()
    model_paths=sorted((OUT/'models').glob('K*.pkl'))
    assert model_paths,'No saved voltage model; parameter estimation requires a separate stage'
    manifest=dict(status='running',scope='Analytic power flow and numerical self-checks only',
        accuracy_comparison='paused_pending_user_confirmation_of_error_definitions',
        parameters_reestimated=False,ac_recomputed=False,models=[p.name for p in model_paths],
        python=sys.version,platform=platform.platform(),host=platform.node(),numpy=np.__version__,
        scipy=scipy.__version__,matplotlib=matplotlib.__version__,cwd=str(ROOT),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        model_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model_paths},
        completed_objects=0,unfinished=['Numerical checks in progress'],
        prior_results='initial_attempt/ retained unchanged; earlier comparisons are not approved evidence')
    write_json(destination/'run_manifest.json',manifest)
    base=mc.base
    ref=np.load(OUT/'ac_results.npz')['reference']
    for path in model_paths:
        stored=pickle.load(path.open('rb'))
        assert stored['forms']['check']<1e-8
        forms={key:stored['forms'][key] for key in ('A','b','c')}
        k=int(path.stem[1:])
        for label,kind,model in stored['models']:
            slug=label.replace(' ','_')
            folder=destination/f'K{k}'/slug;folder.mkdir(parents=True,exist_ok=True)
            an.STRENGTH_NODES,an.U_NODES=40,12
            terms_coarse=an.terms(model,kind,forms)
            an.STRENGTH_NODES,an.U_NODES=80,24
            terms_fine=an.terms(model,kind,forms) if kind!='GMM' else terms_coarse
            for index,obj in enumerate(base.OBJECTS):
                name=mc.object_name(index)
                record_path=folder/f'{name}.json'
                csv_path=folder/f'{name}.csv'
                if record_path.exists() and csv_path.exists():
                    record=json.loads(record_path.read_text())
                    if record.get('status')=='passed':
                        diagnostics.append(record)
                        continue
                tm=terms_fine[index]
                mean,sd,_=moments(tm)
                lo,hi=min(mean-12*sd,float(ref[:,index].min())**(2 if obj['output']=='V' else 1)),max(mean+12*sd,float(ref[:,index].max())**(2 if obj['output']=='V' else 1))
                if obj['output']=='V':lo=max(lo,1e-12)
                # Resolve the complete physical interval and the probability-density peak.
                count=2401 if obj['output']!='loss' else 12001
                qgrid=np.unique(np.r_[np.linspace(lo,hi,count),np.linspace(mean-4*sd,mean+4*sd,count)])
                if obj['output']=='V':qgrid=qgrid[qgrid>0]
                probe=np.linspace(max(lo,mean-6*sd),min(hi,mean+6*sd),101)
                coarse_pdf,coarse_cdf,_=inverse_pdf(terms_coarse[index],probe)
                strength_history=[]
                strength_order=80
                group_order=24
                while True:
                    check_pdf,check_cdf,_=inverse_pdf(tm,probe)
                    dp=float(abs(check_pdf-coarse_pdf).max()*sd)
                    dc=float(abs(check_cdf-coarse_cdf).max())
                    strength_history.append(dict(strength_nodes=strength_order,group_nodes_per_piece=group_order,scaled_pdf_change=dp,cdf_change=dc))
                    if kind=='GMM' or (dp<2e-4 and dc<2e-5):break
                    if strength_order>=640:raise RuntimeError(f'Strength integration unresolved: {name} {label} {strength_history}')
                    coarse_pdf,coarse_cdf=check_pdf,check_cdf
                    strength_order*=2;group_order*=2
                    an.STRENGTH_NODES,an.U_NODES=strength_order,group_order
                    one={key:value[index:index+1] for key,value in forms.items()}
                    tm=an.terms(model,kind,one)[0]
                pdf,cdf,info=inverse_pdf(tm,qgrid)
                boundary_trials=[]
                for boundary_iteration in range(7):
                    boundary_trials.append(dict(lower=float(qgrid[0]),upper=float(qgrid[-1]),cdf_lower=float(cdf[0]),cdf_upper=float(cdf[-1])))
                    if cdf[0]<1e-8 and cdf[-1]>1-1e-8:break
                    lower=mean-2*(mean-qgrid[0]);upper=mean+2*(qgrid[-1]-mean)
                    if obj['output']=='V':lower=max(lower,1e-12)
                    qgrid=np.unique(np.r_[qgrid,np.linspace(lower,qgrid[0],1201),np.linspace(qgrid[-1],upper,1201)])
                    pdf,cdf,info=inverse_pdf(tm,qgrid)
                info['boundary_trials']=boundary_trials
                grid=np.sqrt(qgrid) if obj['output']=='V' else qgrid
                if obj['output']=='V':pdf=pdf*2*grid
                mass=float(np.trapezoid(pdf,grid))
                mass_error=abs(mass-float(cdf[-1]-cdf[0]))
                assert mass_error<3e-4,(name,label,'grid integral',mass_error)
                assert cdf[0]<2e-5 and cdf[-1]>1-2e-5,(name,label,'boundary',cdf[[0,-1]])
                assert np.min(np.diff(cdf))>-2e-7,(name,label,'monotonicity')
                assert pdf.min()*sd>-2e-7,(name,label,'negative density')
                unit='p.u.' if obj['output']=='V' else 'MVAr' if obj['output']=='Q' else 'MW'
                if obj['output']=='V':physical=f"Bus {obj['node']} voltage magnitude"
                elif obj['output']=='loss':physical='System active loss'
                else:
                    br=base.BRANCH[obj['branch']-1]
                    physical=f"Branch {obj['branch']} ({int(br[0])} to {int(br[1])}) sending-end {'active' if obj['output']=='P' else 'reactive'} power"
                pd.DataFrame(dict(value=grid,pdf=pdf,cdf=cdf)).to_csv(csv_path,index=False)
                np.savez_compressed(folder/f'{name}_characteristic_terms.npz',weights=np.array([v[0] for v in tm]),m0=np.array([v[1] for v in tm]),lam=np.array([v[2] for v in tm]),beta=np.array([v[3] for v in tm]))
                record=dict(status='passed',K=k,model=label,object=name,kind=obj['output'],unit=unit,
                    physical_quantity=physical,conditional_weight_sum=float(sum(v[0] for v in tm)),
                    quadratic_form_check=stored['forms']['check'],mass_on_grid=mass,
                    mass_grid_error=mass_error,minimum_density=float(pdf.min()),cdf_start=float(cdf[0]),
                    cdf_end=float(cdf[-1]),minimum_cdf_increment=float(np.diff(cdf).min()),
                    strength_quadrature_history=strength_history,inversion=info,
                    method_comparison=False,clipped=False,smoothed=False)
                write_json(record_path,record);diagnostics.append(record)
                with publication_context(font_family='DejaVu Sans',base_size=10):
                    for quantity,values in [('PDF',pdf),('CDF',cdf)]:
                        fig,ax=plt.subplots(figsize=(100/25.4,76/25.4))
                        ax.plot(grid,values,color='#356D9A',lw=1.1,label=f'{label}, K={k}')
                        ax.set(xlabel=f'{physical} ({unit})',ylabel=f'PDF (1/{unit})' if quantity=='PDF' else 'Cumulative probability')
                        if obj['output'] in ('P','Q','loss') or len(physical)>55:
                            ax.set_xlabel(physical.replace(' sending-end','\nsending-end')+f' ({unit})',fontsize=8)
                        ax.legend(fontsize=7,frameon=False);fig.tight_layout()
                        # Keep raw numerical roundoff, including negative values, visible in data.
                        for ext in ('png','svg'):save_figure(fig,figures/f'{name}_K{k}_{slug}_{quantity}.{ext}',dpi=180,overwrite=True)
                        plt.close(fig)
                manifest.update(completed_objects=len(diagnostics),elapsed_seconds=time.perf_counter()-started,current=f'K{k} {label} {name}')
                write_json(destination/'run_manifest.json',manifest)
                write_json(destination/'integration_diagnostics.json',diagnostics)
                print(json.dumps(dict(stage='analytic_object_saved',K=k,model=label,object=name,completed=len(diagnostics),seconds=manifest['elapsed_seconds'])),flush=True)
    manifest.update(status='complete',unfinished=['Method error definitions and accuracy comparisons paused by user', 'K=2,4 all-object analytic results require saved parameters; no parameter re-estimation in this stage'],completed_objects=len(diagnostics),elapsed_seconds=time.perf_counter()-started)
    write_json(destination/'run_manifest.json',manifest)
    pd.DataFrame([{key:r[key] for key in ('K','model','object','kind','unit','mass_on_grid','mass_grid_error','minimum_density','cdf_start','cdf_end','minimum_cdf_increment')} for r in diagnostics]).to_csv(destination/'numerical_self_checks.csv',index=False)
    print(json.dumps(dict(stage='analytic_complete',objects=len(diagnostics),seconds=manifest['elapsed_seconds'])),flush=True)


if __name__=='__main__':run()
