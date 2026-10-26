"""Check existing saved voltage laws against the active paper moment identities.

Reads completed scenarios and models only; no new injections, AC, or fitting.
"""
import sys,pickle,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats,integrate,special
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests/python'))
import run_direct_heavytail_case33 as r
AUD=Path(__file__).resolve().parent

def true_strength_moments(t):
    if t.gauss:return [1.,1.,1.,1.]
    a=t.nu/2;A=t.nu/2;top=np.log(t.v_max)
    moments=[]
    for k in range(1,5):
        f=lambda z: np.exp((k-a)*z-A*np.exp(-z)+a*np.log(A)-special.gammaln(a))/t.pmax/t.mean**k
        val,err=integrate.quad(f,-40,top,epsabs=1e-9,epsrel=1e-10,limit=200)
        assert err<1e-6*max(1,abs(val)),(val,err)
        moments.append(val)
    return moments

def main():
    component_rows=[];strength_rows=[];physical_rows=[];split_rows=[]
    fourth_rows=[];std_fourth_rows=[];physical_cov_rows=[]
    for scene in r.SCENES:
        d=r.OUT/scene;a=np.load(d/'estimation.npz');coords=dict(np.load(d/'voltage_coordinates.npz'))
        state=np.c_[a['voltage'][:,1:].real,a['voltage'][:,1:].imag]
        x=(state-coords['center'])@coords['basis']/coords['scale'];data=dict(coords)
        forms=r.an.quadratic_forms(data)
        split=json.loads((d/'split.json').read_text())
        split_rows.append(dict(scene=scene,remaining_excess_kurtosis_max=split['remaining_kurtosis_max_after'],
            missing_injection_groups=','.join(sorted(set('P'+str(n) for n in r.NODES)-{p for g in split['groups'] for p in g})),
            first_order_voltage_variance_share=split['voltage_share_first_order'],
            represented_voltage_variance_share=split['amplified_voltage_share']))
        for k in r.KS:
            with (d/f'models_K{k}.pkl').open('rb') as f:models=pickle.load(f)
            _,resp,_=r.log_density(x,models['AQGMM'])
            shift=models['grouped_t_shift'];Q1,Q2=shift['Q1'],shift['Q2'];Q=np.c_[Q1,Q2]
            U=x@Q1;W=x@Q2
            # Use refined quadrature for the actual stored model law, not its 2000-point moment approximation.
            r.an.U_NODES=96;un,uw=r.an.u_nodes()
            r.an.U_NODES=192;un2,uw2=r.an.u_nodes()
            for ci,c in enumerate(shift['shifted']):
                w=resp[:,ci];w=w/w.sum();mu=w@U;E=U-mu;C=E.T@(E*w[:,None]);emp3=w@(E**3)
                stats_grid=[]
                for gridu,gridw in [(un,uw),(un2,uw2)]:
                    S=np.stack([np.ones(len(gridu)) if t.gauss else t.nu/stats.chi2.isf(gridu*t.pmax,t.nu)/t.mean for t in c['trunc']])
                    es=S@gridw;pair=(np.sqrt(S)*gridw)@np.sqrt(S).T
                    covS=(S*gridw)@S.T-np.outer(es,es)
                    gid=c['gid'];g=c['shift'];m_model=c['mu']+g*(es[gid]-1)
                    C_model=c['scatter']*pair[np.ix_(gid,gid)]+np.outer(g,g)*covS[np.ix_(gid,gid)]
                    centered=S-es[:,None];s3=(centered**3)@gridw
                    third=g**3*s3[gid]+3*g*np.diag(c['scatter'])*np.diag(covS)[gid]
                    stats_grid.append((es,m_model,C_model,third))
                es,mm,CC,third=stats_grid[-1]
                relcov=float(np.linalg.norm(CC-C)/np.linalg.norm(C));rel3=float(np.linalg.norm(third-emp3)/np.linalg.norm(emp3))
                component_rows.append(dict(scene=scene,K=k,component=ci,model='grouped_t_shift',
                    group_mean_max_deviation=float(abs(es-1).max()),
                    voltage_U_mean_abs_max=float(abs(mm-mu).max()),covariance_relative_Frobenius_error=relcov,
                    marginal_variance_relative_max=float(abs(np.diag(CC)/np.diag(C)-1).max()),
                    third_moment_relative_vector_error=rel3,mean_shift_scale=c['shift_scale'],cross_covariance_scale=c['cross_shrink'],
                    integration_refinement_covariance_difference=float(np.linalg.norm(stats_grid[0][2]-CC)/max(1,np.linalg.norm(CC))),
                    gaussian_direction_covariance_target_min=float(np.linalg.eigvalsh(C).min())))
                for gi,t in enumerate(c['trunc']):
                    m=true_strength_moments(t)
                    strength_rows.append(dict(scene=scene,K=k,component=ci,group=gi,gaussian=t.gauss,
                        exact_strength_mean=m[0],exact_strength_second_moment=m[1],target_group_radial_fourth_ratio=c['m'][gi],
                        stored_second_moment=1. if t.gauss else t.ratio,
                        relative_second_moment_mismatch=(m[1]/c['m'][gi]-1) if c['m'][gi] else None))
                    group=c['groups'][gi];A=np.linalg.inv(C[np.ix_(group,group)])
                    D=c['scatter'][np.ix_(group,group)];gamma=c['shift'][group];B=A@D
                    es1,es2,es3,es4=m;tr=np.trace(B);ga=gamma@A@gamma;gca=gamma@A@D@A@gamma
                    M3=es3-2*es2+es1;M4=es4-4*es3+6*es2-4*es1+1
                    radial4=(es2*(2*np.trace(B@B)+tr*tr)+M4*ga*ga+M3*(2*tr*ga+4*gca))/(len(group)*(len(group)+2))
                    fourth_rows.append(dict(scene=scene,K=k,component=ci,group=gi,
                        target_radial_fourth=c['m'][gi],actual_radial_fourth=radial4,ratio=radial4/c['m'][gi]))
                # Standard TMM moments should be exact relative to the weighted estimation moments.
                st=models['standard_TMM'];em=w@x;ee=x-em;ec=ee.T@(ee*w[:,None]);nu=st['df'][ci]
                sc=st['scatter'][ci]*(nu/(nu-2) if np.isfinite(nu) else 1.)
                distance=np.einsum('ij,ij->i',ee@np.linalg.inv(ec),ee)
                b=float(w@(distance**2)/(x.shape[1]*(x.shape[1]+2)))
                actual_b=(nu-2)/(nu-4) if np.isfinite(nu) else 1.
                std_fourth_rows.append(dict(scene=scene,K=k,component=ci,target_radial_fourth=b,stored_df=nu,
                    actual_radial_fourth=actual_b,target_unclipped_df=(4*b-2)/(b-1) if b>1 else np.inf,
                    df_floor_active=bool(np.isfinite(nu) and nu<=4.10000001),ratio=actual_b/b))
                G=c['coefficient'];mu_full=Q1@mm+Q2@(c['intercept']+G@mm)
                cov_full=Q@np.block([[CC,CC@G.T],[G@CC,G@CC@G.T+c['residual']]])@Q.T
                T=coords['basis']*coords['scale'];true_physical=T@ec@T.T;model_physical=T@cov_full@T.T
                physical_cov_rows.append(dict(scene=scene,K=k,component=ci,
                    physical_voltage_covariance_relative_error=float(np.linalg.norm(model_physical-true_physical)/np.linalg.norm(true_physical)),
                    physical_voltage_mean_max_difference_pu=float(abs(T@(mu_full-em)).max())))
                component_rows.append(dict(scene=scene,K=k,component=ci,model='standard_TMM',
                    group_mean_max_deviation=0.,voltage_U_mean_abs_max=float(abs(st['means'][ci]-em).max()),
                    covariance_relative_Frobenius_error=float(np.linalg.norm(sc-ec)/np.linalg.norm(ec)),
                    marginal_variance_relative_max=float(abs(np.diag(sc)/np.diag(ec)-1).max()),
                    third_moment_relative_vector_error=None,mean_shift_scale=None,cross_covariance_scale=None,
                    integration_refinement_covariance_difference=0.,gaussian_direction_covariance_target_min=float(np.linalg.eigvalsh(ec).min())))
            # Electrical means from all Gaussian states, independently of CDF inversion.
            means={}
            for name,kind in [('AQGMM','GMM'),('standard_TMM','TMM'),('grouped_t_shift','SHIFT')]:
                if kind in ['GMM','TMM']:
                    model=models[name];mq=np.zeros(97)
                    for pi,mu,C,nu in zip(model['weights'],model['means'],model['scatter'],model['df']):
                        cv=C*(nu/(nu-2) if np.isfinite(nu) else 1.)
                        mq+=pi*(np.einsum('qij,ji->q',forms['A'],cv)+np.einsum('i,qij,j->q',mu,forms['A'],mu)+forms['b']@mu+forms['c'])
                else:
                    r.an.U_NODES=192;mq=np.zeros(97)
                    for pi,mu,cv in r.an.gaussian_states(models[name],kind):
                        mq+=pi*(np.einsum('qij,ji->q',forms['A'],cv)+np.einsum('i,qij,j->q',mu,forms['A'],mu)+forms['b']@mu+forms['c'])
                means[name]=mq
            for name in ['standard_TMM','grouped_t_shift']:
                for j in [16,31,32,48,80,96]:
                    physical_rows.append(dict(scene=scene,K=k,model=name,object=r.object_name(j),
                        electrical_mean_difference_from_AQGMM=float(means[name][j]-means['AQGMM'][j])))
        print(scene,'audited',flush=True)
    pd.DataFrame(component_rows).to_csv(AUD/'component_moment_consistency.csv',index=False)
    pd.DataFrame(strength_rows).to_csv(AUD/'strength_moment_consistency.csv',index=False)
    pd.DataFrame(physical_rows).to_csv(AUD/'electrical_mean_consistency.csv',index=False)
    pd.DataFrame(split_rows).to_csv(AUD/'group_coverage.csv',index=False)
    pd.DataFrame(fourth_rows).to_csv(AUD/'group_voltage_fourth_moments.csv',index=False)
    pd.DataFrame(std_fourth_rows).to_csv(AUD/'standard_TMM_fourth_moment_consistency.csv',index=False)
    pd.DataFrame(physical_cov_rows).to_csv(AUD/'physical_voltage_moment_consistency.csv',index=False)
    t=pd.DataFrame(component_rows)
    print(t.groupby('model')[['group_mean_max_deviation','covariance_relative_Frobenius_error','marginal_variance_relative_max','third_moment_relative_vector_error']].max().to_string())
    print(t[(t.K==2)&(t.model=='grouped_t_shift')].to_string(index=False))

if __name__=='__main__':main()
