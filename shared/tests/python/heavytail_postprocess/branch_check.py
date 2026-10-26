"""Independent SI/AC/quadratic/CDF checks of the frozen IEEE33 study.

No archived code is imported with side effects; inputs and fitted laws are frozen.
"""
import recompute as r
import root_cause as cause
import numpy as np
import pandas as pd
import re,json,time,argparse
from scipy import optimize

OUT=r.ROOT/'diagnostics/branch_check'
GENERATORS=np.array([18,17,16,15,14,13,33,22])-1

def raw_network():
    text=re.sub(r'%[^\n]*','',(r.ARCH/'data/cases/case33bw_modify.m').read_text())
    def table(name):
        content=re.search(r'mpc\.'+name+r'\s*=\s*\[(.*?)\];',text,re.S)[1]
        return np.array([[float(z) for z in row.split()] for row in content.split(';') if row.strip()])
    bus=table('bus');lines=table('branch');base=float(re.search(r'mpc.baseMVA\s*=\s*([\d.]+)',text)[1])
    active=lines[lines[:,10]>0].copy()
    assert len(active)==32 and np.all(active[:,4]==0) and np.all(active[:,8:10]==0)
    assert np.all(bus[:,4:6]==0) and np.all(bus[:,9]==bus[0,9])
    f=active[:,0].astype(int)-1;t=active[:,1].astype(int)-1
    vbase=bus[0,9]*1000;zbase=vbase**2/(base*1e6)
    zohm=active[:,2]+1j*active[:,3];zpu=zohm/zbase
    y=np.zeros((33,33),complex)
    for a,b,z in zip(f,t,zpu):
        y[a,a]+=1/z;y[b,b]+=1/z;y[a,b]-=1/z;y[b,a]-=1/z
    return dict(base=base,bus=bus,raw=active,f=f,t=t,zpu=zpu,zohm=zohm,vbase=vbase,Y=y)

def line_power(v,n):
    va=v[:,n['f']];vb=v[:,n['t']];ipu=(va-vb)/n['zpu']
    sf=n['base']*va*np.conj(ipu);st=-n['base']*vb*np.conj(ipu)
    iphase=(va-vb)*(n['vbase']/np.sqrt(3))/n['zohm']
    sf_SI=3*va*(n['vbase']/np.sqrt(3))*np.conj(iphase)/1e6
    er,ei=va.real,va.imag;dr,di=(va-vb).real,(va-vb).imag
    g=(1/n['zpu']).real;b=(1/n['zpu']).imag
    ir=g*dr-b*di;ii=b*dr+g*di
    cart=n['base']*((er*ir+ei*ii)+1j*(ei*ir-er*ii))
    nodal=n['base']*v*np.conj(v@n['Y'].T)
    loss=n['base']*n['zpu']*abs(ipu)**2
    return sf,st,loss,nodal,float(abs(sf_SI-sf).max()),float(abs(cart-sf).max())

def affine_voltage(co):
    v0=np.r_[1.+0j,co['center'][:32]+1j*co['center'][32:]]
    T=np.vstack([np.zeros(len(co['scale']),complex),(co['basis'][:32]+1j*co['basis'][32:])*co['scale']])
    return v0,T

def independent_forms(co,n):
    v0,T=affine_voltage(co);d=T.shape[1];A=[];b=[];c=[]
    for kind in ['P','Q']:
        for a,bb,z in zip(n['f'],n['t'],n['zpu']):
            row=np.zeros(33,complex);row[a]=1/z;row[bb]=-1/z
            M=np.zeros((33,33),complex);M[:,a]=n['base']*np.conj(row)
            H=(M+M.conj().T)/2 if kind=='P' else (M-M.conj().T)/(2j)
            A.append(np.real(T.conj().T@H@T));b.append(2*np.real(v0.conj()@H@T));c.append(np.real(v0.conj()@H@v0))
    return dict(A=np.array(A),b=np.array(b),c=np.array(c))

def downstream(n):
    masks=[];buses=[]
    for root in n['t']:
        descendants={int(root)}
        for a,b in zip(n['f'],n['t']):
            if int(a) in descendants:descendants.add(int(b))
        assert all(int(a) not in descendants or int(b) in descendants for a,b in zip(n['f'],n['t']))
        masks.append(np.array([int(b) in descendants for b in n['t']],float));buses.append(sorted(descendants))
    return np.array(masks),buses

def loss_forms(co,n):
    v0,T=affine_voltage(co);a=n['f'];b=n['t']
    h=(T[a]-T[b])/n['zpu'][:,None];i0=(v0[a]-v0[b])/n['zpu']
    A=np.real(np.einsum('li,lj->lij',np.conj(h),h));lin=2*np.real(np.conj(i0)[:,None]*h);c=abs(i0)**2
    return {q:dict(A=A*(n['base']*z)[:,None,None],b=lin*(n['base']*z)[:,None],c=c*(n['base']*z)) for q,z in [('P',n['zpu'].real),('Q',n['zpu'].imag)]}

def fixed_Q_forms(co,n):
    mask,buses=downstream(n);lf=loss_forms(co,n)['Q']
    out={key:np.einsum('jl,l...->j...',mask,value) for key,value in lf.items()}
    out['c']+=np.array([n['bus'][ix,3].sum()/1000 for ix in buses])
    return out

def evaluate(z,forms):
    return np.einsum('ni,qij,nj->nq',z,forms['A'],z)+z@forms['b'].T+forms['c']

def independent_ac(power,n):
    target=-(n['bus'][1:,2]+1j*n['bus'][1:,3])/1000/n['base']
    target=target.copy();target[GENERATORS-1]+=power/n['base']
    def mismatch(state):
        v=np.r_[1.+0j,np.exp(state[32:]+1j*state[:32])]
        s=v*np.conj(n['Y']@v)
        return np.r_[(s[1:]-target).real,(s[1:]-target).imag]
    sol=optimize.root(mismatch,np.zeros(64),method='hybr',options={'xtol':1e-10})
    state=sol.x.copy()
    # Polish the independent solve rather than relaxing its power tolerance.
    for _ in range(5):
        residual=float(abs(mismatch(state)).max())
        if residual<2e-12:break
        h=1e-6;eye=np.eye(64)*h
        J=np.column_stack([(mismatch(state+step)-mismatch(state-step))/(2*h) for step in eye])
        state-=np.linalg.solve(J,mismatch(state))
    residual=float(abs(mismatch(state)).max())
    assert residual<2e-10,(sol.message,residual,power)
    return np.r_[1.+0j,np.exp(state[32:]+1j*state[:32])],residual

def basic_checks():
    n=raw_network();ns=r.numerical_functions();rows=[];mcrows=[]
    for scene in r.SCENES:
        d=r.OLD/scene;co=dict(np.load(d/'voltage_coordinates.npz'))
        a=np.load(d/'estimation.npz');b=np.load(d/'selection.npz');v=np.r_[a['voltage'],b['voltage']]
        sf,st,loss,nodal,sierr,carterr=line_power(v,n)
        canonical=ns['base'].evaluate_outputs(v,ns['base'].BASE,ns['base'].BUS,ns['base'].BRANCH)
        stater=np.c_[v[:,1:].real,v[:,1:].imag];z=(stater-co['center'])@co['basis']/co['scale']
        v0,T=affine_voltage(co);vr=v0+z@T.T
        ind=independent_forms(co,n);old=ns['quadratic_forms'](co)
        canonical_forms={key:old[key][32:96] for key in ['A','b','c']}
        # Include off-AC voltage combinations to expose formula bugs independent of power balance.
        probe=np.r_[z[:512],np.random.default_rng(2026102602).standard_normal((512,len(co['scale'])))*3]
        probe_v=v0+probe@T.T;psf,*_=line_power(probe_v,n)
        exact=np.c_[psf.real,psf.imag];indval=evaluate(probe,ind);oldval=evaluate(probe,canonical_forms)
        newQ=fixed_Q_forms(co,n);qval=evaluate(z,newQ)
        mask,buses=downstream(n);qloss=loss.imag@mask.T+np.array([n['bus'][ix,3].sum()/1000 for ix in buses])
        row=dict(scene=scene,fast_scenarios=len(v),SI_vs_pu_max_MVA=sierr,Cartesian_vs_complex_max_MVA=carterr,
            independent_vs_existing_max_MW_Mvar=float(abs(np.c_[sf.real,sf.imag]-canonical[:,:64]).max()),
            branch_loss_identity_max_MVA=float(abs(sf+st-loss).max()),
            independent_quadratic_vs_complex_max_MW_Mvar=float(abs(indval-exact).max()),
            archived_quadratic_vs_complex_max_MW_Mvar=float(abs(oldval-exact).max()),
            coefficient_A_max_difference=float(abs(ind['A']-canonical_forms['A']).max()),
            coefficient_b_max_difference=float(abs(ind['b']-canonical_forms['b']).max()),
            coefficient_c_max_difference=float(abs(ind['c']-canonical_forms['c']).max()),
            projection_voltage_max_pu=float(abs(vr-v).max()),
            projected_fixed_Q_vs_Sf_max_Mvar=float(abs(qval-sf.imag).max()),
            unprojected_fixed_Q_vs_Sf_max_Mvar=float(abs(qloss-sf.imag).max()),
            full_fixed_Q_nodal_residual_max_Mvar=float(abs(nodal[:,1:].imag+n['bus'][1:,3]/1000).max()))
        assert max(sierr,carterr,row['independent_vs_existing_max_MW_Mvar'],row['independent_quadratic_vs_complex_max_MW_Mvar'],row['archived_quadratic_vs_complex_max_MW_Mvar'])<1e-9,row
        rows.append(row)
        ref=np.load(d/'mc_outputs.npy',mmap_mode='r');power=np.load(d/'mc_injection_MW.npy',mmap_mode='r')
        chosen=set(np.random.default_rng(2026102603).choice(len(ref),24,replace=False).tolist())
        for col in [16,32,*range(64,96)]:
            chosen.update([int(np.argmin(ref[:,col])),int(np.argmax(ref[:,col]))])
        for index in sorted(chosen):
            vv,res=independent_ac(power[index],n);ss,tt,ll,nn,*_=line_power(vv[None,:],n)
            result=np.r_[abs(vv[1:]),ss.real.ravel(),ss.imag.ravel(),ll.real.sum()]
            error=abs(result-ref[index])
            qdirect=ll.imag@mask.T+np.array([n['bus'][ix,3].sum()/1000 for ix in buses])
            mcrows.append(dict(scene=scene,index=index,power_residual_pu=res,
                voltage_vs_saved_MC_max_pu=float(error[:32].max()),
                branch_vs_saved_MC_max_MW_Mvar=float(error[32:96].max()),
                loss_vs_saved_MC_MW=float(error[96]),
                fixed_Q_vs_Sf_max_Mvar=float(abs(qdirect-ss.imag).max())))
        print('basic',scene,row,'MCchecked',len(chosen),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'three_phase_unit_power_formula.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(mcrows).to_csv(OUT/'independent_acpf_vs_original_mc.csv',index=False,encoding='utf-8-sig')
    assert max(x['branch_vs_saved_MC_max_MW_Mvar'] for x in mcrows)<1e-7
    r.write_json(OUT/'base_power_calc_check.json',dict(cases=4,branches=32,MC_checks=len(mcrows),independent_solver='SciPy finite-difference polar root; flat voltage start',baseMVA=n['base'],base_kV=n['vbase']/1000,passed=True))

def direct_distribution_check():
    n=raw_network();rows=[];N=2**18
    dkw=np.sqrt(np.log(2*(4*3*64)/.05)/(2*N))
    for scene in r.SCENES:
        d=r.OLD/scene;co=dict(np.load(d/'voltage_coordinates.npz'));v0,T=affine_voltage(co)
        models=r.read_models(d/'models_K2.pkl')
        for model,kind in r.MODELS.items():
            rng=np.random.default_rng(2026102604);pieces=[]
            for start in range(0,N,16384):
                z=cause.draw(models[model],kind,min(16384,N-start),rng)
                v=v0+z@T.T;sf,*_=line_power(v,n);pieces.append(np.c_[sf.real,sf.imag])
            q=np.concatenate(pieces);a=np.load(d/f'CDF_{model}_K2.npz');thresholds=a['thresholds'];F=a['CDF']
            for j in range(64):
                values=np.sort(q[:,j]);emp=np.searchsorted(values,thresholds[32+j,:100],side='right')/N
                error=abs(emp-F[32+j,:100]);rows.append(dict(scene=scene,model=model,K=2,quantity=('P' if j<32 else 'Q')+str(j%32+1),n=N,
                    max_direct_vs_analytic_CDF_pp=float(error.max()*100),direct_vs_analytic_CDF_RMSE_pp=float(np.sqrt(np.mean(error**2))*100),
                    simultaneous_DKW_95_bound_pp=float(dkw*100),within_DKW_95_bound=bool(error.max()<dkw)))
            print('distribution',scene,model,'maxpp',max(x['max_direct_vs_analytic_CDF_pp'] for x in rows if x['scene']==scene and x['model']==model),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'same_voltage_law_analytic_vs_direct_cdf.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('step',choices=['basic','distribution']);args=p.parse_args()
    if args.step=='basic':basic_checks()
    else:direct_distribution_check()
