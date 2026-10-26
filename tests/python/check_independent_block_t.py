"""Small independent likelihood/parameter-origin/limit/W1 checks; no AC computation."""
import json
from pathlib import Path
import numpy as np
from scipy import stats
from independent_block_t_fit import CORE,block_density,geometry,OUT
from independent_tail_w1 import tail_w1


def run():
 checks=[];x=np.linspace(-2,2,45).reshape(15,3);mu=np.array([.1,-.2,.3]);C=np.array([[1.,.2,0],[.2,2.,.1],[0,.1,.7]]);model=dict(weights=np.ones(1),means=mu[None],scatter=C[None],df=np.array([7.]))
 value=CORE['log_density'](x,model)[0];exact=stats.multivariate_t.logpdf(x,loc=mu,shape=C,df=7);err=float(abs(value-exact).max());assert err<1e-12;checks.append(dict(check='standard t density',maximum_difference=err))
 block=dict(model,intercept=np.zeros((1,0)),coefficient=np.zeros((1,0,3)),residual=np.zeros((1,0,0)))
 err=float(abs(block_density(x,np.empty((len(x),0)),block)[0]-value).max());assert err<1e-12;checks.append(dict(check='r=d block equals standard t',maximum_difference=err))
 U=x[:,:1];W=x[:,1:];b=dict(weights=np.ones(1),means=np.array([[.1]]),scatter=np.array([[[1.2]]]),df=np.array([np.inf]),intercept=np.array([[.2,-.1]]),coefficient=np.array([[[.3],[-.2]]]),residual=np.array([[[.8,.1],[.1,1.1]]]))
 G=b['coefficient'][0];cov=np.block([[b['scatter'][0],b['scatter'][0]@G.T],[G@b['scatter'][0],G@b['scatter'][0]@G.T+b['residual'][0]]]);m=np.r_[b['means'][0],b['intercept'][0]+G@b['means'][0]]
 err=float(abs(block_density(U,W,b)[0]-stats.multivariate_normal.logpdf(x,mean=m,cov=cov)).max());assert err<1e-12;checks.append(dict(check='block Gaussian limit',maximum_difference=err))
 b0=dict(weights=np.ones(1),means=np.empty((1,0)),scatter=np.empty((1,0,0)),df=np.array([np.inf]),intercept=mu[None],coefficient=np.empty((1,3,0)),residual=C[None]);err=float(abs(block_density(np.empty((len(x),0)),x,b0)[0]-stats.multivariate_normal.logpdf(x,mean=mu,cov=C)).max());assert err<1e-12;checks.append(dict(check='r=0 Gaussian',maximum_difference=err))
 nu=1e6;b['df'][:]=nu;limit=float(abs(block_density(U,W,b)[0]-stats.multivariate_normal.logpdf(x,mean=m,cov=cov)).max());assert limit<2e-5;checks.append(dict(check='large finite nu Gaussian limit without forced substitution',nu=nu,maximum_difference=limit))
 ref=np.sort(np.array([-2.,-1.,-.3,0.,.2,.4,.5,.9,1.2,2.]));emp=lambda p:ref[np.ceil(len(ref)*np.asarray(p)).astype(int)-1]
 z,diag=tail_w1(emp,ref,.01,.05);assert z==0;checks.append(dict(check='W1 same empirical distribution zero',value=z))
 for a,bnd in [(.01,.05),(.95,.99)]:
  value,diag=tail_w1(lambda p:emp(p)+.37,ref,a,bnd);assert abs(value-.37)<1e-13;checks.append(dict(check='W1 shift identity',window=[a,bnd],value=value))
 ref=stats.norm.ppf((np.arange(3504)+.5)/3504);value,diag=tail_w1(stats.norm.ppf,ref,.95,.99,stats.norm.cdf);scaled,d2=tail_w1(lambda p:10*stats.norm.ppf(p),10*ref,.95,.99,lambda x:stats.norm.cdf(x/10));assert abs(scaled-10*value)<1e-10;checks.append(dict(check='W1 empirical jumps convergence and units',value=value,scaled=scaled,diagnostic=diag))
 source=(Path(__file__).parent/'independent_block_t_fit.py').read_text();forbidden=['block_from_gmm','models/K1.pkl','gmm_warm'];assert all(word not in source for word in forbidden);g=geometry();assert g['Q1'].shape==(64,8);selected=json.load(open(OUT/'selected_models.json'));assert len(selected)==9 and all(r['status']=='converged' for r in selected);checks.append(dict(check='parameter origin and selected-model convergence',forbidden_calls_absent=True,old_parameter_files_read=False,selected_models=9))
 (OUT/'early_numerical_checks.json').write_text(json.dumps(dict(status='passed',checks=checks),indent=2)+'\n');print('PASSED',len(checks),'checks')

if __name__=='__main__':run()
