"""Independent references, complete result joins and checks of tail-analysis data."""
import recompute as r
from tail_w1 import interval_w1
from mean_variance import gaussian_quadratic,VoltageMagnitude
from tail_analysis import pwm_shape
import numpy as np
import pandas as pd
from scipy import stats,integrate
import json
from pathlib import Path
from PIL import Image

OUT=r.ROOT/'tail_distribution'

def trusted_references():
    # Deterministic quantile points approximate the known reference without PPF.
    raw=stats.norm.ppf((np.arange(1000000)+.5)/1000000);checks=[]
    class Normal:
        def __init__(self,loc,scale):self.loc=loc;self.scale=scale
        def cdf(self,x):return stats.norm.cdf(x,self.loc,self.scale)
    for loc,scale in [(.3,1.),(0.,1.2),(.2,.8)]:
        kernel=Normal(loc,scale);known=stats.norm.ppf([.001,.005,.01,.99,.995,.999],loc,scale)
        for a,b in [(.001,.05),(.95,.999)]:
            value=interval_w1(kernel,32,raw,a,b,known,stats.norm.ppf(.75)-stats.norm.ppf(.25))
            exact=integrate.quad(lambda p:abs(stats.norm.ppf(p,loc,scale)-stats.norm.ppf(p)),a,b,epsabs=1e-12)[0]/(b-a)
            checks.append(dict(check='tail_W1_CDF_vs_known_quantiles',location=loc,scale=scale,a=a,b=b,computed=value['W1'],reference=exact,absolute_difference=abs(value['W1']-exact)))
            assert abs(value['W1']-exact)<3e-4,checks[-1]
    for nu in [5.,7.,10.]:
        mu=np.array([.4]);C=np.array([[.09]]);forms=dict(A=np.ones((1,1,1)),b=np.zeros((1,1)),c=np.zeros(1))
        m,v=gaussian_quadratic(forms,mu,C);factor=(nu-2)/(nu-4);v+=(factor-1)*3*C[0,0]**2
        scale=np.sqrt(.09*(nu-2)/nu)
        exactmean=stats.t.moment(2,nu,loc=.4,scale=scale);exactvar=stats.t.moment(4,nu,loc=.4,scale=scale)-exactmean**2
        checks.append(dict(check='t_quadratic_mean_and_variance',df=nu,mean=float(m[0]),reference_mean=exactmean,variance=float(v[0]),reference_variance=exactvar))
        assert abs(m[0]-exactmean)<1e-12 and abs(v[0]-exactvar)<1e-12
    for shape in [-.2,0.,.2,.5]:
        values=stats.genpareto.ppf((np.arange(1000000)+.5)/1000000,shape)
        estimated,_=pwm_shape(values)
        assert abs(estimated-shape)<.003,(shape,estimated)
        checks.append(dict(check='GPD_PWM_shape_vs_known_quantiles',reference=shape,computed=estimated))
    basis=np.zeros((64,2));basis[0,0]=1.;basis[32,1]=1.
    coords=dict(center=np.r_[np.ones(32),np.zeros(32)],basis=basis,scale=np.ones(2))
    vm=VoltageMagnitude(coords,np.zeros(2),32);projection,corr,distance=vm.terms(np.zeros(2),np.eye(2)*.01)
    mean=vm.a[0]+projection[0]+corr[0];variance=distance[0]-2*vm.a[0]*corr[0]-(projection[0]+corr[0])**2
    exact_mean,exact_var=stats.rice.stats(10.,scale=.1,moments='mv')
    assert abs(mean-exact_mean)<1e-9 and abs(variance-exact_var)<1e-9
    checks.append(dict(check='voltage_magnitude_vs_Rice_distribution',computed_mean=mean,reference_mean=exact_mean,computed_variance=variance,reference_variance=exact_var))
    r.write_json(OUT/'independent_formula_check.json',checks)
    return checks

def join_results():
    s=pd.read_csv(OUT/'focus_quantities_mc_selection.csv')
    m=pd.read_csv(r.OUT/'full_mean_variance_all_quantities.csv');w=pd.read_csv(r.OUT/'tail_quantile_w1_all_quantities.csv')
    c=pd.read_csv(r.OUT/'tail_cdf_rmse_all_quantities.csv');q=pd.read_csv(r.OUT/'extreme_quantile_error_all_quantities.csv')
    keys=['情形','方法','K','电气量','类型','尾侧']
    frame=w.merge(s[['情形','电气量','尾侧','电气量说明','高斯标准化尾部延伸比_尾概率0_1百分比','尾部特征解释']],on=['情形','电气量','尾侧'],validate='m:1')
    frame=frame.merge(c[keys+['CDF_RMSE_百分点']],on=keys,validate='1:1')
    cols=['MC均值','方法均值','均值误差','均值绝对误差','MC方差','方法方差','方差相对误差','方差比_方法除以MC','MC标准差','方法标准差','方差单位']
    frame=frame.merge(m[[k for k in keys if k!='尾侧']+cols],on=[k for k in keys if k!='尾侧'],validate='1:1')
    qw=q.pivot(index=keys,columns='尾概率',values='绝对分位误差').reset_index().rename(columns={.001:'极端分位绝对误差_尾概率0_1百分比',.005:'极端分位绝对误差_尾概率0_5百分比',.01:'极端分位绝对误差_尾概率1百分比'})
    frame=frame.merge(qw,on=keys,validate='1:1');assert len(frame)==432
    frame.to_csv(OUT/'focus_quantities_three_metrics_mean_variance.csv',index=False,encoding='utf-8-sig')
    return frame

def variance_concentration():
    selected=pd.read_csv(OUT/'focus_quantities_mc_selection.csv');rows=[]
    for scene in r.SCENES:
        data=np.load(r.OLD/scene/'mc_outputs.npy',mmap_mode='r')
        for row in selected[selected.情形==r.CN[scene]].itertuples():
            j=next(j for j in range(97) if r.object_info(j)[0]==row.电气量);x=np.array(data[:,j]);square=(x-x.mean())**2
            rows.append(dict(情形=row.情形,电气量=row.电气量,最偏离均值0_1百分比场景的方差贡献=np.partition(square,-1000)[-1000:].sum()/square.sum(),单一最大偏差场景的方差贡献=float(square.max()/square.sum()),场景总数=len(x)))
    pd.DataFrame(rows).to_csv(OUT/'focus_quantities_mc_extreme_variance_share.csv',index=False,encoding='utf-8-sig')

def check_all():
    trusted=trusted_references();frame=join_results();variance_concentration()
    w=pd.read_csv(r.OUT/'tail_quantile_w1_all_quantities.csv');m=pd.read_csv(r.OUT/'full_mean_variance_all_quantities.csv')
    assert len(w)==9312 and w.W1数值收敛.all() and (w.尾部分位区间W1>=0).all()
    assert len(m)==4656 and (m.方法方差>=0).all()
    power=m[m.类型!='V'].pivot(index=['情形','K','电气量'],columns='方法',values='方法均值')
    same_mean=float(abs(power.AQGMM-power.标准TMM).max())
    # Display units are kW/kvar; check below 1e-10 MW/Mvar, not an arbitrary
    # displayed decimal count after conversion by 1000.
    assert same_mean/1000<1e-10
    p=pd.read_csv(OUT/'pdf_display_range_check.csv');assert p.最小区间概率差.min()>-1e-7
    figures=pd.read_csv(OUT/'figure_index.csv');assert len(figures)==108
    imgs=[]
    for row in figures.itertuples():
        with Image.open(OUT/row.PNG) as im:
            width,height=im.size;dpi=im.info.get('dpi',(0,0))
            assert width>2000 and height>1000 and min(dpi)>299
            imgs.append(dict(file=row.PNG,width=width,height=height,dpi=dpi))
        pdf=OUT/row.PDF
        content=pdf.read_bytes();assert content.startswith(b'%PDF') and b'%%EOF' in content[-1024:]
    manifest=pd.read_csv(r.ROOT/'computation_records/frozen_archive_sha256.csv');bad=[]
    for row in manifest.itertuples():
        if r.digest(r.ROOT/row.path)!=row.SHA256:bad.append(row.path)
    assert not bad,bad
    errors=[]
    for path in (OUT/'computation_records').glob('*moments.json'):
        for check in json.loads(path.read_text())['checks']:errors.append(dict(file=path.name,**check))
    result=dict(MC_quantities=388,MC_voltage_branch_quantities=384,MC_tails=776,
        selected_quantities=36,selection='MC only; top3 by robust 0.1% extension ratio within each scene and V/P/Q type',
        W1_rows=len(w),W1_all_numerically_converged=bool(w.W1数值收敛.all()),moment_rows=len(m),selected_combined_rows=len(frame),
        same_GMM_TMM_quadratic_mean_max_display_unit_difference=same_mean,
        figures=len(figures),full_and_tail_method_PDF_figures=72,MC_comparison_figures=36,
        minimum_PDF_bin_probability=p.最小区间概率差.min(),negative_PDF_mass_intervals_over_1e7=int((p.最小区间概率差 < -1e-7).sum()),
        frozen_archive_files=len(manifest),frozen_archive_hash_mismatches=bad,new_injection_AC_or_fitting=False,
        statistical_limit='Fixed-threshold tail-shape intervals are pointwise exploratory, not simultaneous asymptotic heavy-tail proofs',
        moment_note='Actual full saved-model laws; voltage magnitude, not squared magnitude; no finite model-draw moment estimates',
        numerical_checks=trusted,moment_refinement_checks=errors,image_metadata=imgs)
    r.write_json(OUT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['moment_refinement_checks','image_metadata','numerical_checks']},ensure_ascii=False),flush=True)

if __name__=='__main__':check_all()
