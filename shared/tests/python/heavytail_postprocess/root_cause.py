"""Read-only causal checks of saved voltage laws; no input/AC/fitting changes.

The fixed-Q comparison is a diagnostic enforcing the known terminal load only.
It is not a replacement probabilistic power-flow method.
"""
import recompute as r
import numpy as np
import pandas as pd
from scipy import stats, special

OUT=r.ROOT/'diagnostics/root_cause_check'
N=2**20

def psdroot(c):
    a,b=np.linalg.eigh(c)
    return b*np.sqrt(np.maximum(a,0))

def draw(model,kind,n,rng):
    if kind in ('GMM','TMM'):
        ix=rng.choice(len(model['weights']),n,p=model['weights'])
        z=np.empty((n,len(model['means'][0])))
        for j in range(len(model['weights'])):
            mask=ix==j; count=mask.sum()
            y=rng.standard_normal((count,z.shape[1]))@psdroot(model['scatter'][j]).T
            if kind=='TMM' and np.isfinite(model['df'][j]):
                nu=model['df'][j];y*=np.sqrt(nu/rng.chisquare(nu,count))[:,None]
            z[mask]=model['means'][j]+y
        return z
    cs=model['shifted'];p=np.array([c['weight'] for c in cs]);p/=p.sum()
    ix=rng.choice(len(cs),n,p=p);z=np.empty((n,len(model['Q1'])))
    for j,c in enumerate(cs):
        mask=ix==j; count=mask.sum();u=rng.uniform(size=count)
        strengths=np.stack([np.ones(count) if t.gauss else t.nu/stats.chi2.isf(u*t.pmax,t.nu)/t.mean for t in c['trunc']],axis=1)
        s=strengths[:,c['gid']]
        y=c['mu']+c['shift']*(s-1)+(rng.standard_normal((count,len(c['mu'])))@psdroot(c['scatter']).T)*np.sqrt(s)
        w=c['intercept']+y@c['coefficient'].T+rng.standard_normal((count,model['Q2'].shape[1]))@psdroot(c['residual']).T
        z[mask]=y@model['Q1'].T+w@model['Q2'].T
    return z

def cdf(x,t):return np.searchsorted(np.sort(x),t,side='right')/len(x)

def check_power_balance():
    ns=r.numerical_functions();base=ns['base'];line=base.BRANCH[16]
    impedance=line[2]+1j*line[3];qload=base.BUS[17,3]
    rows=[];points=[];checks=[]
    for scene in r.SCENES:
        d=r.OLD/scene;models=r.read_models(d/'models_K2.pkl')
        co=dict(np.load(d/'voltage_coordinates.npz'))
        T=(co['basis'][:32]+1j*co['basis'][32:])*co['scale']
        v0=co['center'][:32]+1j*co['center'][32:]
        ref=np.load(d/'mc_outputs.npy',mmap_mode='r');qref=np.array(ref[:,80])
        qloss_ref=line[3]/base.BASE*(ref[:,48]**2+ref[:,80]**2)/ref[:,15]**2
        checks.append(dict(scene=scene,MC_terminal_balance_max_Mvar=float(abs(qref-qload-qloss_ref).max())))
        thresholds=np.quantile(qref,1-r.LEVELS);mc=cdf(qref,thresholds)
        for mi,(name,kind) in enumerate(r.MODELS.items()):
            rng=np.random.default_rng(202610260001+mi);qs=[];qc=[];ev=[]
            for start in range(0,N,32768):
                z=draw(models[name],kind,min(32768,N-start),rng)
                va=v0[15]+z@T[15];vb=v0[16]+z@T[16];current=(va-vb)/impedance
                q=base.BASE*(va*np.conj(current)).imag
                loss=base.BASE*line[3]*abs(current)**2
                # Q17 = line reactive loss - node18 net reactive injection.
                # The diagnostic restores only the known Q18=-Qload18.
                corrected=qload+loss;error=q-corrected
                qs.append(q);qc.append(corrected);ev.append(error)
            q=np.concatenate(qs);corrected=np.concatenate(qc);error=np.concatenate(ev)
            f=cdf(q,thresholds);fc=cdf(corrected,thresholds)
            saved=np.load(d/f'CDF_{name}_K2.npz')['CDF'][80,50:100]
            qq=np.quantile(q,r.PROBS);cq=np.quantile(corrected,r.PROBS);rq=np.quantile(qref,r.PROBS)
            rows.append(dict(scene=scene,model=name,K=2,n=N,
                MC_Q17_sd_kvar=float(qref.std()*1000),
                model_Q17_sd_kvar=float(q.std()*1000),
                fixed_Q_diagnostic_sd_kvar=float(corrected.std()*1000),
                spurious_Q18_RMS_kvar=float(np.sqrt(np.mean(error**2))*1000),
                spurious_Q18_variance_over_Q17_variance=float(error.var()/q.var()),
                covariance_term_kvar2=float(2*np.cov(corrected,error,ddof=0)[0,1]*1e6),
                original_tail_CDF_RMSE_pp=float(np.sqrt(np.mean((f-mc)**2))*100),
                saved_analytic_tail_CDF_RMSE_pp=float(np.sqrt(np.mean((saved-mc)**2))*100),
                fixed_Q_diagnostic_tail_CDF_RMSE_pp=float(np.sqrt(np.mean((fc-mc)**2))*100),
                sample_vs_saved_analytic_max_CDF_pp=float(abs(f-saved).max()*100),
                MC_q999_kvar=float(rq[-1]*1000),model_q999_kvar=float(qq[-1]*1000),
                fixed_Q_diagnostic_q999_kvar=float(cq[-1]*1000),
                original_q999_abs_error_kvar=float(abs(qq[-1]-rq[-1])*1000),
                fixed_Q_diagnostic_q999_abs_error_kvar=float(abs(cq[-1]-rq[-1])*1000),
                original_fraction_below_Qload=float(np.mean(q<qload)),
                diagnostic_fraction_below_Qload=float(np.mean(corrected<qload))))
            for i,p in enumerate(r.LEVELS):
                # Conservative pointwise binomial approximation for the diagnostic CDF.
                se=np.sqrt(fc[i]*(1-fc[i])/N)
                points.append(dict(scene=scene,model=name,tail_probability=p,threshold_kvar=thresholds[i]*1000,
                    MC_CDF=mc[i],model_sample_CDF=f[i],model_saved_analytic_CDF=saved[i],
                    diagnostic_CDF=fc[i],diagnostic_CDF_95_halfwidth_pp=1.96*se*100))
            print(scene,name,'RMSE',rows[-1]['original_tail_CDF_RMSE_pp'],'->',rows[-1]['fixed_Q_diagnostic_tail_CDF_RMSE_pp'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'branch17_Q_physical_condition_compare.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(points).to_csv(OUT/'branch17_Q_per_threshold_cdf.csv',index=False,encoding='utf-8-sig')
    r.write_json(OUT/'mc_end_reactive_balance_check.json',checks)

def check_df_directions():
    rows=[];margins=[]
    for scene in r.SCENES:
        d=r.OLD/scene;co=dict(np.load(d/'voltage_coordinates.npz'));a=np.load(d/'estimation.npz')
        v=a['voltage'];st=np.c_[v[:,1:].real,v[:,1:].imag]
        x=(st-co['center'])@co['basis']/co['scale']
        models=r.read_models(d/'models_K2.pkl');g=models['AQGMM'];t=models['standard_TMM']
        log=[]
        for pi,mu,C in zip(g['weights'],g['means'],g['scatter']):
            log.append(np.log(pi)+stats.multivariate_normal.logpdf(x,mean=mu,cov=C))
        log=np.array(log).T;resp=np.exp(log-special.logsumexp(log,axis=1)[:,None])
        for j in range(2):
            w=resp[:,j];w/=w.sum()
            for dimension in sorted(set([4,8,10,12,16,x.shape[1]])):
                y=x[:,:dimension];mu=w@y;e=y-mu;C=e.T@(e*w[:,None])
                delta=np.einsum('ni,ni->n',e@np.linalg.inv(C),e)
                b=float(w@(delta**2)/(dimension*(dimension+2)))
                rows.append(dict(scene=scene,component=j,dimensions=dimension,
                    smallest_retained_direction_sd_pu=co['scale'][dimension-1],radial_fourth_ratio=b,
                    implied_unfloored_df=(4*b-2)/(b-1) if b>1 else np.inf,
                    saved_full_dimension_df=t['df'][j]))
            # Real voltage at18 dominates its magnitude response near this operating point.
            direction=co['basis'][16]*co['scale'];e=(x-g['means'][j])@direction
            variance=w@(e**2);fourth=w@(e**4)/variance**2;third=w@(e**3)/variance**1.5
            nu=t['df'][j]
            margins.append(dict(scene=scene,component=j,quantity='real_V18',weight=g['weights'][j],
                empirical_weighted_skew=third,empirical_weighted_fourth=fourth,
                standard_TMM_fourth=3*(nu-2)/(nu-4) if np.isfinite(nu) else 3.,
                standard_TMM_component_skew=0.,df=nu))
    pd.DataFrame(rows).to_csv(OUT/'dof_retained_voltage_direction_check.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(margins).to_csv(OUT/'bus18_V_component_skew_fourth_moment.csv',index=False,encoding='utf-8-sig')

def check_voltage_df_ablation():
    """Training-only diagnostic: set df from V18 real-part fourth moments.

    Keep the original GMM partitions, weights, means and covariance unchanged.
    This single-bus diagnostic is not proposed as a network-wide estimator.
    """
    ns=r.numerical_functions();rows=[];detail=[]
    margins=pd.read_csv(OUT/'bus18_V_component_skew_fourth_moment.csv')
    for scene in r.SCENES:
        d=r.OLD/scene;co=dict(np.load(d/'voltage_coordinates.npz'))
        models=r.read_models(d/'models_K2.pkl');g=models['AQGMM']
        model={key:np.array(value,copy=True) for key,value in g.items()}
        for j in range(2):
            b=float(margins[(margins.scene==scene)&(margins.component==j)].empirical_weighted_fourth.iloc[0])/3
            nu=(4*b-2)/(b-1) if b>1 else np.inf
            model['df'][j]=nu;model['scatter'][j]=g['scatter'][j]*((nu-2)/nu if np.isfinite(nu) else 1.)
            detail.append(dict(scene=scene,component=j,original_df=models['standard_TMM']['df'][j],diagnostic_df=nu))
        full=ns['quadratic_forms'](co);forms={key:full[key][16:17] for key in ['A','b','c']}
        tm=ns['terms'](model,'TMM',forms)[0];kernel=r.CDFKernel(tm)
        ref=np.load(d/'mc_outputs.npy',mmap_mode='r');voltage=np.array(ref[:,16])
        thresholds=np.quantile(voltage,1-r.LEVELS);mc=cdf(voltage,thresholds)
        f=kernel.cdf(thresholds**2)
        qref=float(np.quantile(voltage,.999))
        root=r.optimize.brentq(lambda y:kernel.cdf([y*y])[0]-.999,.8,1.8,xtol=1e-13)
        refined=r.CDFKernel(tm,cutoff_multiplier=2.)
        f2=refined.cdf(thresholds**2)
        root2=r.optimize.brentq(lambda y:refined.cdf([y*y])[0]-.999,.8,1.8,xtol=1e-13)
        original=np.load(d/'CDF_standard_TMM_K2.npz')['CDF'][16,50:100]
        rows.append(dict(scene=scene,K=2,quantity='V18_upper',
            original_CDF_RMSE_pp=float(np.sqrt(np.mean((original-mc)**2))*100),
            diagnostic_CDF_RMSE_pp=float(np.sqrt(np.mean((f2-mc)**2))*100),
            MC_q999_pu=qref,diagnostic_q999_pu=root2,diagnostic_q999_abs_error_pu=abs(root2-qref),
            quadrature_refinement_max_CDF_difference=float(abs(f2-f).max()),
            quadrature_refinement_q999_difference_pu=abs(root-root2)))
        print('V18 diagnostic',rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'bus18_V_error_after_dof_change.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(detail).to_csv(OUT/'bus18_V_dof_change.csv',index=False,encoding='utf-8-sig')

def check_balance_cancellation():
    """Trace the lost voltage response cancellation within each saved component."""
    ns=r.numerical_functions();base=ns['base'];line=base.BRANCH[16]
    impedance=line[2]+1j*line[3];qload=base.BUS[17,3];rows=[]
    for scene in r.SCENES:
        d=r.OLD/scene;co=dict(np.load(d/'voltage_coordinates.npz'))
        v=np.load(d/'estimation.npz')['voltage'];st=np.c_[v[:,1:].real,v[:,1:].imag]
        x=(st-co['center'])@co['basis']/co['scale'];models=r.read_models(d/'models_K2.pkl');g=models['AQGMM']
        log=np.array([np.log(pi)+stats.multivariate_normal.logpdf(x,mean=mu,cov=C) for pi,mu,C in zip(g['weights'],g['means'],g['scatter'])]).T
        resp=np.exp(log-special.logsumexp(log,axis=1)[:,None])
        T=(co['basis'][:32]+1j*co['basis'][32:])*co['scale'];v0=co['center'][:32]+1j*co['center'][32:]
        h=(T[15]-T[16])/impedance;i0=(v0[15]-v0[16])/impedance;k=base.BASE*line[3]
        A_loss=k*np.real(np.outer(np.conj(h),h));b_loss=2*k*np.real(np.conj(i0)*h);c_loss=qload+k*abs(i0)**2
        forms=ns['quadratic_forms'](co);A=forms['A'][80]-A_loss;b=forms['b'][80]-b_loss;c=forms['c'][80]-c_loss
        for j in range(2):
            w=resp[:,j];w/=w.sum();mu=g['means'][j];C=g['scatter'][j];e=x-mu
            linear=e@(2*A@mu+b);quadratic=np.einsum('ni,ij,nj->n',e,A,e)
            lc=linear-w@linear;qc=quadratic-w@quadratic
            vl=w@(lc**2);vq=w@(qc**2);cv=w@(lc*qc)
            trace=np.trace(A@C);tr2=np.trace(A@C@A@C);linvar=(2*A@mu+b)@C@(2*A@mu+b)
            nu=models['standard_TMM']['df'][j];factor=(nu-2)/(nu-4) if np.isfinite(nu) else 1.
            gaussian_quadvar=2*tr2;t_quadvar=factor*(2*tr2+trace**2)-trace**2
            value=linear+quadratic+mu@A@mu+b@mu+c
            rows.append(dict(scene=scene,component=j,weight=g['weights'][j],df=nu,
                actual_weighted_voltage_response_correlation=float(cv/np.sqrt(vl*vq)),
                actual_linear_sd_kvar=float(np.sqrt(vl)*1000),actual_quadratic_sd_kvar=float(np.sqrt(vq)*1000),
                actual_balance_RMS_kvar=float(np.sqrt(w@(value**2))*1000),
                symmetric_GMM_or_TMM_linear_quadratic_covariance=0.,
                GMM_implied_balance_sd_kvar=float(np.sqrt(max(0,linvar+gaussian_quadvar))*1000),
                standard_TMM_implied_balance_sd_kvar=float(np.sqrt(max(0,linvar+t_quadvar))*1000),
                Gaussian_linear_variance_kvar2=float(linvar*1e6),Gaussian_quadratic_variance_kvar2=float(gaussian_quadvar*1e6),
                TMM_quadratic_variance_kvar2=float(t_quadvar*1e6)))
    pd.DataFrame(rows).to_csv(OUT/'fixed_reactive_voltage_response_cancellation.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

def write_report():
    q=pd.read_csv(OUT/'branch17_Q_physical_condition_compare.csv');v=pd.read_csv(OUT/'bus18_V_error_after_dof_change.csv')
    qt=pd.read_csv(r.OUT/'extreme_quantile_error_all_quantities.csv')
    lines=['# 重尾电气量误差的根因检查','',
        '本次固定原注入、基础负荷、相关系数和独立MC结果，仅读取已有模型。全部新增文件位于本独立副本中，原项目未写入。主诊断范围为四种注入情形、K=2、节点18电压和支路17（17→18）无功。','',
        '## 1 已定位的主要原因','',
        '**支路无功的大误差主要来自拟合电压分布未保持固定无功平衡。分组t另有已确认的参数计算问题；标准TMM还存在用整组电压的四阶统计量统一确定各方向尾部的问题。这三项需要分开处理。**','',
        '在本算例中，节点18为末端节点，无功负荷固定为40 kvar，电源无功注入为零。精确支路关系为：','',
        '`支路17首端无功 = 节点18无功负荷 + 支路17无功损耗`。','',
        '因此首端无功不能低于40 kvar，随机变化完全来自线路无功损耗。拟合电压分布虽然可以保持均值和协方差，却没有强制每个电压组合都满足该节点的固定无功平衡；末端电压与线路电流之间偏离了原交流潮流对应的关系。','',
        'Pareto情形下，MC支路无功标准差只有0.144 kvar；三个拟合模型却分别有约32.3%、28.7%、32.8%的结果低于40 kvar。这是直接违反当前运行条件的结果，不能解释为真实的重尾现象。','',
        '### 已追查到的电压分布构造原因','',
        '进一步把节点18的无功平衡偏差按交流潮流关系展开为两部分：随电压变化的一阶项，以及随电压变化的二阶项。真实潮流中，这两部分必须随同一组电压配合变化、互相抵消，才能保持无功负荷不变。','',
        '四种情形、K=2的全部8个分量中，原潮流场景在分量权重下的一阶项、二阶项相关系数都接近−1。Pareto第1分量，两项标准差均为1.733 kvar，但抵消后的无功平衡偏差均方根只有约0.000017 kvar。','',
        '**当前GMM和标准TMM的单个分量均关于中心对称；在各分量均值附近，一阶项与二阶项的协方差必为零。** 原因是一阶项乘二阶项包含三阶电压变化，而中心对称分量的三阶中心矩为零。当前构造由此丢失了真实潮流所要求的抵消关系。各节点电压本身可以拟合得较接近，末端支路无功仍会发生明显偏差。','',
        '在保持均值、协方差相同的前提下，t分量降低自由度会增大二阶项的波动，却不会恢复上述抵消关系。对Pareto第1分量，从已保存参数直接计算：GMM造成的节点18无功平衡偏差标准差为2.009 kvar，标准TMM为4.742 kvar。这里是分量内分布的解析矩结果，不依赖电气量CDF积分，也不依赖本次随机抽样。','',
        '这给出了“已发生重尾的支路无功，t模型反而可能更差”的具体原因：加厚了未满足固定无功关系的电压波动。分量数增加可以改善局部近似，但本次不能据此承诺某一分量数必然解决。分组t虽引入非对称变化，仍需检查能否保持这些支路两端的联合关系；已有参数错误必须先修复。详见电压响应抵消.csv。','',
        '## 2 功率平衡对照及误差','',
        '从每个已保存电压模型独立生成1048576个电压组合，先计算原支路无功，再保留同一组线路电流及其无功损耗，仅将节点18无功恢复为已知的−40 kvar。输入、电压模型参数和MC均未改变。这项对照只用于判断误差来源，不能作为已修复的新方法。两列CDF-RMSE均由本次同一批电压组合计算，单位为百分点；正式原始解析结果另列在CSV中。','',
        '| 注入情形 | 电压模型 | 原上尾CDF-RMSE | 固定无功平衡对照 | 原99.9%分位误差/kvar | 对照分位误差/kvar |',
        '|---|---|---:|---:|---:|---:|']
    for _,x in q.iterrows():
        lines.append(f"| {r.CN[x.scene]} | {r.CN[x.model]} | {x.original_tail_CDF_RMSE_pp:.3f} | {x.fixed_Q_diagnostic_tail_CDF_RMSE_pp:.3f} | {x.original_q999_abs_error_kvar:.3f} | {x.fixed_Q_diagnostic_q999_abs_error_kvar:.3f} |")
    lines += ['',
        '对照后，四种情形、三个模型的支路17上尾CDF-RMSE均降至1个百分点以内。这说明固定无功关系未保持是该支路误差的主要来源；剩余的电压、电流分布误差仍然存在。标准TMM和AQGMM也出现此问题，因此不能全部归因于分组t的参数错误。','',
        'MC支路功率与上述精确关系的最大差小于7.3×10⁻¹¹ Mvar。随机抽取的原模型CDF与已存解析CDF最大差为0.113个百分点；对照CDF的逐阈值近似95%区间半宽最大为0.048个百分点。它们远小于本次约15—31个百分点的误差下降。区间只描述此次模型抽样，不包括参数估计误差。','',
        '## 3 节点电压为何仍有偏差','',
        '标准TMM的自由度来自同一分量中整组电压的四阶统计量。随后，同一个自由度被用于该分量内所有电压方向。这保持了特定的整体统计量，但没有逐节点保持偏度和四阶矩。','',
        '例如，Pareto、K=2的第1分量中，节点18电压实部的加权样本偏度为2.119，普通标准化四阶矩为12.224；当前标准t分量的偏度为0，四阶矩却为48.600。第2分量的四阶矩为2.142，模型为5.005。这里说的是单个对称t分量；多个分量组成的TMM整体仍可以有偏度。电压实部用于追查定参问题，正式电压误差仍按电压幅值评价。','',
        '因此，即使全局四阶关系计算正确，也可能同时出现某些节点尾部过宽、另一些方向尾部不足。当前结果不能简单归结为“输入厚尾，所以任意t定参都应更优”。','',
        '为检验只改自由度是否足够，另用参数估计场景中节点18电压实部的四阶矩确定自由度，保持分量归属、权重、均值和协方差不变。没有利用MC选择参数。以下为解析电压CDF，积分范围加倍后的最大CDF变化小于2×10⁻¹⁵：','',
        '| 注入情形 | 原标准TMM上尾CDF-RMSE/百分点 | 单节点定参对照 | 原99.9%分位误差/p.u. | 对照分位误差/p.u. |',
        '|---|---:|---:|---:|---:|']
    for _,x in v.iterrows():
        old=qt[(qt['情形']==r.CN[x.scene])&(qt['方法']=='标准TMM')&(qt.K==2)&(qt['电气量']=='V18')&(abs(qt['分位水平']-.999)<1e-10)].iloc[0]
        lines.append(f"| {r.CN[x.scene]} | {x.original_CDF_RMSE_pp:.3f} | {x.diagnostic_CDF_RMSE_pp:.3f} | {old['绝对分位误差']:.5f} | {x.diagnostic_q999_abs_error_pu:.5f} |")
    lines += ['',
        '**单改自由度没有同时改善全部指标。** 例如对数逻辑斯蒂的极端分位误差明显下降，但尾部CDF-RMSE增大；Pareto的CDF-RMSE下降，但极端分位误差略增。这排除了“只把自由度改一下即可解决”的解释，也说明单一四阶统计量不足以同时约束尾部概率与极端分位值。该对照不是推荐的全网定参方案。','',
        '对Pareto第1分量，仅使用前16个电压方向时自由度约4.129，使用全部21个方向时约4.132。因此，不能将该分量的异常完全归因于最后几个微小方向的数值误差。其余方向对其他情形的影响见自由度来源.csv，尚未独立区分非线性响应与数值误差。','',
        '## 4 分组t另有的确定性实现问题','',
        '已有独立参数复核确认：有限场景最大值被当作无界注入的强度上限；2000个均匀分位中点未准确计算强度矩；散布矩阵正定化后未重新保持目标协方差；加入均值移动后仍沿用对称t的四阶定参关系。','',
        'Pareto、K=2、第1分量、第1组：强度二阶矩实际比目标高41.6%；最终电压组的四阶统计量为目标的4.71倍。两个分量的原电压协方差相对差约14.4%、5.9%。这些是参数构造与当前文稿要求不一致的实现问题，不能用它们的结果判断理论方法是否有效。证据见../../audit_theory/README_zh.md。','',
        '## 5 理论结论和后续修正边界','',
        '现有解析计算回答的是：给定某个电压联合分布，如何求电气量分布。当前检查没有发现该解析关系导致上述大误差。真正需要补齐的是：从实际潮流电压构造联合分布后，是否仍保持本算例固定负荷对应的功率平衡，以及是否正确保留目标电气量的尾部。','',
        '下一步应固定注入、负荷、相关系数和MC，先修正并验收分组t的参数关系；再处理电压联合分布的固定P/Q约束。可研究在随机注入空间保留给定分布、由潮流关系计算电压，或在电压分布构造中保留运行约束；两种方向均会影响当前解析框架，不能把本次单支路对照当作全网修复。节点电压方面还需处理分量内偏态和不同节点尾部差异，不能只调整一个全局自由度。','',
        '本次已定位支路17的大误差主因，并确认分组t的参数实现错误；尚未证明修正后TMM对所有已筛选重尾电气量均优于GMM。Q17的明显上尾和V18的尾部延伸由原MC筛选，未按方法优劣筛选。有限样本的偏度、峰度和尾部延伸不等于严格的渐近重尾证明。','',
        '## 6 复算和文件','',
        '- 完整复算：在独立副本根目录执行 `python3 -B ../../../tests/python/heavytail_postprocess/root_cause.py`。不启动MATLAB、不求新潮流、不拟合新电压分布。','- branch17_Q_physical_condition_compare.csv：12组模型及功率平衡对照；branch17_Q_per_threshold_cdf.csv：600个逐阈值对照。','- bus18_V_error_after_dof_change.csv、bus18_V_dof_change.csv：单节点定参对照及数值收敛。','- bus18_V_component_skew_fourth_moment.csv、dof_retained_voltage_direction_check.csv：逐分量和电压方向的定参检查。','- mc_end_reactive_balance_check.json、verification.json：MC关系检查及冻结文件SHA256核验。','']
    (OUT/'README.md').write_text('\n'.join(lines))
    archive=pd.read_csv(r.ROOT/'computation_records/frozen_archive_sha256.csv');bad=[]
    for _,x in archive.iterrows():
        if r.digest(r.ROOT/x['path'])!=x.SHA256:bad.append(x['path'])
    r.write_json(ROOT/'computation_records/original_metric_recompute_check.json',dict(frozen_files=len(archive),mismatched_files=bad,
        original_archive_unchanged=not bad,random_voltage_draws_per_saved_model=N,
        injection_or_MC_changed=False,new_power_flow=False,new_voltage_fit=False,
        diagnostics='K2 terminal-Q balance and training-only V18 component-df ablation',
        source_script_SHA256=r.digest(r.ROOT/'../../../tests/python/heavytail_postprocess/root_cause.py')))
    assert not bad,bad

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    import sys
    if '--voltage-only' in sys.argv:check_voltage_df_ablation()
    elif '--cancellation-only' in sys.argv:check_balance_cancellation()
    elif '--report-only' in sys.argv:write_report()
    else:check_df_directions();check_power_balance();check_voltage_df_ablation();check_balance_cancellation();write_report()
