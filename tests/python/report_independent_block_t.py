"""Compact descriptive report, retaining every object and both references."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from independent_block_t_fit import OUT,SRC,geometry
from independent_block_t_numeric import specs,projected_parts,decompose


def table(frame):
 columns=list(frame.columns);f=lambda z:f'{z:.7g}' if isinstance(z,(float,np.floating)) else str(z)
 return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+''.join('| '+' | '.join(f(z) for z in row)+' |\n' for row in frame.itertuples(index=False,name=None))


def run():
 errors=pd.read_csv(OUT/'per_object_errors_and_moments.csv');q=pd.read_csv(OUT/'extreme_quantiles.csv');checks=pd.read_csv(OUT/'integration_audit.csv');fit=json.load(open(OUT/'selected_models.json'));main=errors[errors.reference=='independent_AC'];focus=['V18','V33','P1','P16','P32','Q17','loss'];models=['AQGMM','Standard_TMM','Block_TMM'];short=['G','S','B']
 attempts=[]
 for p in sorted((OUT/'fits').glob('*.json')):
  r=json.load(open(p));attempts.append({key:r.get(key) for key in ['family','K','partition','initial_df','status','iterations','selection_ll','train_ll','df','weights','seconds','floor_active_max','effective_counts','parameter_source','old_parameters_read']})
 for r in attempts:
  for key in ['df','weights','effective_counts']:r[key]=json.dumps(r[key])
 pd.DataFrame(attempts).to_csv(OUT/'all_fit_attempts.csv',index=False)
 text='# 8站直接光伏出力：独立估计Block_TMM新一轮结果\n\n固定8站各0.60MW、原负荷、219/73/73天及原映射电压/3504独立AC解；原模型与旧结果保留。本轮没有新AC求解或新源荷场景。G=AQGMM，S=标准多元t混合，B=单t块+独立Gaussian条件残差的Block_TMM。固定8/56维分块。三方法从原电压分别初始化，按自身联合似然更新，仅按selection电压似然选初值，不使用GMM计算结果估计t模型。\n\n'
 ft=pd.DataFrame([dict(K=r['K'],model=r['family'],nu=json.dumps(r['df']),status=r['status'],selection_LL=r['selection_ll'],iterations=r['iterations']) for r in fit]);text+='## 独立估计与选择\n\n'+table(ft)+'\n\n全部尝试的对数似然、运行时间、下限、有效分量数和来源见all_fit_attempts.csv及fits/*.json，全部9个选中模型保存为根目录NPZ。\n'
 text+='\n## 左右尾CDF-RMSE与概率区间截取W1\n\nCDF-RMSE为原始概率误差的100倍，即百分点；W1为对应物理单位，V为p.u.、P/网损为MW、Q为MVAr。每个格为左/右两侧数值，较小表示该指标更接近独立AC，不代表全局优劣。\n\n'
 rows=[]
 for k in [1,2,4]:
  for name in focus:
   rr=dict(K=k,object=name)
   for model,ab in zip(models,short):
    v=main[(main.K==k)&(main.object==name)&(main.model==model)].iloc[0];rr[ab+'_CDF_RMSE_pp']=f'{v.left_CDF_RMSE_pp:.6g} / {v.right_CDF_RMSE_pp:.6g}';rr[ab+'_W1']=f'{v.left_trimmed_W1:.6g} / {v.right_trimmed_W1:.6g}'
   rows.append(rr)
 text+=table(pd.DataFrame(rows))
 text+='\nW1_[a,b]=integral_a^b |Qmodel(p)-QAC(p)|dp/(b-a)。它是各自按概率区间截取并归一分布之间的W1，不是固定物理阈值条件分布W1，也不是全分布W1。窗口为左[.01,.05]、右[.95,.99]；AC反函数用ceil(Np)-1阶统计量，按所有跳点分段，另在绝对差变号处细分，Gauss阶数加密核对。未归一积分、W1及数值差分别保存。CDF-RMSE阈值和下表q01/q99仍使用原linear经验分位，实际经验CDF右连续。\n'
 for prob in [.01,.99,.005,.995]:
  rows=[]
  for k in [1,2,4]:
   for name in focus:
    subset=q[(q.K==k)&(q.object==name)&(q.reference=='independent_AC')&(q.probability==prob)];row=dict(K=k,object=name,unit=subset.iloc[0].unit,AC=subset.iloc[0].reference_quantile)
    for model,ab in zip(models,short):
     v=subset[subset.model==model].iloc[0];row[ab+'_value']=v.model_quantile;row[ab+'_signed_error']=v.signed_error;row[ab+'_absolute_error']=v.absolute_error
    rows.append(row)
  text+=f'\n## q{prob:g}及物理量误差\n\n'+table(pd.DataFrame(rows))
 rows=[]
 for k in [1,2,4]:
  for name in focus:
   for model in models:
    v=main[(main.K==k)&(main.object==name)&(main.model==model)].iloc[0];rows.append(dict(K=k,object=name,model=model,AC_mean=v.reference_mean,model_mean=v.model_mean,mean_error=v.mean_signed_error,AC_var=v.reference_variance,model_var=v.model_variance,variance_relative_error=v.variance_relative_error))
 text+='\n## 均值与方差\n\n方差使用物理单位平方。电压为|V|幅值；V²均值另使用精确二次型矩。功率矩用精确逆伽马一二阶矩，而非有限尺度节点的矩。\n\n'+table(pd.DataFrame(rows))
 text+='\n## 数值诊断与适用范围\n\n独立3504日期为主要比较；10512估计日期的辅助结果单独标记reference=estimation_dates，不代替独立结果。两组日期本身有出力统计差别，均保留。全部97对象逐模型、逐K、逐指标保存，不以全网平均误差给出总排名。近4.1的自由度可使极少概率的远尾对精确全局方差产生很大贡献，即使常用分位区间的CDF/W1较小，全局方差仍可能很差。这是模型结果，不能用有限求积低估的二阶强度矩替代。\n\n'
 text+=f'873组反演完成。尺度加密CDF最大差{checks.strength_CDF_change.max():.7g}；PDF积分与CDF端点差最大{checks.PDF_CDF_mass_difference.max():.7g}；CDF反求残差最大{checks.maximum_inverse_residual.max():.7g}。这些是数值检查，不是严格统计置信界。完整原始CDF/PDF、CF参数和各次数值诊断云端保留。\n'
 text+='\n旧同步截断均值移动分组模型的退化说法不适用于本轮Block_TMM。旧模型要退化标准t还需要无截断、无均值移动、r=d和单共同尺度；本轮这些条件明确，r=d密度等价已验证。本轮不含多尺度t、强度均值移动或任何条件修正模型。\n\n图在figures/，包括完整范围、AC运行范围和左右尾，原始全域数据均保留。源荷注入PDF复用上级figures/PV*_active_output及bus*_net_P/Q，输入同源且未重造。Python图为诊断，正式论文图另行导出。\n'
 (OUT/'comparison_report_zh.md').write_text(text)
 # Q17 / loss decomposition and implicit node Q18, including unmodified original data.
 diag=[];geo=geometry();objects=specs(geo)
 for k in [1,2,4]:
  for model in models:
   for name in ['Q17','loss']:
    a=json.load(open(OUT/'distributions'/f'K{k}'/model/f'{name}.json'));row=dict(K=k,model=model,object=name,**a['model_variance_decomposition']);params=dict(np.load(OUT/f'{model}_K{k}.npz'));spec=objects[80 if name=='Q17' else 96];own=decompose(list(projected_parts(params,model,geo,spec)),spec,center='model');row.update({'own_model_mean_center_'+key:v for key,v in own.items()});row['shared_center']='original estimation-voltage mean';row.update({'est_'+key:v for key,v in a['estimation_variance_decomposition'].items()})
    if name=='Q17':row['implicit_Q18_mean'],row['implicit_Q18_variance']=a['implicit_Q18_mean_variance']
    diag.append(row)
 pd.DataFrame(diag).to_csv(OUT/'Q17_loss_decomposition_and_Q18.csv',index=False)
 (OUT/'report_summary.json').write_text(json.dumps(dict(status='complete',selected_models=9,distributions=873,main_rows=len(main),auxiliary_rows=len(errors)-len(main),fit_attempts=len(attempts),all_selected_converged=all(r['status']=='converged' for r in fit),no_GMM_parameter_reuse=True,original_AC_recomputed=False,scope='descriptive object-wise matched K results; no network-average ranking'),indent=2)+'\n')
 print('REPORT_COMPLETE',len(main),'independent rows',len(attempts),'fit attempts')

if __name__=='__main__':run()
