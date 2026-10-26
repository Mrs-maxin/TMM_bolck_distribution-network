"""Write the result index, field definitions and evidence-bounded interpretation."""
from pathlib import Path
import json,csv,re,shutil,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]/'result/heavytail_injection/case33bw_20261026'

ENGLISH='''scene|注入情形编号：pareto、lognormal、loglogistic、mixed
model|已保存的电压分布模型；模型名称不代表精度次序
object|电气量编号：V为节点电压，P/Q为支路首端有功/无功，loss为总有功网损
kind|电气量类型：V、P、Q、loss
quantity|本次检查的电气量或电压方向
side|评价尾侧：lower为下尾，upper为上尾
K|混合分量数
component|分量编号，从0开始
group|电源或电压方向分组编号，从0开始
weight|该分量权重，无量纲
df|标准t分量自由度，无量纲
stored_df|已保存的自由度，无量纲
saved_full_dimension_df|保留全部电压方向时已保存的自由度
original_df|改变定参方式前的自由度
diagnostic_df|仅用于原因检查的自由度，不是主结果的新模型
implied_unfloored_df|按本行四阶统计量反算、尚未施加下限的自由度
target_unclipped_df|按目标四阶统计量反算、尚未施加下限的自由度
df_floor_active|是否触及自由度下限
dimensions|保留的电压变化方向数，不是节点数
smallest_retained_direction_sd_pu|最弱保留方向的电压标准差，p.u.
radial_fourth_ratio|该组归一化电压距离的四阶统计量，无量纲，不是某个节点的普通峰度
target_radial_fourth|目标电压组四阶统计量，无量纲
actual_radial_fourth|已保存模型实际对应的电压组四阶统计量，无量纲
target_group_radial_fourth_ratio|目标电压组四阶统计量，无量纲
ratio|本表实际值与目标值之比，无量纲
empirical_weighted_fourth|按分量归属加权的节点18电压实部标准化四阶矩，未减3
empirical_weighted_skew|按分量归属加权的节点18电压实部偏度
standard_TMM_fourth|标准t分量对应的节点18电压实部标准化四阶矩，未减3
standard_TMM_component_skew|单个对称t分量的偏度，不是整个TMM的偏度
gaussian|该组是否按高斯分布处理
exact_strength_mean|独立积分得到的随机强度均值
exact_strength_second_moment|独立积分得到的随机强度二阶矩
stored_second_moment|原参数计算程序保存的随机强度二阶矩
relative_second_moment_mismatch|实际随机强度二阶矩除以目标值再减1
group_mean_max_deviation|各组随机强度均值与1的最大绝对差
voltage_U_mean_abs_max|分组后的归一化电压均值最大绝对偏差，不是p.u.
covariance_relative_Frobenius_error|电压协方差差值的Frobenius范数除以目标协方差范数
marginal_variance_relative_max|各电压方向方差的最大相对偏差
third_moment_relative_vector_error|各方向三阶中心矩差向量范数除以目标向量范数
mean_shift_scale|为使散布矩阵正定而对均值移动量使用的缩放系数
cross_covariance_scale|为使散布矩阵正定而对组间散布项使用的缩放系数
integration_refinement_covariance_difference|加密强度积分前后所得协方差的差异，用于数值检查
gaussian_direction_covariance_target_min|目标归一化电压协方差的最小特征值
remaining_excess_kurtosis_max|未进入重尾组的电压方向最大超额峰度
missing_injection_groups|未被原分组规则纳入电源组的注入编号
first_order_voltage_variance_share|一阶电压响应方向承担的电压方差比例
represented_voltage_variance_share|所选电压方向承担的电压方差比例
mean|MC均值；原单位随类型为p.u.、MW、Mvar、MW
sd|MC标准差；原单位随类型为p.u.、MW、Mvar、MW
skew|MC偏度，无量纲
excess_kurtosis|MC超额峰度，即普通峰度减3，无量纲
extension_lower|MC下尾延伸比，无量纲
extension_upper|MC上尾延伸比，无量纲
extended_lower|MC下尾延伸比是否超过既定阈值1.15，不等于严格重尾证明
extended_upper|MC上尾延伸比是否超过既定阈值1.15，不等于严格重尾证明
q001|MC的0.1%分位值；单位随类型为p.u.、MW、Mvar、MW
q999|MC的99.9%分位值；单位随类型为p.u.、MW、Mvar、MW
lower_tail_pp|历史下尾CDF平均绝对误差MAE，百分点，不是RMSE
upper_tail_pp|历史上尾CDF平均绝对误差MAE，百分点，不是RMSE
lower_signed_pp|历史下尾平均有符号概率偏差，百分点
upper_signed_pp|历史上尾平均有符号概率偏差，百分点
tail_error_pp|历史筛选电气量对应尾侧的CDF平均绝对误差MAE，百分点
full_CDF_max_pp|历史251个MC分位阈值上的全分布CDF最大绝对差，百分点；不是全分布RMSE，也不是精确KS距离
CF_end_abs|频率积分截断终点的特征函数模，无量纲
MC_CDF|MC在所列物理阈值处的累积概率，取值0—1
CDF_RMSE|尾部CDF-RMSE原始概率值；乘100才是百分点
MC_Q17_sd_kvar|MC支路17无功标准差，kvar
MC_q999_kvar|MC的99.9%分位值，kvar
MC_q999_pu|MC节点18电压幅值的99.9%分位值，p.u.
model_Q17_sd_kvar|从已保存电压分布直接计算的支路17无功标准差，kvar
fixed_Q_diagnostic_sd_kvar|使用已知固定无功条件时支路17无功标准差，kvar；只用于原因检查
spurious_Q18_RMS_kvar|已保存电压分布导致的节点18无功平衡偏差均方根，kvar
spurious_Q18_variance_over_Q17_variance|节点18无功平衡偏差方差与原支路17无功方差之比
covariance_term_kvar2|无功损耗与节点无功平衡偏差之间协方差的两倍，kvar²
original_tail_CDF_RMSE_pp|从已保存电压分布直接计算的支路17上尾CDF-RMSE，百分点
saved_analytic_tail_CDF_RMSE_pp|正式结果中已保存的支路17上尾解析CDF-RMSE，百分点
fixed_Q_diagnostic_tail_CDF_RMSE_pp|使用固定无功条件时的支路17上尾CDF-RMSE，百分点；不代表已修复模型
sample_vs_saved_analytic_max_CDF_pp|直接计算与已保存解析CDF之间的最大差，百分点
model_q999_kvar|直接计算的支路17无功99.9%分位值，kvar
fixed_Q_diagnostic_q999_kvar|使用固定无功条件时支路17无功99.9%分位值，kvar
original_q999_abs_error_kvar|原支路17无功99.9%分位值绝对误差，kvar
fixed_Q_diagnostic_q999_abs_error_kvar|使用固定无功条件时支路17无功99.9%分位值绝对误差，kvar
original_fraction_below_Qload|原支路17首端无功低于节点18固定无功负荷的概率
diagnostic_fraction_below_Qload|使用固定无功条件后支路17首端无功低于该负荷的概率
tail_probability|尾概率，原始概率值，不是百分点
threshold_kvar|用于比较CDF的支路无功阈值，kvar
model_sample_CDF|从已保存电压分布直接计算所得的CDF，原始概率值
model_saved_analytic_CDF|已保存模型解析CDF，原始概率值
diagnostic_CDF|仅用于检查固定无功条件影响的CDF，原始概率值
diagnostic_CDF_95_halfwidth_pp|该检查中直接抽取电压得到CDF的逐点近似95%区间半宽，百分点
original_CDF_RMSE_pp|原标准TMM的节点18电压上尾CDF-RMSE，百分点
diagnostic_CDF_RMSE_pp|只改变自由度后的节点18电压上尾CDF-RMSE，百分点
diagnostic_q999_pu|只改变自由度后的节点18电压幅值99.9%分位值，p.u.
diagnostic_q999_abs_error_pu|上述电压幅值99.9%分位值绝对误差，p.u.
quadrature_refinement_max_CDF_difference|积分范围加倍前后CDF最大变化，原始概率值
quadrature_refinement_q999_difference_pu|积分范围加倍前后节点18电压99.9%分位值变化，p.u.
actual_weighted_voltage_response_correlation|真实潮流的一阶、二阶电压响应在分量内的加权相关系数
actual_linear_sd_kvar|固定无功关系中一阶电压响应的标准差，kvar
actual_quadratic_sd_kvar|固定无功关系中二阶电压响应的标准差，kvar
actual_balance_RMS_kvar|两部分电压响应抵消后的固定无功平衡偏差均方根，kvar
symmetric_GMM_or_TMM_linear_quadratic_covariance|对称分量内一阶项与二阶项的协方差，理论上为0
GMM_implied_balance_sd_kvar|高斯分量导致的节点18无功平衡偏差标准差，kvar
standard_TMM_implied_balance_sd_kvar|标准t分量导致的节点18无功平衡偏差标准差，kvar
Gaussian_linear_variance_kvar2|高斯分量的一阶电压响应方差，kvar²
Gaussian_quadratic_variance_kvar2|高斯分量的二阶电压响应方差，kvar²
TMM_quadratic_variance_kvar2|标准t分量的二阶电压响应方差，kvar²
fast_scenarios|已保存快速映射电压对应的源荷场景数
SI_vs_pu_max_MVA|三相SI单位与标幺值计算线路功率的最大差，MVA
Cartesian_vs_complex_max_MVA|实虚部公式与复数公式计算线路功率的最大差，MVA
independent_vs_existing_max_MW_Mvar|独立线路公式与原程序有功/无功的最大差，MW或Mvar
branch_loss_identity_max_MVA|首末端功率和与阻抗损耗表达式的最大差，MVA
independent_quadratic_vs_complex_max_MW_Mvar|独立构造的电压二次型与复数线路公式的最大差，MW或Mvar
archived_quadratic_vs_complex_max_MW_Mvar|原电压二次型与复数线路公式的最大差，MW或Mvar
coefficient_A_max_difference|两种构造所得支路功率二次项系数最大差
coefficient_b_max_difference|两种构造所得支路功率一次项系数最大差
coefficient_c_max_difference|两种构造所得支路功率常数项最大差
projection_voltage_max_pu|保留电压方向后重构电压与原电压的最大差，p.u.
projected_fixed_Q_vs_Sf_max_Mvar|重构电压下固定无功关系与线路首端公式的最大差，Mvar
unprojected_fixed_Q_vs_Sf_max_Mvar|原快速映射电压下固定无功关系与线路首端公式的最大差，Mvar
full_fixed_Q_nodal_residual_max_Mvar|原快速映射电压的节点无功平衡最大偏差，Mvar
index|原百万场景MC中的行号，从0开始
power_residual_pu|独立交流潮流的最大节点功率平衡残差，p.u.
voltage_vs_saved_MC_max_pu|独立交流潮流电压与原MC的最大差，p.u.
branch_vs_saved_MC_max_MW_Mvar|独立交流潮流支路功率与原MC的最大差，MW或Mvar
loss_vs_saved_MC_MW|独立交流潮流总有功损耗与原MC之差的绝对值，MW
fixed_Q_vs_Sf_max_Mvar|同一独立交流潮流电压下两种无功表达式的最大差，Mvar
n|本次直接分布检查抽取的电压组合数；不等于新增MC交流潮流场景数
max_direct_vs_analytic_CDF_pp|同一电压分布下直接计算与解析计算的最大CDF差，百分点
direct_vs_analytic_CDF_RMSE_pp|两种计算方式CDF差的RMSE，百分点；不是相对MC的模型精度
simultaneous_DKW_95_bound_pp|直接电压抽取对应的同时95%分布差界，百分点
within_DKW_95_bound|直接计算与解析CDF之差是否落在所列同时界内
path|冻结存档中相对本文件夹根目录的路径
bytes|文件字节数
SHA256|文件SHA256校验值
AQGMM|在此表指标下AQGMM的数值，具体指标及单位见文件名和本目录说明
'''
DEFINITIONS=dict(line.split('|',1) for line in ENGLISH.strip().splitlines())
DEFINITIONS.update({
 'j':'原MC电气量列号，从0开始：0—31为V2—V33，32—63为P1—P32，64—95为Q1—Q32，96为网损',
 'unit':'电气量汇报单位：p.u.、kW或kvar',
 'scale':'原单位换算到汇报单位的系数：电压为1，MW至kW或Mvar至kvar为1000',
 'variance':'完整百万场景MC的经验方差，使用1/N，单位为原电气量单位的平方',
 'iqr':'MC的75%分位值减25%分位值，原电气量单位',
 'PNG':'相对尾部分布目录的PNG预览图路径',
 'PDF':'相对尾部分布目录的PDF图文件路径；图中PDF指概率密度函数',
 '尾部分位区间W1':'同一尾部分位区间内方法与MC物理分位值绝对差的平均积分；单位见单位列；不是完整W1',
 'W1除以MC四分位距':'尾部分位区间W1除以MC四分位距，无量纲',
 'W1积分加密差':'前后两档积分所得W1的绝对差，物理单位见单位列',
 'W1积分阶数':'最后采用的每个物理子区间的Gauss–Legendre积分阶数',
 'W1数值收敛':'积分加密差不超过1e-4倍MC四分位距加1e-3倍当前W1；不是模型精度通过',
 '概率区间下限':'W1所比较的CDF概率区间左端；下尾0.001，上尾0.95',
 '概率区间上限':'W1所比较的CDF概率区间右端；下尾0.05，上尾0.999',
 '方法区间左端':'方法在W1概率区间左端的物理分位值，单位见单位列',
 '方法区间右端':'方法在W1概率区间右端的物理分位值，单位见单位列',
 'MC区间左端':'MC在W1概率区间左端的物理分位值，单位见单位列',
 'MC区间右端':'MC在W1概率区间右端的物理分位值，单位见单位列',
 'MC均值':'完整百万场景MC的电气量均值，单位见单位列；电压为幅值',
 '方法均值':'完整已保存分布的电气量均值，单位见单位列；电压为幅值',
 '均值误差':'方法均值减MC均值，单位见单位列',
 '均值绝对误差':'方法均值与MC均值之差的绝对值，单位见单位列',
 '均值误差除以MC标准差':'均值误差除以MC标准差，无量纲；不以可能接近零的均值作分母',
 'MC方差':'完整百万场景MC的经验方差，单位见方差单位列',
 '方法方差':'完整已保存模型分布的实际方差，非名义目标方差；单位见方差单位列',
 '方差误差':'方法方差减MC方差，单位见方差单位列',
 '方差相对误差':'方法方差/MC方差−1，无量纲，未乘100',
 '方差比_方法除以MC':'方法方差/MC方差，无量纲，1表示一致，未乘100',
 'MC标准差':'MC方差的平方根，单位见单位列',
 '方法标准差':'方法方差的平方根，单位见单位列',
 '方差单位':'p.u.²、kW²或kvar²；不能与均值单位混用',
 '矩计算':'针对完整已保存分布的矩计算方式，不截取到绘图范围',
 '尾部特征解释':'当前MC尾段的探索性形状描述；不是渐近重尾证明，也不由方法误差决定',
 '判别范围':'本轮判断仅针对MC有限概率范围，不是分布无限远处的定理',
 '最偏离均值0_1百分比场景的方差贡献':'最大1000个平方偏差之和除以全部平方偏差之和；保留全部百万场景',
 '单一最大偏差场景的方差贡献':'最大单一平方偏差除以全部平方偏差之和，保留该场景',
 'CDF_RMSE_百分点':'尾部CDF-RMSE乘100后的数值，百分点；不是相对百分比误差',
 '平均尾概率偏差_百分点':'50档阈值处“方法尾概率减MC尾概率”的平均值，百分点；正值表示该尾侧概率偏高',
 '尾概率偏差_百分点':'本行阈值处“方法尾概率减MC尾概率”，百分点',
 '分位误差':'方法分位值减MC分位值；物理单位见单位列',
 '绝对分位误差':'方法分位值与MC分位值之差的绝对值；物理单位见单位列',
 'MC尾部延伸':'该电气量、该尾侧由MC先行标记；不等于严格渐近重尾证明',
 'MC超额峰度':'MC普通标准化四阶矩减3，无量纲',
 'MC偏斜度':'MC偏度，无量纲',
 'MC尾部延伸比':'本行尾侧的MC分位距离与相同标准差高斯分布对应距离之比',
 '模型状态':'参数关系检查或自由度限制状态；不表示该行误差一定较大',
 '尾概率':'下尾为F，或上尾为1−F，对应的原始概率值',
 '分位水平':'CDF概率水平；0.001为下侧0.1%分位，0.999为上侧99.9%分位',
 'MC分位下限_95':'MC次序统计量给出的逐点95%分位值区间下限，单位见单位列',
 'MC分位上限_95':'MC次序统计量给出的逐点95%分位值区间上限，单位见单位列',
 'MC分位值':'逐场景交流潮流MC的物理分位值，单位见单位列',
 '方法分位值':'反求已保存电压模型对应电气量CDF所得的物理分位值，单位见单位列',
 'CDF反求残差':'反求所得物理分位值处的数值CDF与目标分位水平之差的绝对值，原始概率值；不是模型精度',
 'MC_CDF':'MC在当前物理阈值处的CDF，原始概率值',
 '方法CDF':'方法在当前物理阈值处的CDF，原始概率值',
 'MC尾概率':'下尾取MC_CDF，上尾取1−MC_CDF，原始概率值',
 '方法尾概率':'下尾取方法CDF，上尾取1−方法CDF，原始概率值',
 '物理阈值':'当前MC分位阈值；单位见阈值单位列',
 '分组t减AQGMM_百分点':'同一行分组t的尾部CDF-RMSE减AQGMM；负值表示分组t误差较小',
 '分组t减标准TMM_百分点':'同一行分组t的尾部CDF-RMSE减标准TMM；负值表示分组t误差较小',
 '误差数值由小到大':'只按当前同一电气量的尾部CDF-RMSE数值排序，不表示统计显著性',
 '绝对误差数值由小到大':'只按当前同一电气量、同一分位水平的绝对分位误差排序',
})

def table_md(frame,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,row in frame.iterrows():
        values=[]
        for col in columns:
            v=row[col];values.append(f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)

def write_notes():
    counts=pd.read_csv(ROOT/'tail_screening/tail_cdf_rmse_win_counts_by_K.csv')
    w=pd.read_csv(ROOT/'results/tail_cdf_rmse_method_compare.csv')
    cols=['K','比较项数','分组t误差小于AQGMM项数','分组t误差小于标准TMM项数','分组t误差三者最小项数']
    examples=w[(w.情形=='Pareto')&(((w.K==4)&(w.电气量=='V18')&(w.尾侧=='上尾'))|((w.K==2)&(w.电气量=='Q17')&(w.尾侧=='上尾')))]
    examples=examples.rename(columns={m+'_尾部CDF_RMSE_百分点':m for m in ['AQGMM','标准TMM','分组t']})
    root='''# IEEE33重尾测试：结果阅读入口

本文件夹的主要误差只有两项：**尾部CDF-RMSE**与**极端分位数误差**。当前没有按新标准计算的全分布CDF-RMSE表。

原“结果/CDF.csv”现名为“结果/tail_cdf_rmse_all_quantities.csv”。它每行汇总某一尾侧50个阈值的概率误差，不是CDF曲线，也不是全局CDF误差。原“尾段.csv”现名为“tail_cdf_per_threshold.csv”，才是逐阈值的CDF及尾概率明细。

## 先看这三个文件

- [筛选电气量的尾部CDF-RMSE三方法比较](tail_screening/tail_cdf_rmse_method_compare.csv)：1492行，每行直接比较同一情形、同一K、电气量和尾侧，误差单位为百分点。
- [筛选电气量的极端分位误差三方法比较](tail_screening/extreme_quantile_error_method_compare.csv)：4476行，每个分位水平分别比较，单位见表中单位列。
- [可筛选结果表](results_browser.html)：包含全部电气量，可切换“MC已标记尾部延伸”或“全部电气量”，不预先固定某个K或情形。

筛选依据为既有MC的尾部延伸比超过1.15，共373个“情形×电气量×尾侧”。这是有限概率范围的筛选，不等于373项都已证明为严格渐近重尾。

## 分组t的结果应如何表述

**现有结果不支持“分组t整体较差”的概括。**它在部分节点电压和支路功率的尾部误差较小，在部分电气量上较大；两类误差也可能给出不同次序。此前过多强调K=2的个别无功结果，不能据此概括其他K、节点和线路。

分组t的参数一致性未通过，与它在某些电气量上误差更小可以同时成立。当前数值改善如实保留；参数问题另见[模型参数核查](model_parameter_checks/README.md)，不能把参数未通过解释成每个电气量都较差，也不能把某些误差更小解释成参数关系已经满足。

以下仅统计MC已标记的373项，方便定位结果；这是逐项比较数量，不是全网平均误差或理论有效性的总评分。

'''+table_md(counts,cols)+'''

下列两项同时保留，CDF-RMSE单位均为百分点：

'''+table_md(examples,['情形','K','电气量','尾侧','AQGMM','标准TMM','分组t'])+'''

极端分位误差必须另看。例如Pareto、K=2的节点18电压上尾，分组t的尾部CDF-RMSE较大，但99.9%分位值绝对误差较小。不能由一个指标推断另一个指标的次序。

## 文件用途

| 位置 | 内容 | 是否用于当前精度比较 |
|---|---|---|
| [结果](results/README.md) | 最新两项误差、逐阈值CDF、MC分位值、电气量编号 | 是，逐电气量、逐K、逐尾侧比较 |
| [尾部筛选](tail_screening/README.md) | 按既有MC标记筛选的同一批结果 | 主要阅读入口 |
| [模型参数核查](model_parameter_checks/README.md) | 均值、协方差、随机强度矩、自由度等关系 | 是参数检查，不是CDF误差排名 |
| [诊断记录](diagnostics/README.md) | 改变某一物理计算条件或自由度以追查原因 | 与原模型精度表分开阅读 |
| [历史结果](legacy_metrics/README.md) | 原尾部平均绝对概率误差MAE | 不能当作当前RMSE |
| 计算记录 | 数值求解、积分加密、文件改名与哈希核验 | 辅助核查 |
| 程序 | 重算误差、整理表格及诊断程序 | 不混放数值结果 |
| 存档 | 冻结原始程序、注入、模型、百万场景MC和早期记录 | 保持原样 |

所有当前文件的用途见[file_index.csv](file_index.csv)，逐列含义见[field_dictionary.csv](field_dictionary.csv)，旧名与新名见[文件改名对照](computation_records/file_rename_map_20261026_cn.csv)。

本次整理逐项复算465600个尾部阈值对应的RMSE，核对分位误差和筛选表。主误差数值未更改；未生成注入、求新潮流或拟合模型。441个冻结存档文件的SHA256全部一致。

算例设置见[case_settings.csv](case_settings.csv)，注入PDF见[injection_power_pdf.png](injection_power_pdf.png)。原始MC单位仍为p.u.、MW、Mvar；用于汇报的分位误差为p.u.、kW、kvar。
'''
    (ROOT/'README.md').write_text(root)
    result='''# 当前误差表：文件、指标与单位

## 当前指标

| 文件 | 每行含义 | 指标与单位 | 行数 |
|---|---|---|---:|
| [tail_cdf_rmse_all_quantities.csv](tail_cdf_rmse_all_quantities.csv) | 情形×方法×K×电气量×尾侧 | 50档尾部CDF-RMSE；概率值与百分点均保留 | 9312 |
| [tail_cdf_rmse_method_compare.csv](tail_cdf_rmse_method_compare.csv) | 情形×K×电气量×尾侧 | 三种方法的同一指标，列名明确为百分点 | 3104 |
| [extreme_quantile_error_all_quantities.csv](extreme_quantile_error_all_quantities.csv) | 情形×方法×K×电气量×分位水平 | MC分位值、方法分位值、带符号与绝对误差；p.u./kW/kvar | 27936 |
| [extreme_quantile_error_method_compare.csv](extreme_quantile_error_method_compare.csv) | 情形×K×电气量×分位水平 | 三种方法的分位值及误差，同一行单位相同 | 9312 |
| [tail_cdf_per_threshold.csv](tail_cdf_per_threshold.csv) | 情形×方法×K×电气量×尾侧×阈值 | CDF和尾概率为0—1；概率差另以百分点列出 | 465600 |
| [tail_two_metrics_merged.csv](tail_two_metrics_merged.csv) | 分位误差每行附相应尾侧CDF-RMSE | CDF-RMSE在同侧三个极端分位水平重复，不应重复汇总 | 27936 |
| [mc_extreme_quantiles_ci95.csv](mc_extreme_quantiles_ci95.csv) | 情形×电气量×分位水平 | 仅为MC基准分位值及逐点区间，不是方法误差 | 2328 |
| [mc_quantity_distribution_features.csv](mc_quantity_distribution_features.csv) | 情形×电气量 | MC均值、标准差、偏度、超额峰度和尾部延伸比；原单位见字段说明 | 388 |
| [quantity_ids_units.csv](quantity_ids_units.csv) | 一个电气量 | 节点号、支路方向及汇报单位 | 97 |

尾部CDF-RMSE：下尾取MC的0.1%—5%分位阈值，上尾取95%—99.9%分位阈值，每侧50档。在相同阈值比较方法CDF与MC经验CDF，概率差平方、平均、开平方。乘100后为百分点，不是相对百分比。

极端分位水平：0.001、0.005、0.01、0.99、0.995、0.999。带符号误差＝方法分位值−MC分位值；比较精度时看绝对误差。两项指标不合并为一个分数。

## 当前没有的指标

本目录没有“全分布CDF-RMSE”。历史MAE文件中的full_CDF_max_pp为251个既有MC分位阈值上的CDF最大差，范围0.1%—99.9%；它既不是全分布RMSE，也不是严格KS距离。全分布曲线原始网格保存在冻结存档的CDF_方法_K编号.npz中。

## 比较范围

分组t、标准TMM、AQGMM的误差按同一行条件比较。不同K、不同节点/线路、不同尾侧可能次序不同。参数状态列只说明模型关系检查，不承担误差排序。重点结论见[尾部筛选](../tail_screening/README.md)，完整定义见[误差标准](../error_metrics.md)。
'''
    (ROOT/'results/README.md').write_text(result)
    focus='''# MC已标记尾部延伸的电气量

本目录使用此前保存的MC筛选结果，未改变筛选阈值，也未按方法误差挑选电气量。共373个“注入情形×电气量×尾侧”，对应4个K。

| 文件 | 内容 |
|---|---|
| [tail_extended_quantity_list.csv](tail_extended_quantity_list.csv) | 373项，附MC偏度、超额峰度和该侧尾部延伸比 |
| [tail_cdf_rmse_method_compare.csv](tail_cdf_rmse_method_compare.csv) | 1492行，同一电气量三方法并列，单位为百分点 |
| [extreme_quantile_error_method_compare.csv](extreme_quantile_error_method_compare.csv) | 4476行，逐分位水平比较，单位见表中单位列 |
| [tail_cdf_rmse_screened_quantities.csv](tail_cdf_rmse_screened_quantities.csv) | 4476行，逐方法的尾部CDF-RMSE长表 |
| [extreme_quantile_error_screened_quantities.csv](extreme_quantile_error_screened_quantities.csv) | 13428行，逐方法的极端分位误差长表 |
| [tail_two_metrics_screened_merged.csv](tail_two_metrics_screened_merged.csv) | 13428行，分位误差附相应尾侧CDF-RMSE；后者在同侧三行重复 |
| [tail_cdf_rmse_win_counts_by_K.csv](tail_cdf_rmse_win_counts_by_K.csv) | 逐K记录误差较小的比较项数，不能替代逐电气量结果 |
| [extreme_quantile_win_counts_by_K_level.csv](extreme_quantile_win_counts_by_K_level.csv) | 不同分位水平分开记录，不平均不同物理单位 |

筛选依据：MC的0.5%或99.5%分位到中位数的距离，与相同标准差高斯分布对应距离相比超过1.15。这只能说明该概率范围的尾部延伸，不是严格渐近重尾证明。

分组t的数值改善和变差均保留。参数一致性问题与误差大小分别记录，不以参数检查未通过推断误差一定较大。

'''+table_md(counts,cols)+'\n'
    (ROOT/'tail_screening/README.md').write_text(focus)
    (ROOT/'legacy_metrics/README.md').write_text('''# 历史误差：尾部CDF平均绝对误差MAE

本目录保留首轮指标，不能作为当前尾部CDF-RMSE。

- tail_cdf_mae_all_quantities.csv：全部电气量；lower_tail_pp和upper_tail_pp是50档阈值处的平均绝对概率误差，单位为百分点。
- tail_cdf_mae_screened_quantities.csv：相同旧指标在MC筛选电气量上的结果。

full_CDF_max_pp是251个既有MC分位阈值上的CDF最大绝对差，单位为百分点；不是全分布CDF-RMSE，也不是精确KS距离。CF_end_abs是积分检查量，不是精度指标。原列名保留，逐列说明见根目录field_dictionary.csv。
''')
    (ROOT/'diagnostics/README.md').write_text('''# 诊断记录与当前精度表分开阅读

本目录追查误差来源，不能把改变物理计算条件或改变自由度后的数值直接当作原三方法精度。

| 位置 | 已完成内容 | 阅读限制 |
|---|---|---|
| [根因检查](root_cause_check/README.md) | 支路17固定无功条件比较、节点18自由度改变及电压响应检查 | 同时含直接抽取电压计算与解析计算，具体含义见字段说明 |
| [支路检查](branch_check/README.md) | 独立三相功率计算、190个原注入场景的独立交流潮流、同一电压分布下解析CDF核查 | 是基础计算核查，不是相对MC的三方法误差排名 |
| branch_check/incomplete_analytic_check | 尚未完成的固定无功条件解析积分试算 | 部分线路尚未数值收敛，未并入results/或tail_screening/ |

当前正式误差仍是已保存三种模型在results/中的两项指标。基础功率公式检查已通过，不能据此声称分组t处处较差；支路17的物理问题也不能代替其他节点、线路和K的结果。
''')
    (ROOT/'diagnostics/branch_check/README.md').write_text('''# 支路功率基础计算检查

- three_phase_unit_power_formula.csv：4种注入情形、全部32条在运线路；比较三相SI单位、标幺值、实虚部公式、复数公式与独立二次型。
- independent_acpf_vs_original_mc.csv：190个既有MC注入场景，包含普通与尾部运行点；独立求解后核对原MC节点电压、支路P/Q和总有功损耗。
- same_voltage_law_analytic_vs_direct_cdf.csv：K=2，4种注入情形、3种已有电压模型、64个支路P/Q，共768项。比较的是两种计算方式是否给出同一分布，不是各模型相对MC的误差。
- incomplete_analytic_check/：固定无功条件下的解析积分试算，仅部分模型输出完成，且部分线路未收敛。数据保留，未用于当前三方法精度判断。

现有基础检查中，独立支路功率与原MC的最大差约8.04×10⁻¹¹ MW/Mvar；同一电压分布的解析CDF与直接计算差在本次抽取电压的同时95%分布界内。检查范围和单位逐列见根目录field_dictionary.csv。以上不能解释成修正后的电压分布已经满足完整交流潮流。
''')
    # Correct reader links and stale scope statements without changing frozen evidence.
    p=ROOT/'model_parameter_checks/README.md';text=p.read_text()
    text=text.replace('状态：分组t参数一致性检查不通过。本批次的分组t结果暂停用于检验新理论场景下的方法优劣。',
        '状态：分组t参数一致性检查不通过。这不代表它在各电气量上的误差都更大。当前实现的精度改善和变差均应按结果表如实报告；参数未满足时，不能将这些结果视为已完整验证文稿中的模型。')
    text=text.replace('此前用固定无功保持不足解释支路无功误差，只描述了一个已观察到的物理偏差，尚不足以确定其主因。本次发现的上游参数问题也可能造成这项偏差，不能省略上游检查。',
        '此段原为早期判断。后续支路17检查已量化固定无功关系未保持的影响，见../diagnostics/root_cause_check/README.md；该支路的原因不代表分组t在其他电气量上较差。上游参数问题仍需单独保留。')
    if '## 本目录当前文件' not in text:
        text += '''\n## 本目录当前文件\n\n- voltage_mean_cov_third_moments.csv：逐分量参数关系，不是尾部CDF误差。\n- grouped_t_strength_moments.csv：随机强度均值及二阶矩，不是电压PDF或CDF。\n- grouped_t_voltage_fourth_moments.csv：组内四阶统计量，不是单节点普通峰度。\n- standard_TMM_dof_fourth_moments.csv：自由度下限及四阶关系。\n- source_groups_residual_voltage_directions.csv：分组覆盖和剩余方向统计特征。\n\n原英文文件名对应冻结audit_theory/中的证据；本目录采用上述简明文件名。参数检查不能替代当前results/和tail_screening/中的逐项误差比较。\n'''
    p.write_text(text)
    p=ROOT/'diagnostics/root_cause_check/README.md';text=p.read_text()
    names={'支路无功.csv':'branch17_Q_physical_condition_compare.csv','无功尾段.csv':'branch17_Q_per_threshold_cdf.csv',
        '节点电压对照.csv':'bus18_V_error_after_dof_change.csv','自由度对照.csv':'bus18_V_dof_change.csv',
        '节点18分量.csv':'bus18_V_component_skew_fourth_moment.csv','自由度来源.csv':'dof_retained_voltage_direction_check.csv',
        '电压响应抵消.csv':'fixed_reactive_voltage_response_cancellation.csv','功率平衡.json':'mc_end_reactive_balance_check.json'}
    for old,new in names.items():text=re.sub(r'(?<![\w])'+re.escape(old),new,text)
    text=re.sub(r'(?<!\.\./)\.\./存档/','../../../../../../archive/heavytail_source_snapshot_20261026/',text).replace('python3 -B root_cause.py','python3 -B ../../../tests/python/heavytail_postprocess/root_cause.py')
    text=text.replace('功率平衡.json、verification.json','mc_end_reactive_balance_check.json、../../computation_records/original_root_cause_check.json')
    text=text.replace('mc_end_reactive_balance_check.json、verification.json','mc_end_reactive_balance_check.json、../../computation_records/original_root_cause_check.json')
    if '## 本次整理补充' not in text:
        text+='\n## 本次整理补充\n\n本报告针对K=2的固定电气量追查原因，不能据此判断其他K或全部电气量的精度。分组t在多个已标记尾部延伸的电气量上误差更小，完整比较见../../tail_screening/README.md。支路固定无功条件的直接分布检查不等于后续解析修正已完成；尚未收敛的积分试算已单独放在branch_check/incomplete_analytic_check。\n'
    p.write_text(text)
    p=ROOT/'error_metrics.md';text=p.read_text()
    for old,new in {'results/误差.csv':'results/tail_two_metrics_merged.csv','results/旧误差.csv':'legacy_metrics/tail_cdf_mae_all_quantities.csv'}.items():text=text.replace(old,new)
    if '## 7 文件命名和比较方向' not in text:
        text+='''\n## 7 文件命名和比较方向\n\n当前主表文件名明确区分尾部CDF-RMSE、极端分位数误差和逐阈值CDF取值。当前没有全分布CDF-RMSE文件。三方法比较表中，误差从小到大排列；负的“分组t减AQGMM”表示分组t误差较小。\n\n参数关系未通过与部分电气量误差更小可以同时出现。必须如实保留这部分改善，同时说明当前实现尚未满足的参数关系；不得从参数未通过直接推断尾部精度较差。两项误差次序不一致时分开报告。\n'''
    p.write_text(text)

def purpose(path):
    s=str(path)
    if '尾部分位区间W1' in s:return '截去最外侧0.1%的尾部分位区间W1；下尾[0.001,0.05]、上尾[0.95,0.999]；单位p.u./kW/kvar','新增尾部误差；不是完整W1'
    if '全分布均值方差' in s:return '完整MC与完整模型分布的均值、方差及其误差；电压为幅值，方差单位另列','完整分布矩比较'
    if s.startswith('tail_distribution/'):
        if '三项误差及均值方差' in s:return 'MC先行选出的36项、全部K和方法；三项尾部误差与完整分布均值方差并列','重点电气量完整比较；432行'
        if '全部电气量_MC尾部特征' in s:return '全部节点、线路及网损上下尾的稳健分位延伸比、GPD形状及阈值变化','MC先行分析；776个尾侧'
        if '重点电气量_MC先行选择' in s:return '按每情形每类型的稳健0.1%尾部延伸比取前三项；未使用模型误差','36个重点电气量'
        if '尾部延伸较弱的比较电气量' in s:return '按MC形状选出的较弱比较电气量；用于标准化PDF比较','12个MC比较电气量'
        if 'PDF区间平均密度数据' in s:return '同一物理分箱上的MC与解析CDF区间平均密度，未作核平滑或尾部重新归一化','PDF的原始绘图数值'
        if 'PDF显示范围与数值检查' in s:return '各图显示范围内的完整分布概率、非正密度区间及数值差','PDF显示和积分检查'
        if '图形索引' in s:return '108张图的情形、电气量、图式、单位和PNG/PDF文件位置','图册导航'
        if '/figures/' in s:return 'MC尾部特征图或共用分箱的PDF研究检查图；同名manifest记录尺寸、字体和条件','研究图；未作为原项目正式论文图导出'
        if '/computation_records/' in s:return '每个已保存模型的W1、PDF网格或完整矩加密记录；没有重新估计模型','新增分析计算记录'
        if '方差贡献' in s:return '最偏离均值的0.1%场景和最大单一场景承担的方差比例；未删除极端场景','MC经验方差稳定性说明'
        if s.endswith('.md'):return '全部节点与线路尾部形状、PDF、W1和完整矩的解释或图册','本轮分析阅读入口'
        return '新增MC尾部形状、PDF、尾部W1和全分布矩分析的设置或核验','独立副本中的新增分析'
    if 'incomplete_analytic_check' in s:return '未完成的固定无功条件解析积分试算；不并入当前精度结果','未完成；含未收敛线路'
    if s.startswith('legacy_metrics/') and s.endswith('.csv'):return '历史尾部CDF平均绝对误差MAE及旧网格检查量；不是当前RMSE','历史指标'
    if s.startswith('model_parameter_checks/'):return '电压分布参数或分组关系检查；不是电气量CDF误差排名','参数核查'
    if s.startswith('diagnostics/root_cause_check/'):
        if '逐阈值' in s:return '支路17上尾；原解析CDF、直接计算CDF和固定无功条件比较','原因检查；K=2'
        if '物理条件' in s:return '支路17无功固定条件比较；含尾部CDF-RMSE和极端分位误差','原因检查；不是原三方法主结果'
        if '自由度改变后的误差' in s:return '节点18电压改变自由度后的两项上尾误差','原因检查；不是新推荐模型'
        if '抵消' in s:return '节点18固定无功关系中的一阶与二阶电压响应','物理关系检查'
        return '节点18电压或支路17无功的定参与物理条件检查','原因检查'
    if s.startswith('diagnostics/branch_check/'):
        if '解析CDF' in s:return '同一电压分布下两种计算方式的CDF差；不是相对MC的模型误差','基础计算核查；768项'
        return '支路单位、功率公式或独立交流潮流与原MC一致性','基础计算核查'
    if '三方法对比' in s:
        return ('逐电气量、逐尾侧三方法尾部CDF-RMSE；百分点' if 'CDF' in s else '逐电气量、逐分位水平三方法分位值及带符号/绝对误差；p.u./kW/kvar'),'当前已保存模型结果'
    if '尾部CDF_RMSE_' in s:return '每个尾侧50档阈值的CDF-RMSE；概率值与百分点；不是全分布CDF','当前已保存模型结果'
    if '极端分位数误差_' in s:return '六个极端分位水平的带符号和绝对误差；p.u./kW/kvar','当前已保存模型结果'
    if '逐阈值' in s:return '每个尾侧50档物理阈值处的CDF、尾概率及概率差','当前CDF取值及误差明细'
    if '合并查看表' in s:return '分位误差附该侧CDF-RMSE；CDF-RMSE在同侧三行重复','合并查看；不能重复汇总CDF-RMSE'
    if 'MC极端' in s:return 'MC极端分位值及逐点95%区间；不是方法误差','MC基准'
    if 'MC电气量' in s:return 'MC均值、标准差、偏度、超额峰度及两侧延伸比；原单位见字段说明','MC分布特征'
    if '电气量编号' in s:return '节点电压、支路首端P/Q、网损编号、方向与报告单位','电气量定义'
    if '尾部延伸电气量清单' in s:return '既有MC筛选的373个情形×电气量×尾侧；有限范围尾部延伸','筛选条件固定'
    if '比较项数' in s:return '逐K或逐分位水平记录误差较小的项数；不是全网平均或合成评分','辅助定位结果'
    if s=='case_settings.csv':return '基础负荷、8个电源完整注入分布参数和相关系数','算例设置'
    if s.startswith('../../../tests/python/heavytail_postprocess/'):return '复算、文件整理或诊断程序；未完成试算在文件名中标明','程序'
    if s.startswith('computation_records/'):return '数值积分、分位反求、原核验记录或文件校验；不是方法精度主表','辅助或历史计算记录'
    if s.endswith('.md'):return '本目录文件含义、范围及结果解释','阅读说明'
    if s=='results_browser.html':return '三项尾部误差和全分布均值方差的逐电气量筛选查看表','当前结果查看入口'
    if s=='injection_power_pdf.png':return '完整电源有功注入分布的PDF和尾部曲线','算例输入分布'
    if s=='file_index.csv':return '每个当前文件的用途、数据行数、列数和状态','文件索引'
    if s=='field_dictionary.csv':return '各CSV逐列的指标、单位和比较含义','字段索引'
    return '文件整理索引或说明','辅助记录'

def definition(col):
    if col in DEFINITIONS:return DEFINITIONS[col]
    for m in ['AQGMM','标准TMM','分组t']:
        if col==m+'_尾部CDF_RMSE_百分点':return m+'在本行电气量和尾侧的尾部CDF-RMSE，百分点，越小越准确'
        for key in ['方法分位值','分位误差','绝对分位误差']:
            if col==key+'_'+m:return m+'：'+DEFINITIONS[key]
        for key in ['方法均值','均值误差','方法方差','方差相对误差','方差比_方法除以MC','方法标准差']:
            if col==key+'_'+m:return m+'：'+DEFINITIONS[key]
    if col.startswith('GPD形状_'):
        return '所列尾概率阈值下的GPD形状参数（SciPy正号表示较厚幂尾倾向）；区间为固定阈值的逐点探索性95%区间；不是渐近重尾证明'
    if col.startswith('高斯标准化尾部延伸比_'):
        return '该尾概率的分位值到中位数距离，除以同侧四分位到中位数距离，再除以高斯对应比值；1为高斯同尺度，无量纲'
    if col.startswith('极端分位绝对误差_尾概率'):
        return '该尾侧、该尾概率的方法物理分位值与MC之差的绝对值，单位见单位列'
    if col=='MC所选尾段跨度_尾概率0_1至5百分比':return 'MC同侧尾概率0.1%与5%对应物理分位值之间的距离，单位见单位列'
    if '误差小于' in col:return '当前同一电气量、同一K和同一指标的数值比较；不表示统计显著性'
    if '最小' in col:return '当前这一比较项中误差数值是否为三者最小；不代表参数一致性通过'
    return col+'；数值定义以本文件用途及目录说明为准'

def write_index():
    # Keep the move-time hash and current hash distinct for prose/code revisions.
    ledger_path=ROOT/'computation_records/file_rename_map_20261026_cn.csv'
    ledger=pd.read_csv(ledger_path)
    def digest(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            while block:=f.read(8*1024*1024):h.update(block)
        return h.hexdigest()
    ledger['最终文件SHA256']=[digest(ROOT/p) for p in ledger.新路径]
    ledger['后续修改说明']=['数值内容未变' if a==b else '仅程序路径、查看表或阅读说明已更新；主误差值另行核验' for a,b in zip(ledger.移动后SHA256,ledger.最终文件SHA256)]
    ledger.to_csv(ledger_path,index=False,encoding='utf-8-sig')
    rows=[];fields=[];missing=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if rel.parts[0]=='存档' or p.name.startswith('.') or '__pycache__' in rel.parts or 'old_bytecode_cache' in rel.parts:continue
        if str(rel) in ['file_index.csv','field_dictionary.csv']:continue
        desc,state=purpose(rel);nrows='';ncols=''
        if p.suffix=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.reader(stream);columns=next(reader);ncols=len(columns);nrows=sum(1 for _ in reader)
            for col in columns:
                fields.append(dict(文件=str(rel),字段=col,含义及单位=definition(col)))
                if re.match('[a-zA-Z]',col) and not re.search('[\u4e00-\u9fff]',col) and col not in DEFINITIONS:missing.append((str(rel),col))
        rows.append(dict(文件=str(rel),用途及指标=desc,状态=state,数据行数=nrows,列数=ncols,字节数=p.stat().st_size))
    assert not missing,missing
    pd.DataFrame(fields).to_csv(ROOT/'field_dictionary.csv',index=False,encoding='utf-8-sig')
    rows += [dict(文件='field_dictionary.csv',用途及指标='逐CSV逐列的指标、单位和比较含义',状态='字段索引',数据行数=len(fields),列数=3,字节数=(ROOT/'field_dictionary.csv').stat().st_size),
             dict(文件='../../../../archive/heavytail_source_snapshot_20261026/',用途及指标='441个冻结程序、注入、模型及MC文件；逐文件SHA256见computation_records/frozen_archive_sha256.csv',状态='冻结；未改动',数据行数='',列数='',字节数='')]
    pd.DataFrame(rows).to_csv(ROOT/'file_index.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(dict(indexed_files=len(rows),field_definitions=len(fields)),ensure_ascii=False))

if __name__=='__main__':
    write_notes()
    if (ROOT/'tail_distribution/focus_quantities_three_metrics_mean_variance.csv').exists():
        from tail_distribution_report import main as tail_report
        tail_report()
    write_index()
