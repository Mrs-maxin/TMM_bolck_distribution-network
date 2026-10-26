# 8站直接光伏出力：云端解析概率潮流

当前阶段已完成K=1三种已保存联合电压模型的97个电气量解析PDF/CDF，共291组分布、582张单模型图，每张PNG/SVG各一份。源荷注入和交流潮流保持原值。当前阶段不作方法精度或优劣结论。

用户最新要求先完成解析计算，方法误差口径待讨论确认。K=2、4旧拟合记录和有限结果保留；当时完整参数未保存，本阶段不重新估计这些参数，未扩展其97个电气量计算。

## 运行条件及交流潮流

8站按18、17、16、15、14、13、33、22接入，每站0.60 MW、无功为零。使用17520组原始同步记录，负荷0.8倍及原约1%波动。219/73/73天整体分配，独立交流潮流3504组。NREL数据为气象驱动模拟出力。参考电压范围0.9344555903—1.0947944859 p.u.，馈线首端有功范围−0.7841775380—3.0276081539 MW，反向比例39.01255708%，任一节点超过1.07 p.u.比例28.90981735%。容量未按模型误差改变。

## 当前结果

- `analytic_only/K1/<model>/<object>.csv`：完整PDF/CDF数值，电压p.u.、P及网损MW、Q为MVAr，PDF为相应单位倒数。
- 同名JSON：条件特征函数、强度积分、频率与求积加密、归一化及CDF检查。`*_characteristic_terms.npz`保留条件权重和二次关系特征函数参数。
- `analytic_only/figures/`：单模型PDF和CDF的PNG/SVG，不叠加方法曲线或交流潮流直方图。
- `analytic_only/numerical_self_checks.csv`、`integration_diagnostics.json`、`calculation_summary.json`：逐对象数值自检，不是方法精度比较。
- `figures/PV*_active_output.*`、`bus*_net_P.*`、`bus*_net_Q.*`：保留的8站和32个非平衡节点净注入直方密度及分箱CSV，零值未删除。`total_PV_output.*`为总出力。
- `inputs.npz`、`date_split.json`、`ac_results.npz`：原始源荷、日期划分和已完成交流潮流。
- `models/K1.pkl`：当前使用的三种联合电压分布参数与电压坐标，解析阶段没有重新估计。
- `tail_inventory_all97.csv`：97个电气量的交流潮流尾部诊断。无符合重尾判据对象；32个电压、29条支路P符合单侧长尾，支路Q与网损不符合，有限记录不证明渐近重尾。

## 数值检查

电气量由联合电压分布、交流功率二次关系与条件特征函数反演得到，没有逐电气量单独拟合。电压幅值按V²分布作F_V(v)=F_V²(v²)、f_V(v)=2v f_V²(v²)。普通电气量使用复合Gauss-Legendre傅里叶积分；网损采用均匀傅里叶求和并加密频率网格及加倍区间。强度积分按同一模型加密，不作模型之间的精度判断。

PDF积分与1的最大绝对差为3.9137444e-08；PDF积分与CDF端点差的最大差为3.9147146e-08。下界CDF最大为9.9214358e-09，上界遗漏概率最大为5.5979412e-09。CDF最小相邻增量为-4.8849813e-15，原始密度最小值为-3.790802e-13，浮点负残差保留，没有裁零或平滑。强度求积加密后的CDF最大变化为1.9639098e-05。

已知高斯、64维卡方、一维t分布检查见`analytic_only/inversion_known_distribution_checks.json`。图的坐标、单位、图例和画布检查见`analytic_only/figure_check_report.json`。宽区间扩展前的曲线在`analytic_only/boundary_refinement_history/`保留。

## 保留的旧输出及未完成项

`initial_attempt/`保留原入口完整运行及原始网损负密度、振荡、截断不足诊断。根目录旧`per_object_comparison.csv`、原方法叠加图和拟合记录均保留，属于阶段调整前输出，当前不采用其中的误差或优劣判断。新的方法误差表、同K/跨K精度判定和方法叠加图已暂停。

`logs/`记录原运行、中断、求积修复、边界扩展和图片检查。`cloud_environment.json`记录实际环境、依赖及数据SHA256；三个直接光伏文件字节完全符合清单。既有原始站点CSV为LF换行，其CRLF字节表示符合来源SHA256；该差别已记录，未覆盖原文件。备用ZIP及其两份出力CSV亦经核验。

基线b8fe31e1bd2f6e973cec0e92a73f9e9905f9b2ea，分支codex/direct-pv8-cloud-test。Git命令网络连接失败后，经GitHub连接器恢复差异，完整树哈希与基线一致。未修改归档程序或原项目，未强制重置、清理或推送。

当前入口从已保存模型续算：激活`/workspace/.cloud-setup/tmm/activate.sh`后运行`python -u tests/python/run_direct_pv8_python_cloud.py --confirm-openai-cloud`。完整源代码和说明副本在`reproducibility/`；旧准备包在`original_test_package.zip`。当前入口复用已完成模型与交流潮流，不启动参数精度比较。
