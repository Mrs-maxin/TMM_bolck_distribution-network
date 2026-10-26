# 直接PV出力云端基础测试包

## 2026-10-26 当前阶段

用户最新要求先完成解析概率潮流及数值自检，方法误差口径讨论后再比较。实际云端运行已完成源荷、交流潮流及K=1三种完整联合电压模型参数保存；当前入口只从`result/case33bw/direct_pv8_python_cloud_20261026/models/K1.pkl`续算各模型97个电气量的PDF/CDF，不重新计算源荷和交流潮流，不估计参数，不生成方法误差表或叠加比较图。已完成解析结果及自检见`analytic_only/`，`README.md`明确本阶段范围与未完成项。K=2、4旧拟合记录及有限结果保留，完整参数未保存，当前阶段不重估。以下为原准备包说明，原始未修复结果保留在`initial_attempt/`，不作为当前精度结论。

本包只在OpenAI云端执行数值计算。准备时没有在Mac运行潮流、拟合或绘图。当前仅有静态语法检查，不代表已通过云端计算。

在OpenAI云端Linux工作目录解压完整测试包。Python 3.11以上，按需安装依赖：

```sh
python -m pip install -r tests/python/requirements.txt
python -u tests/python/run_direct_pv8_python_cloud.py --confirm-openai-cloud
```

`--confirm-openai-cloud`是执行者对运行位置的声明；程序不能仅根据Linux平台独立证明执行者使用了OpenAI云端。运行清单记录实际主机、路径和版本。本机macOS入口拒绝数值执行。

入口设置为8站各0.60 MW，负荷0.8倍基础值和原约1%小幅波动，K=1、2、4。光伏只读取17520组同步标幺出力；不使用预测、误差、附加光伏扰动或复制光伏记录。本站来源为NREL气象驱动模拟出力，不是电站实测数据。

基础测试复用现有Python快速映射、独立极坐标交流潮流、AQGMM和两种TMM。电气量PDF由电压联合分布按交流功率二次关系和特征函数反演，不对每个电气量单独拟合混合分布。

输出目录为`result/case33bw/direct_pv8_python_cloud_20261026/`：

- `run_manifest.json`：实际执行位置、版本、阶段、运行范围和反演检查；只有status为complete才完成。
- `inputs.npz`、`date_split.json`、`ac_results.npz`：源荷注入及日期划分、交流潮流数据。
- `tail_inventory_all97.csv`：逐节点、逐支路重尾和长尾迹象的完整清单。
- `selected_objects_before_fit.json`：方法拟合前固定的PDF对象。
- `voltage_groups.json`、`fit_audit.json`：分组、初始参数选择和退化情况。
- `per_object_comparison.csv`：逐对象、逐K、逐方法的上下尾及完整CDF比较。
- `figures/`：8站出力、节点净P/Q、所选电压和支路P/Q及网损的PNG/SVG、CSV与来源说明。CDF/PDF用同一模型计算，参考PDF为独立日期交流潮流的经验直方密度。

运行后须检查：全部日期和场景完整；电压和反向潮流状态合理解释；PDF反演端点与负密度残差；图形坐标和尾部范围无裁错；出现未满足重尾/长尾判据的对象时明确说明；不利于TMM的结果照实保留。程序未经端到端运行，云端若报错需保留failure_traceback.txt后修复，不能声称已有结果。

交回完整输出目录，保留运行日志。论文正式图仍后续由MATLAB导出；本包图片只是Python基础诊断图。
