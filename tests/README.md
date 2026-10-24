# tests

`python/`为当前计算程序，`python/archived/`为原光伏算例计算程序（保持原字节，不修改）。`map_injections_fast_case33.m`为快速映射的MATLAB入口，由`python/fast_voltage_map.py`调用（需要MATLAB与MATPOWER，默认路径`../../00codezip/matpower8.0`，可用环境变量`MATPOWER_ROOT`、`MATLAB_BIN`指定）。

## 当前方法使用的程序

| 程序 | 作用 |
|---|---|
| `fit_nrel_pv_forecast_error_case33.py` | 由`data/nrel_sind/co/`的NREL记录构造光伏预测误差规律 |
| `run_pv_nrel_case33.py` | 算例A的源荷场景、MC方法精确值和AQGMM拟合（调用`run_pv_fixed_forecast_case33.py`） |
| `fast_voltage_map.py` | 快速映射：注入场景到节点电压 |
| `block_split_rule_check.py` | 天气驱动电压方向的确定（method_zh.md第4.2节步骤1—3） |
| `moment_matched_tmm_case33.py` | 标准TMM按矩确定参数；多尺度t（AQPPF_TMM主分支方法） |
| `grouped_strength_tmm_case33.py` | 分组t混合参数估计、模型抽样与尾部指标 |
| `truncated_strength_case33.py` | 强度上限（步骤4）与截断强度比较 |
| `analytic_limit_risk_pv_fixed_forecast.py` | 特征函数、强度积分与Gil-Pelaez反演（标准TMM、分块TMM已实现） |

## 支撑性与历史程序

`block_split_rule_scenarios.py`、`apply_block_split_rule_case33.py`、`compare_block_split_case33.py`（分组规则检验）；`extreme_weather_case33.py`、`summarize_extreme_weather_case33.py`（极端天气）；`feasibility_case1a_all_weather.py`（全部注入同受天气）；`grouped_nu_estimation_case33.py`、`plot_full_distribution_grouped.py`（分组t诊断）；`check_nrel_sind_forecast_error.py`（NREL数据检查）。

已不在主线：`tail_baselines_pv_nrel_case33.py`（比较方法首轮复现，未按原文）、`analytic_limit_risk_pv_nrel_case33.py`、`summarize_pv_nrel_case33.py`、`plot_case1_pdfs_pv_nrel.py`（算例1首轮）、`run_pv_shared_residual_case33.py`、`shared_residual_mixture.py`、`select_k_by_pv_block*.py`（共用残差，已归档；`shared_residual_mixture.py`仍被`run_pv_nrel_case33.py`导入）、`run_wind_fixed_forecast_case33.py`（风电最小测试）。

测试结果只有登记到`01_IDEA/evidence_map.md`后才可作为论文证据。
