# IEEE 33节点光伏算例结果

本目录由原项目 `wuhanichina/AQPPF_TMM` 的 `result/validation/` 同名目录复制而来，只保留判定表、汇总表、诊断图、计算清单和运行记录。逐次重复的明细误差表、模型参数与电压数组未上传，其SHA256见各目录 `run_manifest.json`。

| 目录 | 内容 |
|---|---|
| `pv_fixed_forecast_case33_20261021/` | 基准场景：GMM、标准TMM、分块TMM，越限风险解析计算与诊断图 |
| `pv_fixed_forecast_case33_20261021_rep1/` | 另一组独立生成的源荷场景重新估计参数 |
| `pv_fixed_forecast_case33_20261021_gaussian_control/` | 天气强度恒定的高斯波动比较情形 |
| `pv_strong_weather_case33_20261021/` | 场景A：云层强弱差别更大 |
| `pv_strong_weather_nominal_load_case33_20261021/` | 场景A+B：同时负荷取额定值 |
| `pv_two_cloud_regions_case33_20261021/` | 场景C：两片独立云区，含两块TMM |

结果解释见 `01_IDEA/pv_fixed_forecast_results_20261021.md`。
