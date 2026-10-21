# tests

## 当前验证程序

`python/` 为本项目当前的计算程序：`run_pv_fixed_forecast_case33.py` 依次完成源荷场景生成与交流潮流、电压分布参数估计、精确结果比较和分块TMM计算；`analytic_limit_risk_pv_fixed_forecast.py` 按特征函数与反演解析计算越限风险；`plot_*.py` 输出诊断图。`python/archived/` 为原光伏算例计算程序，保持原字节。


本目录保留给具体项目的临时验证、调参检查和本地调试入口。模板不附带默认 MATLAB 测试函数或统一测试入口，部署和复制模板时不得自动执行 MATLAB。

项目作者可以按真实模型、数据、toolbox 和求解器依赖在本目录添加检查。测试结果只有通过正式算例或已说明的验证入口生成，并登记到 `01_IDEA/evidence_map.md` 后，才可作为论文证据。
