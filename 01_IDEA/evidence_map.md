# Evidence Map

每个证据编号对应生成入口、结果目录和关键数值。路径相对仓库根目录；`*.npz`、`*.npy`只保存在本地，不进入Git。

## 2026-10-22至2026-10-24

| 证据编号 | 支撑结论 | 生成入口 | 结果文件 | 关键数值 | 状态 |
| --- | --- | --- | --- | --- | --- |
| E-RECORD-SCENARIOS-20261024 | 算例输入与精确结果 | `tests/python/generate_record_scenarios_case33.py`（S1、S2E1、S2E2、S3）；检查`check_record_resampling_case33.py`、`check_record_tail_options_case33.py` | `result/case33bw/record_scenarios_20261024/<运行条件>/run_info.json`、`run_all.log`；MC方法97个电气量`cache/record_scenarios_20261024/<运行条件>_mc_outputs.npy`（本地）；历史版本`..._band40_60`、`..._band60_80_cap0p5805` | 光伏偏差取自预测出力60%—80%的记录时段，每站0.4146 MW、预测出力70%；场景与记录的单站峰度（S1 5.14/5.10）、多站同时大偏差概率等相差约2%以内；快速映射均1个锚点、无逐场景回退；MC方法10⁶场景120—191 s。最高节点电压中位数1.032 p.u.、99.9%分位1.051—1.055 p.u.，按GB/T 12325限值无越限；节点电压峰度最大6.2—7.2，支路有功峰度中位数4.4—5.1（S2E2最大30.0） | supported |
| E-FASTMAP-20261024 | C5 | `tests/python/fast_voltage_map.py`调用`tests/map_injections_fast_case33.m`（`+BTMM_core/+inverse/map_samples_with_atlas.m`） | `cache/fast_map/voltage_*.json`（映射记录，可重算） | 算例1、两片云区各8192+4096个场景：1个锚点，全部场景功率残差<1e-8 p.u.，无逐场景回退，每批2—4 s；与逐场景交流潮流电压最大差2.7e-8 p.u.，拟合坐标最大差5.9e-4（归一化单位） | supported |
| E-STD-TMM-DILUTION-20261024 | C3 | 推导 | `02_PAPER/manuscript/method_zh.md`式(12) | $b-1=r(r+2)(b_u-1)/[d(d+2)]$；33节点$d=64$、$r=8$系数0.019 | supported（理论推导） |
| E-NREL-INPUT-20261022 | 算例输入 | `tests/python/fit_nrel_pv_forecast_error_case33.py` | `result/case33bw/nrel_pv_input_case33_20261022/fit_HA4_0.4_0.6.json`；数据`data/nrel_sind/co/` | Denver 8个分布式光伏站，HA4预测，40%—60%预测出力段4000个时段；共同强度对数正态对数标准差1.192；2026-10-24由`data/`重算，172个数值与原结果一致 | supported |
| E-CASE1-NREL-20261022 | 历史 | `tests/python/run_pv_nrel_case33.py` | `result/case33bw/pv_nrel_case33_20261022/`（不含场景文件）；说明`01_IDEA/results/case1_nrel_results_20261022.md` | 首轮（旧分块）：节点电压0.1%越限概率共用残差误差−28%—−46%，AQGMM −71%—−85% | archived（被新分块替代） |
| E-SPLIT-RULE-20261022 | C4 | `tests/python/block_split_rule_check.py`；`tests/python/block_split_rule_scenarios.py` | `result/case33bw/block_split_rule_check_20261022/` | 6个已有场景、4个构造场景、AQGMM前作输入判别正确；8192个场景时2处临界误判；NREL输入最大二次响应方向95.7%在潮流二次响应空间内 | supported |
| E-CASE1-NEWSPLIT-20261022 | C1、C2 | `tests/python/run_pv_nrel_case33.py`（新分块）；`tests/python/compare_block_split_case33.py` | `result/case33bw/pv_nrel_case33_newsplit_20261022/`；说明`01_IDEA/results/case1_newsplit_results_20261022.md` | K=6、8、12：4个节点电压尾段误差为AQGMM的1/5—2/3；P16、P17支路尾段AQGMM更准 | supported（单次运行） |
| E-CASE1A-FEAS-20261022 | C3 | `tests/python/feasibility_case1a_all_weather.py` | `result/case33bw/feasibility_case1a_all_weather_20261022/`、`..._invgamma_df5_20261022/` | 全部注入同受天气：标准TMM自由度4—33，似然高于GMM，尾段不优于GMM（对数正态强度）；逆伽马强度时K≥3与GMM相当或更好 | supported（单次运行） |
| E-EXTREME-WEATHER-20261022 | 深尾边界 | `tests/python/extreme_weather_case33.py`；`tests/python/summarize_extreme_weather_case33.py` | `result/case33bw/extreme_weather_case33_20261022/`；说明`01_IDEA/results/extreme_weather_results_20261022.md` | MC方法10⁶场景；K≥6时TMM尾段多数优于AQGMM 20%—55%；全部模型低估0.1%以外深尾；拟合场景增至65536无改善 | supported |
| E-MOMENT-TMM-20261022 | C6 | `tests/python/moment_matched_tmm_case33.py` | `result/case33bw/pv_nrel_case33_newsplit_20261022/nrel_ha4_40_60/moment_matched/`；说明`01_IDEA/results/moment_matched_tmm_results_20261022.md` | 最大似然K=1节点18电压0.5%分位0.9936 p.u.（MC方法0.9727，GMM 0.9891）；按矩后K=1分块TMM 8个目标均优于AQGMM | supported（单次运行） |
| E-GROUPED-TMM-20261023 | C1 | `tests/python/grouped_strength_tmm_case33.py`、`truncated_strength_case33.py`、`grouped_nu_estimation_case33.py`、`plot_full_distribution_grouped.py` | `result/case33bw/grouped_strength_tmm_20261023/`；说明`01_IDEA/results/grouped_strength_tmm_results_20261023.md`第1—3.2.1节 | 两片云区分组正确；组间独立、同步、共同背景三种关系差别≤0.0006；截断只在K=1有改善；二次响应方向单独成组使K=1节点18电压峰度3.26（MC方法3.06） | supported（单次运行） |
| E-GROUPED-REC-20261023 | C1、C2 | 内联运行（命令见`01_IDEA/results/grouped_strength_tmm_results_20261023.md`第3.2.2节），调用`grouped_strength_tmm_case33.py`与`truncated_strength_case33.py` | `result/case33bw/grouped_strength_tmm_20261023/recommended_config/tail_and_ks.csv`、`moments.csv` | 节点电压尾段误差16组中15组最小或并列；K=1时其余对象KS最小；K≥2时KS比AQGMM大约0.01；算例1支路K=2、4多尺度t更小 | supported（单次运行，MC方法32768个场景） |

## 2026-10-21

| 证据编号 | 支撑结论 | 内容与来源 | 状态与范围 |
|---|---|---|---|
| E-WIND-FIXED-MIN-20261021 | S3-B, S3-C | IEEE 33节点18/25/33风电2.972 MW、50%预测出力、0.8倍负荷；共同强度、高斯对照、状态混合三组同均值同协方差输入，各45056次AC全部收敛，K1/3/6各25个候选收敛。方案01_IDEA/plans/wind_fixed_forecast_minimal_protocol_20261021.md，结果01_IDEA/results/wind_fixed_forecast_minimal_results_20261021.md，三目录result/case33bw/wind_fixed_forecast_case33_20261021{,_gaussian_control,_state_mixture}/run_manifest.json | 本地运行，用户授权。按预登记判据本文TMM三组均0/6通过，方法可行性在本场景未成立。K1时本文TMM全分布KS约为GMM的一半，四项预登记越限概率接近参考，GMM低估44%—58%；K3、K6结果混合。K3自由度共同强度组5.3—11.4、状态混合组约200，能区分强度变化与状态混合。白化坐标中其余方向对天气驱动方向的二次依赖在三组均存在。K增大精度不升的原因已由E-WIND-TRAINSIZE-20261021确认为训练样本不足 |
| E-WIND-TRAINSIZE-20261021 | S3-B, S3-C | 方案第6节：共同强度组训练样本增至32768个（种子2026102141），验证与参考输入与主运行逐字节一致，沿用锁定目标，K1/3/6/12，35个候选收敛；结果01_IDEA/results/wind_fixed_forecast_minimal_results_20261021.md第7节，目录result/case33bw/wind_fixed_forecast_case33_20261021_train32768/run_manifest.json | 本地运行。参考集似然最优K由3移到6，K≤6单调上升，K12因最小分量有效样本约550个而下降；6个目标K3→K6尾段误差两种模型全部下降；K6越限概率两种模型与参考之差约在0.2个百分点以内。本文TMM与标准TMM各1/6通过（支路24反送转受电，K3对GMM K6）。本文TMM K1越限概率在4/5事件上与GMM K6相当，属描述性比较 |
| E-WIND-SHARED-20261021 | S3-B, S3-C | 方案第7节，derivation.md式(B20)—(B22)：8192个固定训练场景，各分量共用其余方向残差协方差（SBTMM）及同结构高斯对照（SGMM），K1/2/3/4/6/8/12按验证似然选K；共同强度与状态混合两组，结果01_IDEA/results/wind_fixed_forecast_minimal_results_20261021.md第8节，目录result/case33bw/wind_fixed_forecast_case33_20261021_shared{,_state_mixture}/run_manifest.json | 本地运行。共同强度组最优K由4移到8，最优参考似然由−89.60提高到−89.04 nat，参数8579降为3930；SBTMM K8对GMM K4：KS 6/6、越限事件4/5、尾段4/6更优，预期1、2成立。改进几乎全部来自共用结构，SGMM K8与SBTMM K8接近；t分量作用集中在K1（越限概率平均相对误差7%对29%）。状态混合组SBTMM与SGMM相同。联合似然选K与单个电气量尾部精度不完全一致 |
| E-PV-SHARED-20261021 | S3-B, S3-C | 光伏固定预测共同强度组复测共用残差协方差：方案01_IDEA/plans/pv_shared_residual_protocol_20261021.md，结果01_IDEA/results/pv_shared_residual_results_20261021.md，入口tests/python/run_pv_shared_residual_case33.py（导入原光伏入口不修改），目录result/case33bw/pv_shared_residual_case33_20261021/run_manifest.json；输入与原光伏主运行逐字节一致，沿用原锁定目标 | 本地运行。最优K：GMM 6、SBTMM 8、SGMM 12；最优验证似然SBTMM −85.64、GMM −86.36 nat，参数5995对12869。SBTMM K8对GMM K6：尾段5/6、越限事件6/8更优，深度越限低估由约6—10倍降为约1.5—1.7倍。t分量独立作用按预设判据不成立（对SGMM K12尾段1/6、越限6/8）；同K时SBTMM似然高0.15—0.36 nat，K3尾段6/6更优，作用体现为少分量。深度越限参考事件数7—46 |
| E-PV-FIXED-FORECAST-20261021 | S3-B, S3-C | IEEE 33节点，12:15—12:30时段、50%预测出力、共同天气强度的文献约束输入；45056次AC全部收敛，27个候选收敛，10个模型各4次传播；目标对象拟合前锁定。结果记录01_IDEA/results/pv_fixed_forecast_results_20261021.md，运行清单result/case33bw/pv_fixed_forecast_case33_20261021/run_manifest.json，判定表在同目录common_strength/effectiveness_*.csv | 本地运行，用户授权。6个目标按第8.1节均未通过：K3—12时0.5%—5%尾段CDF误差TMM小25%—35%且超出传播范围，但0.5%分位TMM较差；其余对象K≥5全部误差相当。原因诊断为标准多元t对64个电压方向共用一个随机尺度，负荷小波动方向把自由度拉到30—200。诊断图为matplotlib，不作正式论文图 |
| E-PV-BLOCK-TMM-20261021 | S3-B, S3-C | 同一输入、目标和参考上的分块TMM（方案第12节）：光伏灵敏度确定8个共用强度方向，其余高斯；selected_models_block.csv、block_effectiveness_*.csv、block_global_repeat_summary.csv及diagnostic_pdfs/ | 本地运行。2/6目标通过（支路19、31反送转受电侧，同K尾段及K3少分量均成立，K6全分布KS约为GMM的1/4—2/5）；其余4个目标只在K1成立。自由度4.9—9.7。K1尾段优于GMM K12为事后描述，未预先登记 |
| E-PV-BLOCK-CHECK-20261021 | S3-B, S3-C | 方案第13节：独立训练重复（种子2026102121/22，参考集复用且SHA一致）与强度恒为1的高斯对照，均按主运行锁定目标判定；两目录各有run_manifest.json | 重复中分块TMM通过4/6（支路19、31重现，另加节点29、31），标准TMM 1/6；对照中分块TMM 0/6且KS与GMM相同，自由度60—200。优势归因于共同天气强度已由数值实验支持 |
| E-PV-K-BY-PV-BLOCK-20261021 | S3-B, S3-C | 按光伏驱动方向选择分量数，只用已估计模型；result/case33bw/pv_component_count_by_pv_block_20261021/，说明见01_IDEA/results/pv_component_count_by_pv_block_results_20261021.md | 分块TMM在5/6场景中0.5%—5%尾段误差降至已测K最小值，两块TMM在场景C同样；与共用残差复测组合后发现光伏方向似然随K几乎不变、规则总选单分量，深度越限概率平均相对误差185%对K8的24%，修正为不作一般规则 |
| E-PV-TWO-REGIONS-20261021 | S3-B, S3-C | 方案第15节C两片独立云区，目录含判定表multiblock_effectiveness_*、block_effectiveness_*及selected_models_multiblock.csv | 两块TMM通过4/6、分块TMM 2/6、标准TMM 0/6；两块TMM K1在支路31、20全分布误差约为GMM的1/4—1/5，K1—K3达到GMM所测K内未达到的精度；K6时28%其余对象误差偏高 |
| E-PV-STRONG-WEATHER-20261021 | S3-B, S3-C | 方案第15节场景A（云层强弱差别更大）与A+B（同时负荷取额定值），各自目录含run.log与判定表 | 分块TMM通过3/7、4/7，标准TMM均1/7；通过对象分块TMM K3达到预设精度而GMM未达到。实际单站偏差超额峰度约2.2（设定4.8），受出力边界限制。K5—6时分块TMM在约一半节点上全分布误差高于GMM，K1时约为其一半 |
| E-PV-ANALYTIC-RISK-20261021 | S3-C | 方案第14节，按博二稿特征函数与反演解析计算节点电压及线路越限概率；common_strength/analytic_limit_risk.csv、analytic_limit_risk_check.csv、analytic_limit_risk_summary.json | 二次型重构误差2×10⁻¹³，与数值积分分布差别≤0.11个百分点。K1时节点13—18过电压概率平均误差GMM 0.99、分块TMM 0.21个百分点；深度越限概率GMM在K6—12时低估约3—9倍、K1与K3时更多，分块TMM K≥3低估约2.5—3.9倍 |
| E-PV-LIMIT-PROB-20261021 | S3-C | 主运行越限概率描述表common_strength/limit_probabilities_descriptive.csv | 事后展示指标，未预先登记。GMM各K低估低电压、过电压概率3—10倍，分块TMM K12低估1.6—4倍，K1高估2—10倍；参考事件数7—46 |

| 证据编号 | 生成入口 | 原始结果文件 | 关键数值 | 状态 |
| --- | --- | --- | --- | --- |
| E-PV-CASE141-20261021 | `tests/python/run_pv_fixed_forecast.py --case case141`；`tests/python/analytic_limit_risk_general.py` | `result/case141/pv_fixed_forecast_case141_20261021_common_strength/`、`..._strong_weather/`（含`run_manifest.json`、`*_effectiveness_*`、`analytic_limit_risk.csv`） | 第8.1节判定：分块TMM 4/6、5/6，标准TMM 1/6、0/6；单分量越限概率平均绝对误差分块TMM为GMM的0.23—0.31倍 | supported（141节点两种碎云情形）；结果说明见`case141_results_20261021.md` |

## 状态约定

| 状态 | 含义 |
| --- | --- |
| supported | 当前证据支撑对应结论 |
| weakened | 证据只支撑较弱版本的结论 |
| revised | 证据促使结论表述、工程场景或证据状态被修正 |
| archived | 曾经有用，但不再支撑当前论文主线 |

正式图导出后按`figure_plan.md`登记图表证据包（`.fig`、`.png`、`.svg`、绘图数据CSV、`figure_manifest.jsonl`、检查报告），当前尚无正式图。
