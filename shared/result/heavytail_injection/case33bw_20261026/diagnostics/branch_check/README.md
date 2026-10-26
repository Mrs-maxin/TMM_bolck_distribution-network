# 支路功率基础计算检查

- three_phase_unit_power_formula.csv：4种注入情形、全部32条在运线路；比较三相SI单位、标幺值、实虚部公式、复数公式与独立二次型。
- independent_acpf_vs_original_mc.csv：190个既有MC注入场景，包含普通与尾部运行点；独立求解后核对原MC节点电压、支路P/Q和总有功损耗。
- same_voltage_law_analytic_vs_direct_cdf.csv：K=2，4种注入情形、3种已有电压模型、64个支路P/Q，共768项。比较的是两种计算方式是否给出同一分布，不是各模型相对MC的误差。
- incomplete_analytic_check/：固定无功条件下的解析积分试算，仅部分模型输出完成，且部分线路未收敛。数据保留，未用于当前三方法精度判断。

现有基础检查中，独立支路功率与原MC的最大差约8.04×10⁻¹¹ MW/Mvar；同一电压分布的解析CDF与直接计算差在本次抽取电压的同时95%分布界内。检查范围和单位逐列见根目录field_dictionary.csv。以上不能解释成修正后的电压分布已经满足完整交流潮流。
