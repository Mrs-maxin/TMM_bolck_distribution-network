# 注入功率到节点电压的计算逻辑与学习资料

日期：2026-10-21。依据本项目源码、导入清单、PFIGMM指定提交的前作文稿及推导核对。本轮只读代码和资料，不运行潮流或拟合。

## 已确认的直接来源

MATLAB主流程的批量电压计算直接移植自PFIGMM项目，来源提交为`3734fa5141e05e541ab0703c6e44ccdfc5856d80`。[本项目导入清单](../data/provenance/matlab_import.json)逐项记录了来源文件。当前`map_power_samples_quadratic.m`与该提交的原函数，仅替换项目命名后内容一致，本轮已作文本比对。

前作文稿题名为[解析交流概率潮流中的电压联合分布估计](https://github.com/wuhanichina/PFIGMM/blob/3734fa5141e05e541ab0703c6e44ccdfc5856d80/02_PAPER/manuscript_zh.md)，重点读第2.1节和第3.2节。完整计算公式在[前作推导文件](https://github.com/wuhanichina/PFIGMM/blob/3734fa5141e05e541ab0703c6e44ccdfc5856d80/01_IDEA/derivation.md)第4节，第8节说明二阶项的来源。

按作者当前要求，TMM论文将完整GMM方法作为已发表前作处理。上述题名来自本轮读取的项目稿，正式发表题名、期刊和DOI仍须用正式书目替换。现有记录能够确认代码与推导来源，不能据其推定正式出版信息。

## 两条实际计算路径

| 路径 | 注入功率到电压的计算 | 本轮核对位置 |
|---|---|---|
| MATLAB批量电压流程 | 少量展开点完整交流潮流、二次局部近似、固定雅可比修正、自适应增加展开点、未通过样本完整交流求解 | `prepare_forward_data.m`或`prepare_general_voltage.m`调用`recover_training_voltage.m` |
| 当前固定预测光伏入口及所读归档云团程序 | 对每个注入样本独立执行极坐标Newton交流潮流，逐轮更新雅可比 | `tests/python/run_pv_fixed_forecast_case33.py`复用归档`ac`和`polar_ac` |

两条路径都先得到电压样本，再拟合联合TMM/GMM。其成本和数值误差应按具体入口说明。当前固定预测测试并未调用MATLAB的展开点批量映射，原始云团配置也明确写有`PtoV=False`。

## MATLAB批量电压的计算步骤

### 1. 给定注入，明确待求电压

固定拓扑、导纳和参考节点电压，其余节点为PQ节点。令

\[
s=[P_{\mathrm{PQ}};Q_{\mathrm{PQ}}],\qquad
x=[V_{R,\mathrm{PQ}};V_{I,\mathrm{PQ}}],\qquad s=F(x).
\]

注入取发电减负荷，采用标幺值。参考节点的注入由系统功率平衡决定。光伏电源在这里作为给定P/Q注入处理，不等同于恒电压控制的PV母线。

### 2. 选择一个展开点，计算完整交流潮流

程序先选择距离注入样本均值最近的实际样本，调用MATPOWER得到该点的电压\(x_a\)及注入\(s_a\)，再计算其潮流雅可比。

`build_anchor.m`先取电压幅值、相角下的雅可比，再通过坐标变换得到直角坐标电压对注入变化的局部关系。程序中的`inverseMatrix`对应\(B_a=J_a^{-1}\)，用于把功率不平衡量换算为电压修正量。

### 3. 先计算一阶电压变化

对附近任一目标注入s，使用

\[
h_1=B_a(s-s_a).
\]

这一步给出注入变化对应的初步电压响应。同一展开点的雅可比关系可同时作用于一批样本。

### 4. 用原交流二次关系修正初值

固定参考电压后，直角坐标功率关于电压是精确二次函数。在展开点附近有

\[
F(x_a+h)=s_a+J_ah+\mathcal Q[h,h].
\]

因此，将一阶电压代回原交流功率关系后，得到的功率不平衡量正好给出该点所需的二次项信息。程序按

\[
h_2=-B_a[F(x_a+h_1)-s],\qquad
x_p=x_a+h_1+h_2
\]

计算二阶电压初值，再用同一雅可比关系修正一次

\[
\widehat x=x_p+B_a[s-F(x_p)].
\]

这组计算直接对应`map_power_samples_quadratic.m`。程序通过原交流关系求值，不需要显式保存每个节点的二阶导数张量。精确二次关系用于构造近似电压，有限次反求电压仍需检查功率残差。

### 5. 继续消除功率不平衡量

本项目当前实现还调用`refine_fixed_jacobian.m`，沿用展开点的雅可比进行批量修正

\[
x^{(j+1)}=x^{(j)}+
\alpha_jB_a[s-F(x^{(j)})].
\]

若全步修正使残差增大或电压超出程序接受区间，则逐次缩短步长。达到功率残差与状态残差换算量要求的样本结束计算。

当前默认最多追加10次修正，功率残差无穷范数要求为\(10^{-8}\) p.u.，\(\|B_a\|_2\|s-F(x)\|_2\)要求为\(10^{-7}\)。这些是求解接受标准，后者并非无条件成立的实际电压误差上界。程序电压接受区间与研究中的欠压、过压限值分别设置。

### 6. 增加展开点，处理剩余样本

对未通过的样本，程序按当前残差换算量选择下一展开点，再求解一次完整交流潮流并重复批量计算。展开点上限默认12个；具体算例可覆盖配置。

达到展开点上限后，`recover_training_voltage.m`对仍未通过的样本调用`solve_sample_voltage.m`，逐样本执行MATPOWER交流潮流。计算过程因此包含少量展开点潮流和必要的失败样本补算，不预先保证每个算例都减少同样数量的完整交流求解。

## 当前固定预测光伏入口如何求电压

该入口从归档程序提取`ac`，其内部逐样本调用`polar_ac`。参考电压固定为\(1\angle0\)，其余节点从幅值1、相角0开始，以有功、无功不平衡量构造Newton方程，每轮重新计算极坐标雅可比并更新电压幅值和相角。

所读实现的功率残差容差为\(10^{-11}\) p.u.，最大40轮。收敛后将复电压实部、虚部保存为联合模型的拟合数据。该实现采用常规Newton交流潮流方程，不依赖注入首先服从GMM或TMM。

## 具体学习资料与用途

| 学习顺序 | 资料 | 要学习的内容及与本项目的关系 |
|---|---|---|
| 1 | [PFIGMM前作第3.2节](https://github.com/wuhanichina/PFIGMM/blob/3734fa5141e05e541ab0703c6e44ccdfc5856d80/02_PAPER/manuscript_zh.md) | 直接了解少数展开点如何用于批量计算电压，这是已确认的直接方法来源 |
| 2 | [PFIGMM推导第4、8节](https://github.com/wuhanichina/PFIGMM/blob/3734fa5141e05e541ab0703c6e44ccdfc5856d80/01_IDEA/derivation.md) | 学习一阶项、二次项和残差修正，以及二阶项为何由原交流二次关系确定 |
| 3 | [MATPOWER Technical Note 4](https://matpower.org/docs/TN4-OPF-Derivatives-Cartesian.pdf)，Sereeter与Zimmerman，2018，DOI:10.5281/zenodo.3237909 | 学习直角坐标交流功率及其导数；作为电气方程基础，不是已确认的整套批量算法来源 |
| 4 | [MATPOWER导数函数说明](https://matpower.org/docs/ref/matpower7.1/lib/dSbus_dV.html)与[Newton交流潮流说明](https://matpower.org/docs/ref/matpower7.1/lib/newtonpf.html) | 对照`build_anchor.m`的雅可比及归档`polar_ac`的逐轮求解逻辑；TN2给出极坐标导数基础 |
| 5 | 吴涵，[基于随机变量二次型的配电网交流概率潮流建模和解析计算方法(一)：从电量到功率量的推导](https://csee.publish.founderss.cn/zh/article/doi/10.13334/j.0258-8013.pcsee.242015/)，2026，46(3):928–941 | 学习得到联合电压后如何计算功率分布，属于电压到功率方向；不将其直接视为本项目注入到电压算法的出处 |

MATPOWER的软件引用为Zimmerman、Murillo-Sánchez和Thomas，2011，MATPOWER: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education，IEEE TPWRS，26(1):12–19，DOI:10.1109/TPWRS.2010.2051168，见[官方出版与引用资料](https://matpower.org/publications/)。

本轮没有发现将上述完整二次局部反求公式直接归于另一篇外部论文的明确记录。已确认的直接来源为PFIGMM前作及指定提交代码，常规Newton和导数资料分别承担求解器及电气方程依据。Gao等输入输出GMM论文在前作中作为另一种概率潮流路线综述，未被标注为这段二次批量映射的直接来源。

## 本项目代码阅读顺序

1. [展开点与雅可比](../+BTMM_core/+powerflow/build_anchor.m:35)。
2. [一阶、二阶初值与残差修正](../+BTMM_core/+inverse/map_power_samples_quadratic.m:24)。
3. [固定雅可比追加修正](../+BTMM_core/+inverse/refine_fixed_jacobian.m:20)。
4. [选择展开点与分配样本](../+BTMM_core/+inverse/map_samples_with_atlas.m:36)。
5. [未通过样本的完整交流求解](../+BTMM_core/+powerflow/recover_training_voltage.m:14)。

先掌握前两个文件的物理含义，再看多个展开点及残差接受逻辑，即可理解源荷样本如何转成联合电压模型所需的数据。TMM/GMM的概率建模在电压计算之后进行。
