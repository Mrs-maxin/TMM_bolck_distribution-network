# Writing Lexicon

This rule file defines wording checks for AI-generated research notes,
manuscript drafts, README text, code comments, review responses, and figure
captions in this template.

## Preferred Style

- State the technical conclusion directly.
- Write innovation points as positive statements about the minimum identifiable
  research object, concrete engineering object, contribution-appropriate
  relation, theoretical significance, engineering value, mathematical model,
  engineering setting, measurable result, and evidence source. The relation
  may be physical causality, model mapping, computational property, control
  response, validity condition, observability, or an engineering decision link.
- Use complete declarative sentences.
- Start with the system object, model term, constraint, data source, metric, or
  numerical result.
- Name the contribution-appropriate relation, affected object, case set,
  metric, and evidence state. Use "physical mechanism" only when the evidence
  establishes a physical causal chain and a testable prediction.
- In Chinese power-system prose, name the physical object first: bus, branch,
  feeder, source/load process, operating scenario, voltage, power flow, current,
  constraint, probability distribution, violation event, or recovery action.
- In `BTMM_note.md`, `01_IDEA/story.md`, `01_IDEA/claims.md`, and
  innovation anchors, use field-recognized terms, engineering objects, physical
  quantities, model relations, and verifiable case names in body text.
- Use plain technical verbs such as compute, estimate, solve, bound, compare,
  validate, export, register, update, and test.
- Attach an intensifier to a number, tolerance, scale, or explicit evidence
  source.

## Rewrite Checks

Before releasing AI-generated text, scan for these patterns:

- The technical statement is missing the minimum identifiable research object,
  model relation, measurable result, or evidence source. Add those elements
  directly.
- Human-facing notes, story files, and manuscript prose should organize
  innovation points as minimum identifiable research object, theoretical
  significance, engineering value, model, result, engineering setting, evidence
  source, and next experiment.
- Negative buildup and fragmentary emphasis. Replace them with one complete
  sentence.
- Rhetorical question setup. Replace it with the technical condition or result.
- Prompting phrases that introduce an explanation. Start with the object or
  formulation.
- Meta commentary about the section, future explanation, or authorial process.
- Business jargon and promotional verbs.
- Over-mathematized or programming-shaped carriers. If the reader-facing point
  is about a power-system object, rewrite the sentence around the physical
  object and its equation, constraint, state, or metric.
- Vague consequence statements. Name the concrete implication.
- Scope-free extreme terms. State the case set, time window, model class, solver,
  dataset, or assumption range.
- Hard-banned over-mathematized or programming-shaped labels. If generated text
  contains any token listed in `Hard Banned Terms`, rewrite it before release
  around the power-system object, equation, constraint, state, metric,
  calculation procedure, or model parameter.
- Over-mathematized terms listed in `Use With Care Terms`. Prefer a
  power-system object, equation, constraint, state, metric, calculation
  procedure, or model parameter unless the term is mathematically necessary and
  defined in the derivation.
- Coined method names, internal experiment configuration aliases, temporary
  parameter-group names, branch names, or script abbreviations in body text.
  Move them to evidence ids, entry-file fields, result paths, run manifests,
  Git records, or configuration fields, and rewrite the sentence around the
  engineering object, physical relation, model name, metric, or case name.
- Em dashes. Use commas or periods.
- Empty intensifiers. Delete them or attach a number.

## Use With Care Terms

The following terms are not hard banned, but they should usually be rewritten in
reader-facing prose because they often make a power-system claim sound more
abstract than the engineering object requires.

| Type | Terms | Prefer |
| --- | --- | --- |
| Over-mathematized Chinese labels | 算子, 可微 | equations, mappings, constraints, states, physical relations, model parameters, or calculation steps |
| Over-mathematized English labels | operator, differentiable | equations, mappings, constraints, states, physical relations, model parameters, or calculation steps |

## Hard Banned Terms

The following terms may appear only in this lexicon or in a quoted source being
criticized. They must not appear in `BTMM_note.md`, `01_IDEA/story.md`,
`01_IDEA/claims.md`, innovation anchors, introductions, abstracts,
contribution lists, captions, or reader-facing README prose.

| Type | Forbidden tokens | Rewrite around |
| --- | --- | --- |
| Over-mathematized Chinese labels | 证书, 零空间 | power-system equations, constraints, states, feasible checks, measurements, physical relations, or model parameters |
| Programming-shaped Chinese labels | 范式 | methods, models, calculation procedures, input-output mappings, metrics, or engineering decisions |
| Over-mathematized English labels | certificate, nullspace | power-system equations, constraints, states, feasible checks, measurements, physical relations, or model parameters |
| Programming-shaped English labels | paradigm, estimator | methods, models, calculation procedures, input-output mappings, metrics, or engineering decisions |

## Avoid Terms

Use these tokens only when quoting source material or documenting a writing rule:

- navigate
- unpack
- landscape
- deep dive
- game-changer
- at its core
- it is worth noting
- it turns out
- as we will see
- in this section
- when it comes to
- think about it
- here is what I mean
- significant
- structural
- fundamentally
- inherently
- simply
- really
- every
- always
- never
- all

## Power-system wording in conversation and documents (user requirement 2026-10-21)

All replies and reader-facing documents must use power-system probabilistic power-flow language. Computer-science and machine-learning vocabulary is not allowed in explanations, even when the underlying code uses it. Describe what the injection power does, how voltages and branch powers respond, and how a probability is calculated.

| Do not write | Write instead |
| --- | --- |
| 训练集、验证集、测试集、参考集 | 用于估计电压分布参数的源荷场景；用于选择分量数的另一组源荷场景；逐场景交流潮流计算得到的精确结果 |
| 样本、事件数、N个事件 | 源荷场景、场景数；概率应直接给出数值及其计算方法，不以计数作为方法结果 |
| 种子、随机流、重抽样 | 不出现；需要时说明“另一组独立生成的源荷场景” |
| 对照组、对照 | 比较情形，并写明其注入功率规律，例如“天气强度恒定的高斯波动情形” |
| 标签、训练、学习、过拟合、泛化 | 分量归属、参数估计、分量过多导致的估计偏差、在其他运行条件下的适用性 |
| 传播、冒烟测试、管线、候选 | 由电压分布计算电气量分布、程序检查、计算流程、待选参数组 |

Limit-violation risk is computed analytically from the voltage distribution model, following the characteristic-function and inversion route of the PhD-year-2 draft, rather than by counting violating scenarios. The exact AC result used for comparison is described as the violation frequency obtained by solving AC power flow for each source-load scenario. Scenario descriptions should start from the injection-power law: forecast output, deviation distribution, spatial correlation between stations, and how the deviation magnitude varies with weather.
