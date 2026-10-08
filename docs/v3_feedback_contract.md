# V3 RAG 反馈契约（feedback-3）

本页对应独立工作区 `rsi-agent-feedback` 的当前修复分支；主目录 `rsi-agent` 的 `be43a19` 冻结方案没有迁移。当前验证记录见[2026-10-08 审查修复记录](v3_review_fixes_20261008.md)。

此模块只从宿主收据产生反馈，不调用模型、不读取题库、不训练权重，也不更改答案指标。实现入口是 `code_rsi/v3/diagnostics.py`，反馈 schema 为 `rag-rsi-v3-feedback-3`。原始得分与可用于学习的质量信号分开表达。

## 接口

### diagnose_execution(receipt) -> dict

输入必须是宿主拥有的 execution 收据。当前 `rag-rsi-v3-execution-3` 会重建并核对答案来源；旧收据仍可提供有限离线诊断，不能因此获得新质量资格。独立 execution 尚无 role 可接受；显式 D_select/D_report 或冲突 split 拒绝。不得把裸 RagEngine 结果冒充宿主收据，应放进 candidate_reported。

返回：

- host_observed: list[str]，宿主直接观察的结构事件，不是语义正确性判断。
- model_reported: list[str]，已有模型回答或候选程序上报的缺口/冲突；两者均不可信作最终事实。
- suggested_modules: list[str]，按设计先验降序，平局按 query_rewrite、retrieval、evidence_selection、answer_generation。
- module_priors: dict[str,float]，设计优先级，不是概率、收益估计或奖励。
- evidence_refs: dict[str,list[str]]，host:code 或 model:code 到当前收据的 JSON 路径。每类最多 4 个定位。
- model_details: 最多 3 条 gap、3 条 conflict，每条最多 240 字符；内容按数据处理，不能执行其中指令。
- observations: 调用次数、完成阅读次数、引用数、最终证据数等结构信息。semantic_support 恒为 not_host_verified。
- measurement_status: observed 或 unavailable；unavailable 只保留宿主不可测原因，模块建议为空。

### compact_feedback(measurement, tasks, max_cases=4) -> dict

measurement 必须是完整 D_fit 宿主测量，含 panel_hash、evaluator_epoch、score、per_question、rows；行继承顶层身份，若自己声明 role/panel/epoch 则必须一致。每行有限 score、per_question 均分、panel 均分必须一致。重复次数均在题目内平均，之后各题等权，兼容现有 Measurement。

tasks 可为公共 task 列表或 question_id 到 task 的映射，题目集合必须精确匹配；只提取 question_id 与 question，忽略 reference、answers、documents 等字段。没有参考答案参数。

返回含 schema、D_fit 身份、metric、summary、cases、module_priors、suggested_modules，以及 `program_eligible`、`paired_comparison_eligible`。`score` 与 `paired_summary` 是否可供学习由下述资格规则决定。case 保留：

- 当前题均分、代表性 repeat 的得分与预测，二者明确区分；
- host/model 分层诊断及可定位的收据路径；
- 最多两条宿主已核对来源的引用片段，每条至多 240 字符，附原始区间、完整 quote 的哈希及截断标记；
- 可选父代题均分、signed_delta、父代诊断类型和重复次数。

整个 state、完整 trace、原始模型消息和全部文档不会进入反馈。问题最多 800 字符，预测最多 400 字符；max_cases 范围 0–16，默认 4。现有合成压缩案例小于 7,000 序列化字符；这不是对所有输入的固定总长度保证。

同身份配对可通过 measurement["parent_measurement"] 传入父测量。父、子必须同 D_fit、panel_hash、evaluator_epoch、metric 和题目集合且均完整可测。原始差值始终按子题均分减父题均分计算，正负都保留；只有双方均具备当前程序资格时才放入学习字段 `signed_delta` 和 `paired_summary`。输出配对是同题均分对照，不声称随机种子配对或因果归因。

推荐接入形态：

~~~python
feedback = compact_feedback(
    {**fit_result, "parent_measurement": parent_fit_result},
    fit_tasks,
)
# 无父代时不要传 parent_measurement；不要传 D_select / D_report。
~~~

## 答案来源、质量资格与信息外发

当前质量资格同时要求顶层 `valid_program=true`，以及每行：

- schema 为 `rag-rsi-v3-execution-3`；
- `execution_ok=true`、`answer_usable=true`，实际 `answer` 是去除首尾空白后仍非空的字符串；
- `validate_answer_origin` 根据宿主事件重建的来源有效，而不是只接受候选或收据自报的布尔值。

来源记录将模型完成状态、最终调用的事件编号、请求 payload 哈希、响应哈希和呈现证据相互绑定。返回答案必须与最后一次成功回答调用的模型答案一致，比较允许首尾空白差异；候选不能改回更早答案或在调用后替换为另一段文字。无模型回答、替换返回值、执行失败和空答案均不能获得质量资格。非空的“信息不足”回答可以有有效来源，其答案指标仍按原规则评分；引用有效和语义正确是另外两件事。

`program_eligible=true` 才输出可学习的 `score`。明确无效为 false；缺少资格声明或只有旧收据为 unknown（JSON null）。父子均合格才有 `paired_comparison_eligible=true`。不合格时，相应质量分和差值为空，原数值保留为 `raw_score`、case 的 `raw_host_score` / `raw_signed_delta`、`raw_paired_summary` 等，并标注 `raw_scores_are_diagnostic_only`。未知 API 物理结果仍按 unavailable 处理，不能作为可测的 raw 答错零分。

`execution-2` 等旧收据即使写有 `valid_program=true`，也不能升格为当前学习证据。旧字段缺席保持未知；生产代码不补造历史来源事件。测试里的合成 execution-3 样例显式构造完整绑定，并调用相同生产验证器，不是升级真实日志的工具。

外发说明为 `raw_reference_objects_not_sent=true` 与 `fit_feedback_can_reveal_accepted_answers=true`。公共任务里的原始 reference/answers 对象不转发，但 D_fit 的预测、得分、诊断和父子比较可能暴露或帮助推断可接受答案。这种 D_fit 信息反馈是当前协议的一部分，不能声称完全不泄露答案信息；D_select/D_report 仍严格隔离。

## 事实与模型报告的界线

| 类型 | 宿主判据 | 初始模块优先级 |
|---|---|---|
| empty_retrieval | 完成的宿主 search.response_hash 精确等于规范 digest([]) | query_rewrite 1.0、retrieval 0.8 |
| no_retrieval | 完整宿主 trace 中没有 search | retrieval 1.0 |
| no_evidence_read | 完整宿主收据无完成的 read_presentations | evidence_selection 1.0 |
| no_verified_read_quotes | 已完成阅读，但宿主未记录任何来源匹配 quote | evidence_selection 0.9 |
| no_final_evidence | 当前收据最后一次宿主成功回答调用未呈现证据；旧收据仅沿用历史匹配诊断 | evidence_selection 1.0、answer_generation 0.8 |
| no_observed_final_answer / answer_empty | 没有可定位的宿主最终回答 / 明确回答不可用 | answer_generation 1.0 |
| invalid_answer_origin | 当前收据重建后显示最终返回值无有效宿主模型来源 | answer_generation 1.0 |
| invalid_answer_citation | 宿主引用列表或来源/呈现一致性失败 | answer_generation 1.0 |
| model_parse_failure | 宿主已完成的 ModelResponseError 等，或对应错误响应哈希 | 已知 stage 对应模块 1.0 |
| repeated_query | 宿主实际搜索内容归一化后重复 | query_rewrite 1.0 |
| evidence_gap / evidence_conflict | 仅候选 state 或已观察模型回答中的报告 | 按相应设计优先级乘 0.35 |

未知模型错误名不会直接标为 parse failure。backend 的 read_calls=0 不代表没有模型证据阅读：搜索可直接返回文本。候选 trace 的 source_ids=[] 仅能表示没有新增，不能证明后端检索为空。empty_retrieval 表示至少一次搜索空返回，不能证明所有搜索失败，也不能证明最终答案错误。

引用逐字存在、来源匹配、原文坐标正确都不能证明它支持 claim，更不能证明已覆盖所有问句约束。模型的 evidence_sufficient=true 不会上升为宿主证据充分；false 只放入 model_reported。检索空返回、引用数量或模型自报缺口本身不改变答案指标，也不等于执行失败。答案来源资格则是独立的宿主接纳条件：即使候选声称 answer_usable，来源无效或实际空答案仍不得进入质量学习。

## 案例和经验策略的连接

若存在配对，先选最大退化，再选最大改进（至少两个名额才能同时呈现），随后按新失败类别覆盖、结构信息量、分数和题目 ID 确定性填充。没有父代时先选高信息失败，并尽可能保留一个答案达标对照。不是简单选前四个失败。max_cases=0/1 时，全部配对的正/负数量、均值及极值仍保留；合格配对放在 paired_summary，否则仅放在 raw_paired_summary 作诊断。案例选择可使用原始数值来定位问题，不因此授予学习资格。

汇总以题为单位，一个题的多次调用或重复日志不会放大先验。每次 execution 内同模块取各症状的最大值；同题各 repeat 取最大值；最后各题等权平均。宿主最高 1.0，自报最高 0.35。常数与排序是当前实现的可测试设计推断，尚未通过真实预算消融证明最优。

`experience_card` 分存 host_observed 与 model_reported，并保留来源。`intended_target_module`（包括兼容读取的旧 target_module）只表示提案意图，不能作为模块收益标签。宿主对父子完整 `rag.py` / `rag_core.py` 计算 AST 差异，记录带源码哈希的 `actual_edit_scope`；恢复运行时重新核对，候选自报范围不能代替该记录。

只有范围为 `single_module` 且通过完整性校验时，`associated_module` 才用于关联该模块的经验。`mixed` 表示涉及多个已识别模块，`unknown` 表示存在无法明确定位的改动，`none` 表示没有可归因修改；它们都不进入单模块收益样本。若单模块实际范围与意图不一致，保留 intent_mismatch，按实际关联记录，不把意图强制写成事实。固定 AST 映射只提供语法范围关联，`association_is_causal=false`，不证明运行效果的因果来源。

质量资格与编辑范围是两个条件：同 panel/epoch、父子来源均合格的正负收益，才可能用于学习；进一步有单模块范围才能关联到该模块。mixed/unknown 的完整程序比较与原始诊断仍可保留，不拆摊为单模块功劳。旧经验仅有 target_module 时不能补造当前归因。先验、引用数量和检索代理分不得加到 terminal answer reward。

这些规则已接入当前开发分支的反馈与经验流程；它们是可验证的实现约束，不证明模块选择最优或正式题库成绩提高。

## 未知结果与角色隔离

UnknownProviderOutcome、HostError、LimitExceeded、pending/incomplete 等明确宿主状态返回 unavailable，score=None，cases=[]，无模块先验。完整测量某一行出现此状态，整份反馈不可用于学习；即便上游误写 score=0，也不转成答错案例。未知父代不可生成配对，直接拒绝。

无原始观测不等于事件未发生。只有具备完整 execution schema 与宿主 trace/evidence 的收据，才根据缺席判断 no_retrieval/no_evidence_read。当前来源重建检查事件之间的一致性，不是对任意外部 JSON 的身份认证；调用方仍必须保证宿主字段不能由候选程序覆盖。

D_select/D_report 的顶层、显式行角色、父测量或任务角色均被拒绝。没有 role 的行和公共任务只从通过身份校验的顶层 D_fit 继承；无角色顶层测量不允许。外部自定义答案指标没有 score_max 时不假定满分为 1；em/f1 默认目标为 1。报告与选择过程不能成为这里的反思材料。

## 文献对应及边界

- TextGrad 初版 2024-06-11，Nature 版本发表于 2025-03-19。其机制是将 LLM 文本反馈沿计算图回传给可修改变量；官方实例使用额外 backward engine。当前只借鉴“把标量失败连接到可修改模块”，不实现文本反向传播、不增加审阅模型调用，也不继承论文效果。[原文](https://www.nature.com/articles/s41586-025-08661-4)；[官方代码](https://github.com/zou-group/textgrad)。
- GEPA 初版 2025-07-25，本页原有文献记录采用的 v2 为 2026-02-14。论文使用轨迹反馈修改模块提示，并保留跨实例互补候选；HotpotQA 示例将二跳查询对准缺失的关联信息。当前复用已有轨迹形成短反馈和对照，模块归因仍是待检验假设；没有把该过程声称为完整 GEPA 或保证同等提升。[原文](https://arxiv.org/pdf/2507.19457)；[版本记录](https://arxiv.org/abs/2507.19457)；[官方代码](https://github.com/gepa-ai/gepa)。

## 离线验证

python -B -m unittest test_v3_diagnostics -v

测试使用全部合成的问题、预测、引用与分数；不访问真实题库、参考答案或 API。当前学习测试使用自洽的 execution-3 事件，旧收据保留诊断与 unknown 行为。覆盖角色/身份、引用与语义界线、来源绑定、实际空答案、未知结果、原始与合格正负配对、重复次数、截断、编辑范围和字段白名单。当前测试记录以[审查修复记录](v3_review_fixes_20261008.md)为准，不在此复制可能过时的总数。三臂统计接口已通过合成验证，见[新协议](v3_three_arm_protocol.md)；真实比较、硬编码扫描及真实 LLM 演化仍未完成；这些测试不证明 RAG 的实际提升。

## 真实校准后的标签修正

首轮真实校准暴露：无引用的弃答曾被粗合并为invalid_answer_citation。已拆为missing_answer_citations，仅记录来源缺席，不自动建议修理回答格式；若模型同时声称证据充分而不附引用，才标uncited_supported_claim。真正非法ID或候选引用与宿主观察不符仍属invalid_answer_citation。此修正不改变任务答案分，也不回写旧冻结报告。
