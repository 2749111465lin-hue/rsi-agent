# V3 RAG 反馈契约

此模块只从已有收据产生反馈，不调用模型、不读取题库、不训练权重，也不更改答案指标。实现入口是 code_rsi/v3/diagnostics.py，默认模块名与 experience_policy 一致。

## 接口

### diagnose_execution(receipt) -> dict

输入必须由宿主拥有的 execution 收据组成；当前支持 rag-rsi-v3-execution-*。独立 execution 尚无 role 可接受；显式 D_select/D_report 或冲突 split 拒绝。不得把裸 RagEngine 结果冒充宿主收据，应放进 candidate_reported。

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

返回含 schema、D_fit 身份、score、metric、summary、cases、module_priors、suggested_modules、paired_summary。case 保留：

- 当前题均分、代表性 repeat 的得分与预测，二者明确区分；
- host/model 分层诊断及可定位的收据路径；
- 最多两条宿主已核对来源的引用片段，每条至多 240 字符，附原始区间、完整 quote 的哈希及截断标记；
- 可选父代题均分、signed_delta、父代诊断类型和重复次数。

整个 state、完整 trace、原始模型消息和全部文档不会进入反馈。问题最多 800 字符，预测最多 400 字符；max_cases 范围 0–16，默认 4。现有合成压缩案例小于 7,000 序列化字符；这不是对所有输入的固定总长度保证。

同身份配对可通过 measurement["parent_measurement"] 传入父测量。父、子必须同 D_fit、panel_hash、evaluator_epoch、metric 和题目集合且均完整可测。signed_delta = 子题均分 - 父题均分，保留负值。输出配对是同题均分对照，不声称随机种子配对或因果归因。

推荐接入形态：

~~~python
feedback = compact_feedback(
    {**fit_result, "parent_measurement": parent_fit_result},
    fit_tasks,
)
# 无父代时不要传 parent_measurement；不要传 D_select / D_report。
~~~

## 事实与模型报告的界线

| 类型 | 宿主判据 | 初始模块优先级 |
|---|---|---|
| empty_retrieval | 完成的宿主 search.response_hash 精确等于规范 digest([]) | query_rewrite 1.0、retrieval 0.8 |
| no_retrieval | 完整宿主 trace 中没有 search | retrieval 1.0 |
| no_evidence_read | 完整宿主收据无完成的 read_presentations | evidence_selection 1.0 |
| no_verified_read_quotes | 已完成阅读，但宿主未记录任何来源匹配 quote | evidence_selection 0.9 |
| no_final_evidence | 与实际返回答案匹配的最终调用未呈现证据 | evidence_selection 1.0、answer_generation 0.8 |
| no_observed_final_answer / answer_empty | 宿主最终回答匹配失败 / 明确回答不可用 | answer_generation 1.0 |
| invalid_answer_citation | 宿主引用列表或来源/呈现一致性失败 | answer_generation 1.0 |
| model_parse_failure | 宿主已完成的 ModelResponseError 等，或对应错误响应哈希 | 已知 stage 对应模块 1.0 |
| repeated_query | 宿主实际搜索内容归一化后重复 | query_rewrite 1.0 |
| evidence_gap / evidence_conflict | 仅候选 state 或已观察模型回答中的报告 | 按相应设计优先级乘 0.35 |

未知模型错误名不会直接标为 parse failure。backend 的 read_calls=0 不代表没有模型证据阅读：搜索可直接返回文本。候选 trace 的 source_ids=[] 仅能表示没有新增，不能证明后端检索为空。empty_retrieval 表示至少一次搜索空返回，不能证明所有搜索失败，也不能证明最终答案错误。

引用逐字存在、来源匹配、原文坐标正确都不能证明它支持 claim，更不能证明已覆盖所有问句约束。模型的 evidence_sufficient=true 不会上升为宿主证据充分；false 只放入 model_reported。所有结构诊断都不能自动把 answer_usable 的程序转为无效或 Debug；程序是否执行失败仍由宿主 execution_ok/valid_program 决定。

## 案例和经验策略的连接

若存在配对，先选最大退化，再选最大改进（至少两个名额才能同时呈现），随后按新失败类别覆盖、结构信息量、分数和题目 ID 确定性填充。没有父代时先选高信息失败，并尽可能保留一个答案达标对照。不是简单选前四个失败。max_cases=0/1 时，全部配对的正/负数量、均值及极值仍保留在 paired_summary。

汇总以题为单位，一个题的多次调用或重复日志不会放大先验。每次 execution 内同模块取各症状的最大值；同题各 repeat 取最大值；最后各题等权平均。宿主最高 1.0，自报最高 0.35。常数与排序是当前实现的可测试设计推断，尚未通过真实预算消融证明最优。

experience_card 应分存 host_observed 与 model_reported，并保留来源。检索历史可用相同 failure code + target_module 聚合相关动作经验；模型报告只能作为次级相似性。经验策略仍应以同 panel/epoch 的宿主答案 signed gain、有效/失败状态、不确定性与实测成本决定是否继续该模块。冷启动时先验提供方向；其后应允许正负实测收益纠正方向。先验、引用数量和检索代理分不得加到 terminal answer reward。

该模块只提供契约；主流程和策略接入由调用方负责。不得声称新增此文件本身已经改变模块选择或提高正式题库成绩。

## 未知结果与角色隔离

UnknownProviderOutcome、HostError、LimitExceeded、pending/incomplete 等明确宿主状态返回 unavailable，score=None，cases=[]，无模块先验。完整测量某一行出现此状态，整份反馈不可用于学习；即便上游误写 score=0，也不转成答错案例。未知父代不可生成配对，直接拒绝。

无原始观测不等于事件未发生。只有具备完整 execution schema 与宿主 trace/evidence 的收据，才根据缺席判断 no_retrieval/no_evidence_read。诊断模块不是收据认证器：调用方仍必须保证宿主字段不能由候选程序覆盖。

D_select/D_report 的顶层、显式行角色、父测量或任务角色均被拒绝。没有 role 的行和公共任务只从通过身份校验的顶层 D_fit 继承；无角色顶层测量不允许。外部自定义答案指标没有 score_max 时不假定满分为 1；em/f1 默认目标为 1。报告与选择过程不能成为这里的反思材料。

## 文献对应及边界

- TextGrad 初版 2024-06-11，Nature 版本发表于 2025-03-19。其机制是将 LLM 文本反馈沿计算图回传给可修改变量；官方实例使用额外 backward engine。当前只借鉴“把标量失败连接到可修改模块”，不实现文本反向传播、不增加审阅模型调用，也不继承论文效果。[原文](https://www.nature.com/articles/s41586-025-08661-4)；[官方代码](https://github.com/zou-group/textgrad)。
- GEPA 初版 2025-07-25，当前 v2 为 2026-02-14。论文使用轨迹反馈修改模块提示，并保留跨实例互补候选；HotpotQA 示例将二跳查询对准缺失的关联信息。当前复用已有轨迹形成短反馈和对照，模块归因仍是待检验假设；没有把该过程声称为完整 GEPA 或保证同等提升。[原文](https://arxiv.org/pdf/2507.19457)；[版本记录](https://arxiv.org/abs/2507.19457)；[官方代码](https://github.com/gepa-ai/gepa)。

## 离线验证

python -B -m unittest test_v3_diagnostics -v

测试使用全部合成的问题、预测、引用与分数；不访问真实题库、参考答案或 API。覆盖角色/身份、引用与语义界线、空检索、缺阅读/最终证据、解析失败、未知结果、确定性、高信息案例、正负配对、重复次数、截断和字段白名单。它们证明实现契约，不证明 RAG 的实际提升。
