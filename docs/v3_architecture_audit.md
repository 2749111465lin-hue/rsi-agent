# v3 RAG 架构审查与本地实现

日期：2026-10-07。范围：既有源码只读审查；仅新增 `code_rsi/v3/rag.py` 和根目录 `test_v3_rag.py`。本模块与测试不读取凭据、不访问网络、不加载真实问题或参考答案、不执行候选生成代码。其他 v3 集成由主协调器负责。

## 结论

现有 Code-RSI 已有程序档案、修改器、外部评分、配对测量和交付选择，但 BrowseComp P4 尚是独立的固定证据答案校准。新模块把“原问题→模型规划→搜索→模型读证据→根据所读事实提出下一查询→精确引用核验→简短回答”放在同一可注入执行核心中。它同时保留单次检索基线。

本次验证证明信息链和可信边界可工作，不证明真实 LLM 正确率提高。所有结果的 `correctness` 始终是 `unknown`，正确性只能由运行外部评分器判断。

## 旧架构的实际边界

1. `D:/Project/rsi-agent/main.py` 是早期 API 演示，旁边主要为 Game24/记忆修复实验。
2. 当前 RAG 项目为 `D:/Codex/Projects/RAG_RSI_Pilot_20260919`。`rag_engine.py` 是早期配置菜单引擎；`run_code_rsi_v257.py` 是最新续跑壳，复用 v256 的科学流程。
3. 程序搜索核心是 `code_rsi/v2/search_runner.py`、`development_agent.py`、`measurement.py`、`rag_services.py`、`experience.py`、`archive_policy.py`、`selection_report.py`。配对观察与传输处理分布在 v25/v251/v255/v256/v257。
4. BrowseComp 的 `pilot/offline_unused_transport_p5.py::replay_pair` 使用冻结 follow-up 执行回放。`pilot/paired_answer_p4.py::rebuild_frozen_messages` 从已冻结的 Presented spans 重建消息并比较答案；并未调用 RSI 搜索器。固定轨迹的选择器效果不能直接当作自适应 RAG/程序搜索收益。
5. 主引擎 `RAGBroker` 使用 2048 全消息本地 token 上限，默认根 1024 context/4 文档、回答 256 输出；P4 是 16384 输入/32768 输出。旧主评分为四类 normalized MacroEM，P4 为独立非官方盲评结果。迁移需显式任务、资源和评分适配。
6. 旧经验卡记录丰富，但 `board_from_cards` 主要向修改器传递行为组数、错误、最近失败、最好节点；跨任务机制经验检索尚未形成。

## 新模块契约

`RagEngine(backend, model, config=None).solve(task)`，仅标准库，无项目内相对导入，可复制为独立 `rag_core.py`。

- `task` 只接受 `question`、`instructions`、`task_id`。额外字段直接拒绝；task_id 仅用于返回身份，不进入模型 prompt。
- `backend.search(query, limit)` 返回文档对象列表：`docid`、`text`、`start`、`end`。end 为排他坐标，必须等于 start + len(text)。若 text 缺失，可通过可选的 `backend.read(docid, start, end)` 读取。
- `model.complete(stage, payload)` 返回 JSON 对象。该依赖可以连接真实模型或本地 fixture；核心从不自行读密钥或发请求。
- 模型与 backend 的异常只保留异常类型，避免异常消息带入适配器的私人数据。

### 模型阶段

| 阶段 | 返回字段 | 用途 |
|---|---|---|
| plan | constraints, queries | 识别公开问题约束并提出最初搜索 |
| read | claims, bridge_entities, gaps, conflicts, queries, ready | 阅读当前已呈现原文；提出有出处的主张、桥接实体与后续查询 |
| answer | answer, citation_ids, evidence_sufficient | 简短答案和宿主分配的精确引文 ID，不输出长推理 |

read 的每个 claim 含 text 与 citations。每个 citation 必须含 `source_id/start/end/quote`；坐标使用原文绝对字符区间。宿主仅接受当前 read payload 中已呈现的 source，并逐字检查对应子串。成功后产生 e1/e2 等不可由模型自定的引用 ID。

阶段字符串附加指导通过 `config['prompts']` 传入。候选可改指导，但不能用它替换宿主 JSON 字段验证。注入模型收到的是 JSON 深拷贝，不能通过修改 payload 改写宿主 sources、state 或共享 schema。

### 停止与资源

- `mode='single_pass'`：原问题检索一次，read 后直接 answer。
- `mode='iterative'`：plan 后依 read 的新 queries 继续；没有外部预填 follow-up。
- max_model_calls 始终为 final 留一次调用；紧预算下跳过后续 read，仍尝试 final。
- 重复规范化 query、没有新精确证据/有出处的桥接实体、没有 follow-up、达到轮数或预算即停止。只换写法的 claim 不算新证据进展。
- 新 sources 受每份字符和累计字符限制，截取后更新实际呈现 end；模型不能引用截取之外的字符。重复 source 不再次消耗累计字符预算。
- 最终 payload 超预算时删除完整条目，保留精确 quote；不重写/拼接引文。trace 记录省略项和最终呈现 citation IDs，final 只能引用这次最终 payload 里的 ID。即使冲突细节因预算省略，未解决冲突数量仍保留。
- `_meta.truncated=true` 或 finish_reason=length/max_tokens 的输出不可用；畸形 JSON/超长响应/模式错误记录失败。真实 provider 应在协议不完整时抛异常。
- 这里按模型调用数和字符边界控制流程；供应商 token、费用和物理传输 deadline 仍应由外层 provider/ledger 执行，不把字符计数宣称为实际 token 预算。

## 结果语义

| 字段 | 确切含义 |
|---|---|
| answer_usable | 有符合结构的非空答案；不意味着正确 |
| citations_valid | 最终所报 ID 都来自本次最终呈现证据，且源文 quote 已通过宿主核验 |
| model_claims_evidence | 模型是否自称证据足够；不由宿主伪装成 entailment 判断 |
| evidence_status | unsubstantiated / model_assessed_supported / model_assessed_conflicted |
| correctness | 始终 unknown，交给外部评分 |
| failure_types | 可供经验卡使用的 JSON 列表，单独记录各类失败 |

坏 citation 不覆盖格式有效的 answer：答案文本保留，citations_valid=false，记录 invalid_answer_citation。冲突不会被 ready 或 evidence_sufficient 静默抹除；若模型自称足够而冲突仍在，记录 model_claims_support_despite_conflict。

## 本地最小例子与验证

同一个完全虚构的问题：Where was the discoverer of the Aster comet born?

- 第一份文档：The Aster comet was discovered by Mira Vale.
- read 提出桥接实体 Mira Vale 与新查询 Mira Vale birthplace。
- 第二份文档：Mira Vale was born in Northport.
- 单次基线返回 Insufficient information；迭代路径返回 Northport，引用第二份文档的精确区间。

模型是确定性注入 fixture，只检验调用次序、第二跳信息进入回答、引用来源和停止机制，不是 LLM 能力或实验质量证据。

已执行项目虚拟环境 `python -B -m unittest test_v3_rag -v`：25/25 通过，0 跳过。覆盖第二跳与单次对照、禁止参考答案字段、task_id 不进入 prompt、伪造引文、错误坐标/来源、坏 citation 保留 answer、重复 query、无进展与伪进展、未落地桥接实体、final 预留、模式错误、截断、矛盾、来源裁剪、附加指导、backend 异常、可选 read、模型可变对象攻击与最终上下文压缩。

## 应统一的后续扩展点

保持外层程序档案、沙箱、修改器、配对测量和交付选择。将任务适配、检索 backend、RAG 核心、模型 provider、外部 scorer、经验卡格式分开，避免每次实验再复制一套运行器。P4 可通过任务适配器复用同一核心；旧 P4 原始冻结证据保持原样。

应精简的是 v254/v255/v256 大量平行 provider 方法、每个 run 脚本重复的 hash/freeze/authorization/ledger、分散的固定路径/价格/样本数和过期当前状态文档。保留既有隔离、实际支出账本、未决请求、参考答案屏障与同请求配对约束；历史版本继续作为冻结资产。