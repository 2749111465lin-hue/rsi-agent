# V3 题库组合、隔离接口与指标边界

核对日期：2026-10-07。实现范围为纯标准库适配与合成测试；未下载完整数据、未读取答案封存文件、未调用模型 API。本报告没有使用未经核验的 2026 年模型成绩，也没有声称当前模型已经在任何新增题库上取得结果。

## 决定

保留 MultiHop-RAG 作为现有回归集，新增 MuSiQue-Ans 作为可诊断的多跳开发任务，BRIGHT 作为独立检索侧轨，BrowseComp-Plus 作为高难确认任务。FRAMES 可在第二阶段检验跨题库泛化；LongMemEval 仅在研究范围明确包含会话记忆、时间过滤和知识更新时接入。

当前 P4 B0/D5 的低正确交付率伴随大量截断，不能据此认定题库不合适。首先应分开记录执行完成、截断/超时、检索文档、实际可见证据、最终答案与费用。所有预定题目仍进入主分母；同题多轮是重复测量，不应当作新增独立题目。

## 一手资料核对

| 题库 | 已确认的属性 | 用途和限制 |
|---|---|---|
| BrowseComp-Plus | 830 题，100,195 篇固定文档；人工 evidence/gold 文档标注；端到端答案使用模型裁判 | 适合迭代搜索、阅读与停止策略。高难、较高单题成本；应同时报告证据召回和交付完成。[数据卡](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus/blob/main/README.md) |
| MuSiQue | 约 25K、2–4 跳；原版每题 20 个候选段落；支持段落和问题分解标注；Full 增加不可回答对照 | 适合定位哪一跳丢失、证据充分后是否仍失败。原版是局部上下文任务，不是天然全库检索。[论文](https://arxiv.org/pdf/2108.00573) |
| BRIGHT | 约 1.4K 查询、12 个领域任务；固定文档及相关文档 ID；官方定位全部为评测数据 | 适合查询改写、推理后检索、重排；nDCG 提高不能直接等同 QA 正确率提高。数据卡版本统计应随 revision 固定。[官方仓库](https://github.com/xlang-ai/BRIGHT)、[划分说明](https://github.com/xlang-ai/BRIGHT/blob/main/Dataset_documentation.md) |
| HotpotQA | 113K Wikipedia 问答；句子级支持事实；distractor 10 段、fullwiki 固定语料；答案/支持事实/joint 指标 | 可用于较便宜的回归和既有研究对照。不宣称当前强模型已饱和；需要控制捷径与公开训练污染风险。[主页](https://hotpotqa.github.io/)、[语料说明](https://hotpotqa.github.io/wiki-readme.html) |
| FRAMES | 824 题，2–15 个 Wikipedia 页面，金答案、相关页面链接和推理类型；原论文使用模型裁判 | 可补充数值、表格和时间推理。接入时应冻结页面快照并验证标注证据可达，不能把动态网页与固定索引混算。[数据卡](https://huggingface.co/datasets/google/frames-benchmark/blob/main/README.md)、[论文](https://arxiv.org/html/2409.12941v1) |
| LongMemEval | 500 题，会话/轮次证据标签；覆盖信息提取、跨会话推理、知识更新、时间和拒答；S 每题约 115K tokens 历史 | 用于记忆研究。采用官方清理版本并冻结 revision；静态 RAG 项目无须为了新增题库而引入这套不同任务。[官方仓库](https://github.com/xiaowu0162/LongMemEval) |
| MultiHop-RAG | 2,556 题，证据跨 2–4 篇文档，含元数据信息；同时有检索与 QA 脚本 | 已接入的回归基础。题库扩展应补足诊断与泛化，不能替代执行故障修复。[官方仓库](https://github.com/yixuantt/MultiHop-RAG/) |

上述规模和结构属于来源事实；适用性、相对成本和优先级属于本项目的设计判断。当前固定模型上的改进空间仍须用相同预算下的无检索、现有检索、金证据诊断验证，不能从历史论文成绩直接推定。

## 本次代码接口

文件：`code_rsi/v3/datasets.py`。无导入时执行、文件读取、联网、模型调用、自动解密或下载行为。

四个函数都返回 `(public_task: dict, private_reference: dict)`：

- `adapt_musique(row)`：接受官方 `id/question`、20 个 `paragraphs`；严格验证段落编号和格式。
- `adapt_browsecomp(row, corpus_ref)`：接受已解码 `query_id/query`（或明确同义字段），只记录调用方提供的冻结语料引用。
- `adapt_multihop(row)`：接受 `query` 或 `question`，可带 `documents` 或 `corpus_ref`。原始行无 ID 时由公共问题生成稳定 ID；无语料字段时使用待解析的 `multihop-rag:corpus` 引用。
- `adapt_bright(row, corpus_ref)`：要求 `id/query/excluded_ids`，返回 `task_type="retrieval"`。它不会进入主 QA 评分。

公共任务固定包含 `id`、`question_id`、`dataset`、`question`、`task_type`、`documents`、`corpus_ref`、`corpus_scope`、`excluded_docids`。文档只允许 `docid/text/title/url`。答案、分解、中间答案、支持标记、人工推理和相关性标签均不会通过未知元数据混入公共结构。文档正文原本含有待寻找事实是正常检索输入；本隔离针对标注与评分泄漏，不会删除正文中的真实答案字符串。

调用方应在批量运行前调用 `validate_task_collection(tasks)`。相同问题 ID 的重复或冲突显式失败；同一共享语料引用内，同一文档 ID 的内容冲突也失败。不同题库/领域的源 ID 可能重合，组装混合面板时需用稳定命名空间或分开运行，不允许静默覆盖结果。

`validate_public_task(task)` 对额外公共字段拒绝执行，因此调用方若有运行日志、预算等信息应放在任务外层，不能直接把私有参考或任意元数据展开进任务字典。

`filter_documents(task, documents)` 是无需参考答案的运行时保护：

- 对 MuSiQue，只接受该实例原有文档的原文，任何跨题或被替换的段落均显式失败。
- 对 BRIGHT，移除 `excluded_docids` 中的文档。共享检索器应在排名前执行排除，并在回传给模型前再次执行保护；只保存排除字段而不接入过滤不构成有效隔离。
- 对所有任务，返回新建的允许字段文档，拒绝重复文档 ID。

共享 `corpus_ref` 只是显式引用，不会由本模块自动解析。运行器必须验证其实际内容/索引指纹，未解析的引用不可视为已有完整语料。允许字段结构不能单独证明运行进程没有权限读取私有文件；进程隔离、挂载和权限仍由调用方负责。

## MuSiQue 的局部作用域

每个公共实例的 ID 包括原题 ID 和仅由公共问题及段落计算的上下文指纹。Full 的可答/不可答样本可能共享原题 ID，但不同上下文得到不同实例 ID。原题配对 ID 仅放入私有参考，供成组充分性评估使用。改动答案或支持标签不会改变公共任务或公共 ID。

Full 通过缺失支持证据制造不可回答对照。不能把不同实例段落合库后仍使用其原答案性标签，因为被删证据可能从别的实例被找回。如果今后要做全库版本，必须独立命名为派生任务、重建答案性规则并保留原版对照，不能与官方原版成绩混称。[论文的上下文构造](https://arxiv.org/pdf/2108.00573)

## 评分界限

`evaluate_answer(prediction, reference, metric="em"|"f1") -> float` 是离线规则评分，使用小写、ASCII 标点、英文冠词和空白归一化，取所有参考别名中的最大值。空参考、未知任务、未知指标显式失败。运行器负责在评分前将未完成交付计为失败，不能从草稿或截断内容提取答案后冒充完成。

| 任务 | 本模块支持 | 不应声称 |
|---|---|---|
| MuSiQue-Ans | 与官方答案脚本相同类型的归一化答案 EM/token F1，支持别名 | 单独答案分数不等于支持事实正确、全链条正确或 Full 成组充分性正确。[官方答案评分](https://github.com/StonyBrookNLP/musique/blob/main/metrics/answer.py) |
| MuSiQue-Full | 对可答样本可做答案诊断；不可答样本在普通 EM/F1 中显式拒绝 | 尚未实现官方成组充分性、支持 F1；不能把不可答样本当普通字符串题计分。[官方总评分](https://github.com/StonyBrookNLP/musique/blob/main/evaluate_v1.0.py) |
| BrowseComp-Plus | 可显式计算内部规则代理分数，参考标记 `official_metric="llm_judge"`、`rule_metrics_are_official=False` | 规则 EM/F1 不等于官方答案准确率。2025 论文使用 GPT-4.1，当前官方说明使用 Qwen3-32B；不同裁判版本不能混算。[裁判说明](https://github.com/texttron/BrowseComp-Plus/blob/main/docs/llm_as_judge.md) |
| BRIGHT | 公共检索任务和排除规则 | 本模块未实现 nDCG/Recall，且拒绝 QA 答案评分；不能把检索结果当作端到端正确答案。[数据字段](https://huggingface.co/datasets/xlangai/BRIGHT/blob/main/README.md) |
| MultiHop-RAG | 内部归一化答案 EM/F1 代理指标，金证据保留在私有参考 | 不是对官方 QA 脚本的复制。所打开的上游脚本按预测和金答案词集合有无交集判定；项目必须固定实际使用的脚本 revision，并明确指标名。[上游 QA 脚本](https://github.com/yixuantt/MultiHop-RAG/blob/main/qa_evaluate.py) |

BrowseComp-Plus 原论文的每篇文档前 512 tokens 截断属于检索阅读设置，和当前程序总输出或执行窗口截断不同。文档 ID 被召回，不证明其含答案的片段实际对模型可见；因此证据召回和可见证据覆盖应分别记录。[原论文检索设置](https://arxiv.org/html/2508.06600v1)

## 开发与确认隔离

1. 已反复查看的 32 题只作诊断或开发，不重新标成未见确认集。保留历史暴露清单与运行次数。
2. MuSiQue 开发使用官方 train 中预指定的跳数分层；Full 成对样本不可拆到开发/确认两侧。需要检查源子问题重叠，不能只去重最终题号。官方提供 dev/test 所用种子单跳问题 ID，使用 SQuAD/NQ 等种子数据时需排除。[官方污染提醒](https://github.com/StonyBrookNLP/musique#data)
3. BRIGHT 官方全部是 test。若本项目对部分题目反馈调参，应明确标为内部开发划分，剩余未暴露题目/领域才用于确认；不能仍把全量成绩称为未见评测。
4. 由独立评分进程持有 private_reference。改进器只能访问开发任务、预算和允许的开发反馈；确认问题的答案、相关性标签、逐题裁判结果与轨迹不回流。
5. 固定基础模型、程序版本、语料/索引哈希、工具协议、随机性设置、题目清单、停止预算和评分器，再进行确认。根据确认结果改程序后，原确认切片自动成为已暴露数据。
6. 费用和失败在全部预定题上核算。确认报告使用题级配对差异，并把重复轮次作为同题重复测量；逐级扩样的停止条件须提前固定。
7. 共享公共语料不等于答案泄漏，但支持文档、近重复问题、事实链及子问题重叠应被单独记录。不能声称仅靠题号分离已经证明跨任务泛化。

## 已完成验证与未完成范围

合成测试命令：`D:\python\python.exe -B -m unittest test_v3_datasets -q`。

本次 20 项测试全部通过，覆盖允许字段剥离、私有标签不改变公共 ID、Full 同题不同上下文、局部语料隔离、公共/私有对象脱钩、BRIGHT 排除、问题与文档 ID 冲突、缺失参考、规则 EM/F1 别名、不可答评分边界、格式歧义及额外公共字段拒绝。

测试只证明适配器契约及合成场景的行为。尚未验证真实数据 revision、完整索引、所有官方原始格式变体、上游运行器过滤接入、官方模型裁判或任何真实题库成绩。新增题库运行和付费调用不包含在本次离线实现中。