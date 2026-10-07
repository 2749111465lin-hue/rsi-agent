# V3 任务指标与官方边界

`code_rsi/v3/task_metrics.py` 是独立、纯标准库的宿主评分模块。它不下载题库、不读取文件、不调用模型，也不修改公开任务或私有参考。评分应在生成结果冻结后进行；本次测试只验证规则和接口，没有真实模型效果结论。

## 接口

```python
from code_rsi.v3.task_metrics import score_task, aggregate_musique_full

metrics = score_task(prediction, private_reference, task=public_task)
# BCP 仅在协议明确允许规则诊断时：
proxy = score_task(prediction, private_reference,
                   task=public_task, allow_proxy_metrics=True)
# 收齐同一数据角色下完整的 Full 配对后：
full_metrics = aggregate_musique_full(scored_rows)
```

`prediction` 可为答案字符串或执行结果字典。答案字段支持 `answer` / 官方 `predicted_answer`；两者同时存在必须相同。`question_id` 若给出必须匹配参考；可用 `answer_usable=False` 明确标记已知交付失败。未知 API 物理结果必须先由宿主核账，不能传成空预测或零分。

返回一个字典，固定包含 `answer_em`、`answer_f1`、`support_em`、`support_f1`、`answerability`。分数范围为 0–1；`None` 表示不适用或不可评分，原因在 `metric_status`。还包含 `protocol`、`proxy_metrics`、`official_judge` 和 Full 所需的题目组及可回答性字段。结果不回显答案、支持文档标签或 decomposition；其中 gold 可回答性和分组信息仍属于宿主评分侧数据，不可反流到未完成的题目生成或 D_report 选择。

## MuSiQue 答案与支持段落

规则锚定上游提交 `922ac98f19a201998dbdae6d7f2887a5258dbdeb` 的 [evaluate_v1.0.py](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/evaluate_v1.0.py)。入口仅对 gold `answerable=True` 的分支累计普通答案和支持分，并输出三位小数。它输出答案 EM/F1 和支持 F1；本模块额外保留支持 EM 作为诊断，未将它冒充新的官方主指标。

答案复用本仓库 `datasets.evaluate_answer`：小写、移除 ASCII 标点和英文冠词、合并空白；EM 比较规范化字符串，F1 按词的出现次数求交集，并分别取所有合法答案别名中的最大值。这与已核验的 [AnswerMetric](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/answer.py) 对齐。显式提交空字符串时保留上游的空/空语义；完全缺答案或已知交付失败则明确记零，不把缺失输出当正确空答。

[SupportMetric](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/support.py) 比较支持**段落集合**，重复 ID 不增加权重，F1 为两集合交集大小的两倍除以集合大小之和。`support_set_metrics` 保留上游显式空集对空集得 1 的特例。它不衡量引文语义是否支持答案，也不替代宿主对来源、quote 和位置的校验。

预测支持接受三种一致的表示：

- `predicted_support_idxs`：官方整数段落下标，需要传入本题 `task`，映射至适配器的 `question_id/p/idx`；错误下标仍按官方集合规则计入假阳性。接口比上游隐式 `int()` 转换更严格，不接受字符串或布尔值充当整数。
- `support_docids`：宿主已明确解析的文档 ID。传入 `task` 时核对其局部范围。
- `citations`：现有执行收据中的引文列表，每项须有 `docid`，非空列表需要本题 `task`。只有 `citation_id`、来自别题的 ID 或混入无法映射的引文时，整个支持指标为 `None/unavailable_*`，不能静默删除坏项后拿剩余项评分。

多个表示同时提供必须指向同一集合。完全缺少预测证据字段，或显式空证据对非空 gold，支持分为零。适配器目前在支持标注缺失时也可能生成 `supporting_docids=[]`，因此空列表本身不能证明 gold 是真正空集：默认返回 `unavailable_empty_annotation_ambiguous`。只有私有参考显式提供、且宿主已核验的 `support_annotation_available=True` 才允许将空 gold 视为完整标注；该字段是本地可用性合同，不是虚构的官方字段。缺少标注键或明确 `False` 时不可评分。模块不猜测标注是否完整。

## MuSiQue-Full 配对聚合

[论文 §7.1](https://arxiv.org/pdf/2108.00573) 将 Full 定义为同一问题的充分/不充分上下文对。两条预测均正确判断可回答性，才保留可回答分支的答案或支持分；否则该对联合分为零。上下文仍各自保留 20 段，评分也不会合并语料。

这不是任意数量的 base-ID 分组：上述官方入口第 79–81 行明确断言每个 ID 出现两次；[答案联合指标](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/group_answer_sufficiency.py) 和 [支持联合指标](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/group_support_sufficiency.py) 同样要求两次 sufficiency 预测，并仅保存 gold 可回答分支的答案/支持。

`aggregate_musique_full` 使用私有 `pair_group_id`（适配器保存的官方 source ID），要求两条不同 public ID，且一条 gold True、一条 False。孤立分支、重复 public ID、两个同标签或三个以上变体显式失败。若后续数据确有多变体，须先核对官方版本并另定义配对协议，不能把任意组平均冒充 v1.0。

`answerability` 是逐条布尔预测正确率诊断；必须显式提供 `predicted_answerable`，绝不从答案是否为空、`abstained` 或引文数推断。不可回答分支的普通答案/支持为不适用，即使猜中原答案也不增加普通 QA 分。聚合返回普通答案/支持均值、`group_answer_sufficiency_f1`、`group_support_sufficiency_f1`，以及额外 EM、整对 sufficiency 诊断；按 [GroupMetric](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/group.py) 跨题目对等权平均，最后保留三位小数。独立单位是题目对。

缺失所需分数会使对应聚合指标为 `None`，不丢弃这些题目对来提高均值。这样明确区分“分数为零”和“当前资料不足以复现官方分数”。

## BCP、MultiHop-RAG 和 BRIGHT

BrowseComp-Plus 默认答案 EM/F1 为 `unavailable_official_judge`。只有显式 `allow_proxy_metrics=True` 才返回本地规则分，并标记 `proxy_rule_only`、`proxy_metrics=True`、`official_judge=False`。这些分数与官方 judge 不等价，不能据此声称官方基准提升。MultiHop-RAG 的本地 EM/F1 同样标记为规则代理；未扩展其官方 QA 判分。

检索侧辅助接口 `score_bright(ranked_docids, reference, *, excluded_docids, long_context=False)` 计算二元 nDCG@10。上游 [run.py](https://github.com/xlang-ai/BRIGHT/blob/main/run.py) 将 `gold_ids` / `gold_ids_long` 写成相关性 1，并要求 excluded 文档不在预测或 qrels；[retrievers.py](https://github.com/xlang-ai/BRIGHT/blob/main/retrievers.py) 使用 `pytrec_eval` 的截断 nDCG。本地函数接受已冻结排序而非原始分数，折扣为 `1/log2(rank+1)`，拒绝重复文档和 excluded 冲突，不擅自过滤后重排。空或缺少 qrels 返回不可评分。排序同分规则由调用协议固定；这不是完整上游检索评估报告，更不是 QA 准确率。

## 验证

离线执行 `python -B -m unittest test_v3_task_metrics test_v3_datasets -v`：43 项通过，其中新模块 23 项。测试使用手算的非平凡分数，覆盖重复词、别名、证据遗漏和假阳性、不可映射引文、Full 错误 sufficiency 归零、丢失分数不缩分母、非法多变体、BCP 明示代理及 BRIGHT 排名折扣。未下载真实题库、未读取封存答案、未调用付费 API，也未声称完成官方全量数据实跑。
