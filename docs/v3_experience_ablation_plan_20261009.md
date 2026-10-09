# 经验策略审查与下一版对照规格

日期：2026-10-09。核对源码：`528825f3479d0fea628dd5ef350ab5f094ae299d`。本次只读代码与运行合成宿主函数，真实 API 为 0；没有修改运行源码、MuSiQue 冻结方案、题目或评分。这是已核对缺口与后续待测规格，不是已经实施的消融或效果结论。

实施更新：独立分支已按本规格增加受控反馈、终结机会覆盖和机制记忆开关；以下审查描述保留为修改前证据。[实现与验收](v3_experience_controls_20261009.md)。未运行真实演化对照。

## 1. 当前实现与三个缺口

当前 `choose_next` 完全确定性：优先修复失败父节点，否则选已有最高答案分父节点；按固定步数探索父节点或未测模块。模块依据有符号父子分差、描述性不确定度、失败及成本排序；缺收益时用诊断先验，无先验才轮换。它不是概率抽样，也不是经过校准的置信界算法。

| 已核对问题 | 代码位置 | 实际边界 |
|---|---|---|
| 无归因样本时先验可能持续锁定模块 | `experience_policy.py` 的 `elif not known` / `diagnostic_choice` | read 提示同时影响查询与证据，记为 mixed；它不产生单模块收益。每次仍可能选最大先验，没有按未归因尝试次数消退。真实发生率未知 |
| 保存了机制假设，发给修改器的记忆却不含它 | `evolution.experience_card` / `experience_policy.memory_for_action` | 原卡有 hypothesis，记忆投影未输出。非当前父节点的 mixed/unknown 卡通常也不入选；不能把现状称为通用修复技巧库 |
| 仅总分对照没有独立输入契约 | `ProgramDeveloper.prepare_request` / `compact_feedback` | feedback 含预测、逐题分、引句和执行流，decision 与 experience 也含诊断；删 execution_flow 或设 max_cases=0 仍不是仅总分 |

实际 AST 归因已经修好。mixed/unknown 继续保留整程序有符号收益，只是不进入单模块统计；不能为了增加样本强行按修改器声明归因。模块关联也不是因果贡献。

## 2. 可复现的合成反例

以下在仓库根目录用 Python 执行，仅调用现有测试夹具与纯宿主逻辑，没有模型程序执行、真实题目或网络。

```python
from test_v3_experience_policy import card
from code_rsi.v3.execution import root_files
from code_rsi.v3.edit_scope import observe_edit_scope
from code_rsi.v3.experience_policy import choose_next, memory_for_action

priors = {"evidence_selection": .9, "query_rewrite": .2}
p = card("p", .5, step=0, source="host",
         hypothesis="Synthetic mechanism A",
         diagnostics={"module_priors": priors})
scope = observe_edit_scope(
    root_files(), root_files({"prompts": {"read": "Synthetic guidance"}}),
    "evidence_selection")
c = card("c", .4, step=1, parents=["p"], source="host",
         intended_target_module="evidence_selection", actual_edit_scope=scope,
         hypothesis="Synthetic mechanism B",
         diagnostics={"module_priors": priors})
print("scope:", scope["attribution"])
for step in range(8):
    d = choose_next([p, c], step=step, panel_hash="panel-A", evaluator_epoch="eval-A")
    s = d["diagnostics"]
    m = memory_for_action([p, c], d)
    print(step, d["target_module"], s["accepted_cards"],
          len(s["unattributed_experience_ids"]),
          sum(x["gain_samples"] for x in s["module_statistics"].values()),
          len(m), any("hypothesis" in x for x in m))
```

实际输出：scope 为 mixed；step 0–7 每行均为 `evidence_selection, 2, 1, 0, 1, False`。同一固定历史只改变 step，八次选择都使用 `observed_failure_cold_start_prior`；第 5 步的父节点探索也未改变模块。

这是反驳“步数增长必然摆脱先验”的合成反例，**不是八轮真实演化**，不说明真实反馈中有多少次锁定，更不证明轮换一定更好。

## 3. 文献能支持什么

[GEPA §3](https://arxiv.org/pdf/2507.19457) 将候选父程序选择与模块选择分开；模块采用轮换，反思模型读取模块输入、输出、轨迹与反馈。由此可借鉴轮换基线和组件反馈，但不能声称论文验证了我们的经验选模块公式。

[ACE §3.1、§4.4](https://arxiv.org/html/2510.04618v3) 提供增量经验条目思路，也讨论可靠反馈不足时的经验污染。这支持以后单独检验经验复用；不能将模型提出的机制解释当已验证知识，或直接承诺迁移到本项目有效。

## 4. 对照一：失败轨迹究竟增加了什么

前置：先完成冻结 MuSiQue 校准，确认有可修复失败；不靠事后挑题建立非地板结果。随后另立运行源码版本、输出目录和具体预算。本节没有授权或启动新模型请求。

第一阶段采用固定父程序、固定修改模块、固定评估面板的一代提案实验，先测反馈作用。所有臂修改器、源码、允许编辑范围、提案数量、候选执行次数与交付规则相同；每个候选从相同父程序独立产生，不在臂内继续演化。历史经验全部为空，禁止 decision 携带先验或选择理由。根程序评估可以共享，其购买成本单列。

为避免把“逐题分数”误当“执行轨迹”的作用，定义三个清晰的信息层级：

| 条件 | 允许反馈 | 解释 |
|---|---|---|
| F0 聚合分 | 当前父程序总分、计分题数、合法性与资源合计 | 仅总分基线；不展示案例、预测、诊断先验或历史 |
| F1 案例分 | F0，加冻结案例的题面、预测与逐题分数 | 分离具体案例监督的增量；明确承认预测＋分数可能透露可接受答案 |
| F2 执行轨迹 | F1，加相同案例的检索、入窗、阅读、终答执行信息 | F2−F1 才用于估计轨迹信息增量；F2−F0 是整套反馈信息的差异 |

若首轮只能做两个条件，应预先选择 F1/F2 来检验轨迹本身；不能运行 F0/F2 后将全部差异归因于失败溯源。

### 请求契约

使用字段白名单构造最终请求，不从现有深层对象删几个键后继续发送。

- 共同字段：相同 `source_files`、固定 `operator/target_module`、相同通用编辑约束、上述允许的聚合数值。身份哈希保留在宿主，不传含诊断的整个 decision。
- F0 的 experience 恒为空，feedback 没有任意文本诊断；不通过 reason、memory、最近历史摘要、module_statistics、先验或错误示例补回案例信息。
- F1/F2 案例使用同一事先固定的题号顺序与重复选择规则，不按成败、诊断信息量或模型分差选案例。预算容纳不下完整面板时，先冻结同一子集；不能由各臂自行筛案例。
- F2 区分宿主观测、模型自述与未知。原文引用证明来源，不自动证明语义支持。新增过程信息不改变真实答案主分。
- 两臂共同字段必须在最终出站正文中一致。分别预留容量；若截断器删除了 F1/F2 的不同共同案例，该次对照不合格，不能仅核对裁剪前对象。
- 拒收提案也计入提案预算。语法、来源和硬编码等共同合法性处理保持一致；修复重试若启用，次数与可见拒收信息在协议中固定，不静默补足某臂的成功候选。

### 先做的无 API 验收

1. 在每个禁止字段植入不同哨兵文本，检查最终请求与压缩后的实际请求：F0 不含案例/轨迹哨兵，F1 不含轨迹哨兵，F2 保留已声明范围。
2. 检查共同字段及 F1/F2 案例身份、顺序和重复一致；大输入下不能产生不对称删例。
3. 使用合成 WSL 演示覆盖生成、拒收、有效／无效候选、选择锁定、报告与恢复不重复购买。
4. 未知结果不填零，非法候选不能赚取答案收益；保留全部尝试与费用。不能只对成功编译候选排名后宣称等预算。

之后具体冻结候选数、重复、资源硬帽、分析单位与最低关注效果。等候选数不保证等 token 或等费用，应同时报告实际成本；若研究每元收益，需另用相同总预算协议。开发分数只作诊断，D_select 选择并锁定，D_report 只报告；本轮开发校准题不能冒充最终独立证据。

## 5. 对照二：轮换与经验选模块

反馈形式验证后再另做这项，不同轮同时改记忆和目标。固定模型、反馈、父程序选择规则、模块集合、初始历史与总预算，比较确定轮换和当前经验策略。各臂后续父节点可能因结果不同而分岔；应称“相同父选择规则”，不宣称始终是同一父程序。

报告每步模块、选择理由、实际 AST 范围、可归因／不可归因尝试、先验驱动次数、重复提案、答案增减与真实成本。混合改动仍计整程序成绩，不能删掉以改善某策略的模块均值。反馈阶段先用固定父程序消除模块与选父耦合，完整搜索阶段再按冻结规则比较。

可能的后续修复是按实际已尝试次数安排未测模块，而不是仅凭 gain_samples 判断未尝试；但如何处理 mixed/unknown 必须另立规则，不能把意图当收益归因。本次仅定位风险，没有上线新的探索策略。

## 6. 机制记忆是第三个因素

在前述对照之外，再比较无跨轮记忆与固定容量经验条目。条目包含原修改假设、实际编辑范围、宿主测得的有符号结果、适用条件和来源；假设明确标为模型提出，不能改写成因果事实。mixed/unknown 可作为整程序经验，但不得变成单模块成功样本。

优先修复“已有假设没有传过去”的真实字段缺口，不先新建复杂学习器。条目字数、选择规则、过期规则与额外 token 成本预先固定，失败经验同样保留。模型根据例子反思通用 RAG 行为；宿主核对数据、来源、预算与测量，不手写答案逻辑。

## 7. 当前边界

MuSiQue 校准没有修改器，本次三个缺口不要求撤销或重冻现有计划。下一版演化才接入新的请求投影和策略接口。保持当前冻结源码可运行，比为了准备下一轮立刻混入多个变化更便于解释结果。

执行顺序仍是：**固定流程与支持材料校准 → 依据真实损失修一项 RAG 能力 → 反馈贡献 → 经验复用 → 模块选择**。没有真实结果前，不因本规格完整或测试通过而宣称任何研究假设成立。
