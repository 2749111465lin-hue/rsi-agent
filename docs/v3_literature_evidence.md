# v3 文献证据与经验策略设计记录

核验日期：2026-10-07。只核对论文原文和作者官方材料；未执行论文实现、付费 API 或封存数据实验。下述迁移是本项目的设计推断，不表示论文已验证本项目效果。

## 一手依据

| 工作 | 论文日期/版本 | 对本项目的有限启发 | 不能声称 |
|---|---|---|---|
| [ChainRAG](https://aclanthology.org/2025.acl-long.1089.pdf)；[代码](https://github.com/nju-websoft/ChainRAG) | ACL 2025，2025-07 | 根据前一跳结果补齐下一跳实体并重写检索 | 跨领域必然有效；实体补全错误仍会传播 |
| [PRISM](https://arxiv.org/html/2510.14278v1)；[作者仓库](https://github.com/mahadi-nahid/PRISM) | 2025-10-16 v1 | 以所需事实组织证据选择和补充 | Adder 会自动召回候选池外资料；仓库当前仅说明/图片 |
| [Frontis-MA1 / OpenMLE](https://arxiv.org/html/2607.28568v1)；[代码](https://github.com/FrontisAI/OpenRSI) | 2026-07-30 v1 | 确定性经验卡、全局经验板、按动作取用少量历史 | 原算法消除了概率；其父代仍为质量/进步/新颖度 softmax 抽样 |
| [GEPA](https://arxiv.org/pdf/2507.19457)；[代码](https://github.com/gepa-ai/gepa) | 2025-07-25；2026-02-14 修订 | 用失败轨迹和文字反馈修改具体模块；含多跳 RAG 实验 | 提示优化已解决完整程序结构搜索；结果普遍超过所有 RL |
| [AFlow](https://arxiv.org/html/2410.10762v4)；[代码](https://github.com/FoundationAgents/AFlow) | 首稿 2024-10-14；ICLR 2025 | 以可执行代码表示工作流，并保留修改及实际效果 | Review/Revise 必须每轮调用；其选择机制仍有随机性 |
| [Memento-Skills](https://arxiv.org/pdf/2603.18743)；[源码说明](https://raw.githubusercontent.com/Memento-Teams/Memento-Skills/main/README.md) | 2026-03-19 | 让经验成为可执行行为，并按失败归因更新 | 记忆数量增长等于跨任务能力增长；底座冻结不代表路由器无需训练 |
| [Dream-RSI](https://arxiv.org/html/2609.14858v1)；[官方仓库](https://github.com/zhengkid/Dream-RSI) | 2026-09-14 v1 | 搜索历史充分后，用回放比较预算调度 | 回放能生成未探索动作结果或保证线上提升；代码仍待发布 |
| [EvoRAG（KG 版）](https://arxiv.org/html/2604.15676v1)；[代码](https://github.com/iDC-NEU/EvoRAG) | 2026-04-17 v1 | 将回答反馈关联到证据路径的实际贡献 | 低贡献等于事实错误；使用标准答案的评价可搬到封存测试反馈环 |

名称消歧：OpenMLE 在此指 FrontisAI/OpenRSI 的 MLE 系统，不指同名 CLI。EvoRAG 在此指 KG 反馈传播论文，不指 EMNLP 2025 的旅行规划 EvoRAG。

## 本项目采用的决策规则

实现：code_rsi/v3/experience_policy.py，策略版本 rag-rsi-v3-experience-1。只用标准库、无随机数、无文件读取、无模型调用。

输入必须由宿主构建。准入条件为 role/split 属于 fit、明确匹配 panel_hash 和 evaluator_epoch，并具有完整有效答案成绩或明确失败。缺少评价版本的旧卡不自动继承身份。同一 node_id 的冲突记录全部拒绝，重复相同卡不重复计数。字段检查不替代宿主对来源的保证。

- 没有合法父代：Draft。
- 有未被成功后代修复的明确无效失败：Debug 最新失败。有效答案附带的语义诊断不会触发 Debug。
- 常规步骤：Improve 当前答案成绩最高的有效父代。
- 每五个非零步骤：可确定性探索扩展次数较少或行为不同的有效父代。负改进节点仍留在档案中，当前最佳节点单独报告。

模块收益仅来自同面板/评价版本的有效 child-parent 答案成绩差；多父代与最高分父代比较，负数完整保留。即使卡提供正向 signed_delta，若与合法父子成绩不同，仍使用重新计算的真实差值。父代无有效测量时不伪造 gain。检索相似度、覆盖率等代理不作为答案质量。

按父代失败类别匹配动作历史；failure_classes 支持计数字典及 list/tuple 字符串；失败类别可以是模型诊断，其意义是检索相关经验，不是测得的奖励。某模块的效用为平均 signed gain 减不确定性惩罚、失败比例惩罚及成本惩罚。

当前启发式为：

- 不确定性：0.05 / sqrt(n + 1) + sample_std / sqrt(n)；n 小于 2 时省略标准差项。
- 失败：0.05 * 明确无效失败数 / 动作尝试数。
- 成本：0.03 * log1p(平均有效成本 / 成本尺度)。
- 在匹配动作中统一选一种已有单位，优先 cny、usd、seconds、calls、model_invocations、tokens，不混加不同单位。
- 成本尺度为该单位已知正成本的中位数。缺失成本保守补成已知最大成本与尺度中较大值的两倍；全部缺失则明确报告未知，不伪造零成本。
- 无任何可测模块时按 step 轮转；每四步探索未测模块，或已知模块平均收益均不为正时优先探索未测模块。

这些系数与调度周期是可审查的初始工程规则，不是论文估计值，不构成统计置信界或上线收益保证。应在固定 fit 面板和预算下做消融后调整。

memory_for_action 再次检查身份，取最近的所选父代、同模块相关失败及修复记录；不随机混入完整日志或 report/select 经验。无效测量的 score 仍为 None。它保留诊断来源字段，避免把模型评估描述成宿主答案测量。

## API 与验证

- choose_next(cards, *, step, panel_hash, evaluator_epoch, allowed_modules=None)：返回 parent_node_id、operator、target_module、reason、experience_ids 和 diagnostics，同时携带身份供后续记忆过滤。
- memory_for_action(cards, decision, limit=4)：返回紧凑经验记录的新副本。
- 默认模块：query_rewrite、retrieval、evidence_selection、answer_generation；主流程可传 allowed_modules。
- 测试文件：test_v3_experience_policy.py。
- 离线验证：项目虚拟环境执行 python -m unittest test_v3_experience_policy -v，29 项通过。
- 覆盖：冷启动、负收益、身份隔离、重复/冲突卡、确定性、同失败类型、已知/未知成本、Debug 与语义诊断分离、修复成功、低扩展父代探索、动作专用记忆与输入不变性。
