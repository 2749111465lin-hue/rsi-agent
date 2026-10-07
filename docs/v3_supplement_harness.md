# v3 补充：R-Zero 与 Harness 相关一手证据

核验日期：2026-10-07。以下均已打开原文和列出的官方仓库；日期为论文初稿提交日期。每项精简列出机制、训练要求和迁移边界。没有复现实验、调用付费 API 或更改当前评价协议。迁移建议均为本项目推断。

## 1. R-Zero: Self-Evolving Reasoning LLM from Zero Data

2025-08-07。[原文](https://arxiv.org/html/2508.05004v1)；[官方代码](https://github.com/Chengsong-Huang/R-Zero)。

Challenger 与 Solver 交替用 GRPO 更新权重；主要生成数学题，以 Solver 多数投票作伪标签，不依赖代码执行验证环境。论文报告推理基准改进，但伪标签可靠性随迭代下降；该分析又将 GPT-4o 当外部标注器。RAG 可借“针对能力边缘生成练习”的课程机制；须从 fit 来源构造可核实题目，不能把自投票当真实答案，也不适合直接替代当前不训练底座的程序搜索。

## 2. Continual Harness: Online Adaptation for Self-Improving Foundation Agents

2026-05-11。[原文](https://arxiv.org/html/2605.09998v1)；[官方代码](https://github.com/sethkarten/continual-harness)。

在 Pokémon 模拟器单局内，从帧、局部文本地图、按钮接口持续改提示、子代理、技能和记忆；这条推理时回路不更新权重。另一个共学习实验需要教师、SFT/GRPO 预热及后续参数更新。Pro 获益，但 Flash-Lite 的所有变体弱于简约基线。RAG 可借“最近失败窗口驱动局部修复”；游戏无重置优势不能直接外推为问答增益，也不宜照搬每轮四遍改写。

## 3. EnvHarness: Awakening Static Worlds for Agent Learning

2026-08-20。[原文](https://arxiv.org/html/2608.19880v1)；[官方代码](https://github.com/google-research/envharness)。

通过 reset/step 等接口，用 Stage、Contract、Chain 改初态、交互和组合任务，保留底层验证器。技能学习支路不要求底座更新，RL 支路训练权重。五个基准的技能实验最高增益 9 点；RL 的 ALFWorld OOD 仍有退步。RAG 可在 fit 内改变干扰证据、可见信息和检索限制，再用新执行验证难度；不能改封存题答案或把环境造难成功当答案正确率提升。

## 4. Hyperagents

2026-03-19。[原文](https://arxiv.org/html/2603.19461v1)；[官方代码](https://github.com/facebookresearch/HyperAgents)。

DGM-H 以冻结底座调用工具，共同修改任务代理与元代理程序，保留历代档案；需要可执行评估环境。机器人奖励设计子任务另外训练 PPO 控制策略。论文有四域改进与元层迁移，但主实验仍固定任务分布、父代抽样及评价协议。RAG 可借“任务程序和修改器分开评估、保留有用中间分支”；不能宣称全部控制机制已自改，也不宜让修改器重写答案评价器。

## 5. CORAL: Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery

2026-04-02。[原文](https://arxiv.org/html/2604.01658v1)；[官方代码](https://github.com/Human-Agent-Society/CORAL)。

使用长驻编码代理、共享经验、异步执行和心跳干预，不以更新底座权重为必要步骤；要求独立工作区及明确评价器。四代理内核任务报告 1363→1103 cycles，但多代理 API 成本约单代理 3–4 倍，不能把少评估等同总成本低。RAG 可借共享失败/修复档案和异步候选探索；应继续按真实答案质量、总调用成本和统一预算比较。

**CORAL 同名消歧：**另有 Meta 的 [CORAL: An LLM-Native Harness for Production Recommender Systems](https://arxiv.org/html/2609.02730v1)，2026-09-02，展开为 Constraint-Optimized Recommender via an Agentic Loop。它不更新代理权重，依赖线上 A/B 与数值预算优化，非上述多代理系统。可借显式预算约束；推荐参与度结论不能代替 RAG 答案质量。此次打开原文未核实该篇独立官方代码。Coral Protocol、GPU serving Coral 也不是上述工作。

## 对当前重构的优先级

立即可借：Hyperagents 的任务/修改器分层和中间档案，Continual Harness 的近期失败驱动局部变更，CORAL 的可追溯共享经验。EnvHarness 的 fit 内环境扰动可作为后续独立实验。R-Zero 与需要底座训练的分支暂不属于当前程序自改进实现范围。所有新增机制先保留固定答案评价与报告集隔离，再做预算对齐消融。
