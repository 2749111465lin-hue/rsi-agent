# V3 补充核验：技能优化、上下文演化与记忆架构

核对日期：2026-10-07。阅读原始论文正文及作者仓库；仅研究与写本报告，没有模型 API 调用、数据集下载或主链路修改。日期均指 arXiv 首次提交；评价描述属于作者实验，未在本项目复现。不将论文表格或排行榜数值外推成当前 RAG 程序的预期收益。

## SkillOpt

全名：**SkillOpt: Executive Strategy for Self-Evolving Agent Skills**。2026-05-22；[论文 v1](https://arxiv.org/html/2605.23904v1)、[作者官方代码 microsoft/SkillOpt](https://github.com/microsoft/SkillOpt)。

优化单份技能 Markdown 的增删替换，执行模型与环境固定，不训权重。实测 SearchQA、SpreadsheetBench、OfficeQA、DocVQA、LiveMathematicianBench、ALFWorld。可迁移有界补丁、拒绝编辑记录和验证门；依赖可靠评分，开发期多次评估有成本，不能等同代码自改写。依据：§3–4、附录 B。

## SkillRL

全名：**SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning**。2026-02-09；[论文 v1](https://arxiv.org/html/2602.08234v1)、[官方代码 aiming-lab/SkillRL](https://github.com/aiming-lab/SkillRL)。

优化层次 SkillBank 及策略；先 SFT 再 GRPO，确实更新 Qwen2.5-7B 权重。评估 ALFWorld、WebShop 和七项搜索 QA；搜索训练用 NQ/HotpotQA。可借成功/失败抽象、通用/任务技能检索；论文收益不能归于冻结模型的记忆或代码演化。依据：§3–4、表1–2。

## ACE

全名：**Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models**。2025-10-06；另核对 v3（2026-03-29）；[论文 v3](https://arxiv.org/html/2510.04618v3)、[官方代码 ace-agent/ace](https://github.com/ace-agent/ace)。

优化带 ID 和有益/有害计数的上下文条目，不训权重；经生成、反思、整理做增量合并。评估 AppWorld 的 TGC/SGC 与 FiNER/Formula 准确率。可迁移局部更新、去重及按需取用；依赖反思质量，在线测试后更新不能充当冻结确认。依据：§3–5。

## MemEvolve

全名：**MemEvolve: Meta-Evolution of Agent Memory Systems**。2025-12-21；[论文 v1](https://arxiv.org/html/2512.18746v1)、[官方代码 bingreeky/MemEvolve](https://github.com/bingreeky/MemEvolve)。

演化经验内容及 encode/store/retrieve/manage 实现，不以权重训练为步骤；代码流程生成 Python provider。在 TaskCraft 演化，测 GAIA、WebWalkerQA、xBench-DS 等成功率、token 和延迟。可借双循环；三轮架构搜索与相关任务迁移不证明普适收益。依据：§4–5、官方 create/validate 流程。

## 名称与证据边界

这些条目均定位到具体论文，不按简称替换为同名工程。SkillOpt-Sleep 是官方后续预览功能，不是原论文实验；SkillGrad 是另一篇论文。SkillRL 不等于 MemRL，也不是只编辑 SKILL.md。ACE 采用论文明确链接的 ace-agent/ace，第三方同名复现不作为作者代码。MemEvolve 的 EvolveLab 是配套实现底座，区别于仅增加记忆内容的机制。[SkillOpt 官方说明](https://github.com/microsoft/SkillOpt)、[ACE 复现声明](https://arxiv.org/html/2510.04618v3)、[MemEvolve 官方流程](https://github.com/bingreeky/MemEvolve)

## 可直接实现的三个机制

以下为本项目的设计建议，尚未在本报告中实现或运行。

1. **有界补丁加拒绝记录。** 候选必须列出基础版本哈希、变更目标、预期改善的失败类别、涉及的开发轨迹 ID；限定每轮修改模块数/文本量，确定性应用补丁。记录被拒补丁及实测差值，供后续改进器避免重复。借鉴 SkillOpt，但验收需配对比较与费用约束；一次小幅随机涨分不应自动当作可靠改善。改进器只读开发反馈，选择集仅供控制器判定；反复使用的选择集属于开发环节，不能再称最终未见集。

2. **有来源的结构化技能条目。** 使用 `skill_id / scope / trigger / procedure / failure_modes / source_run_ids / version`，分通用与任务技能；增量更新和去重，固定可注入 token 限额。成功轨迹提供保留项，失败轨迹提供带证据的修复项；有益/有害计数只作观察信号，不能冒充因果贡献。借鉴 SkillRL 的技能层次与 ACE 的条目合并，但不训练权重。仅从开发轨迹提炼程序性知识；禁止写入确认答案、题号到答案映射或金证据定位表。确认时技能库冻结、只读。

3. **架构与记忆内容分开验证。** 把记忆模块限制在 encode/store/retrieve/manage 接口内，候选实现保持调用契约。先用同一冻结记忆快照比较新旧代码，再用同一代码比较新旧记忆；必要时采用“旧代码/新代码 × 旧记忆/新记忆”四格对照，避免把知识增加误报为架构进步。借鉴 MemEvolve；每个候选使用独立副本和相同题目顺序、预算、环境，防止先运行者给后运行者传递记忆。架构诊断只用开发日志，独立确认阶段不得写回长期记忆。

若另研究在线持续学习，须单列“先预测、后更新”的顺序协议，固定题序并禁止未来信息；不得和冻结程序的独立确认成绩混算。所有机制还应保留无记忆/无技能基线，并分开报告完成率、答案正确、证据覆盖及总费用。