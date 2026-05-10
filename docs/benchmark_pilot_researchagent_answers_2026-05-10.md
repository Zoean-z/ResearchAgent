# ResearchAgent Benchmark Pilot 实际答案附录

本文件整理 `ResearchAgent` 在当前最后一轮可用 5 题 pilot 中的实际回答内容，并补充简短点评。

对应题目：

- `Q1`
- `Q3`
- `Q7`
- `Q12`
- `Q13`

## Q1

**问题**

LongSeeker 主要解决什么问题？它提出的核心机制是什么？

**实际答案**

LongSeeker论文主要解决长周期搜索智能体因推理轨迹无界积累导致的上下文爆炸和效率低下问题。其核心机制是Context-ReAct范式，通过五种元操作（跳过、压缩、回滚、片段和删除）让智能体自主决定如何弹性重塑工作上下文。

**点评**

- 这题当前已经是 `correct`
- 问题定义和机制都答到了
- 还补出了五种元操作，细节层次足够

## Q3

**问题**

BAMI 想缓解的是哪几类 GUI grounding 偏差？它的做法为什么是 training-free？

**实际答案**

BAMI 主要缓解了三类 GUI grounding 偏差：知识差距（Knowledge Gap）、精度偏差（Precision Bias）和歧义偏差（Ambiguity Bias）。它是训练无关的（training-free），因为该方法无需重新训练模型，而是通过粗到细聚焦、掩码预测分布（MPD）归因和候选框选择等推理时技术直接减轻偏差，从而提升模型的定位性能。

**点评**

- 这题当前是 `correct`
- 三类 bias 说全了
- “为什么 training-free” 也解释到了推理时技术层，而不是只停在“不训练”

## Q7

**问题**

LongSeeker 和 Executable World Models for ARC-AGI-3 都在讨论 agent。它们分别把“能力提升”押在什么地方？

**实际答案**

LongSeeker's main capability improvement is its Context-ReAct paradigm, which enables elastic context orchestration via meta-operations (Skip, Compress, Rollback, Snippet, Delete) to efficiently manage long-horizon search. Executable World Models for ARC-AGI-3 places its main capability improvement in leveraging executable world models to enhance abstract reasoning performance on the ARC-AGI-3 benchmark within coding agents.

**点评**

- 这题当前是 `correct`
- LongSeeker 抓住了 `context orchestration`
- Executable World Models 抓住了 `executable world model`
- 能完成跨论文对比，而不是退化成单篇摘要

## Q12

**问题**

请总结 LongSeeker 和 Executable World Models for ARC-AGI-3 的核心区别。

**实际答案**

LongSeeker通过搜索增强长上下文推理，而Executable World Models则侧重于在ARC-AGI-3中构建可执行、可解释的世界模型进行程序化推理，核心区别在于推理机制依赖外部搜索还是内部模型执行。

**点评**

- 这题当前是 `partial`
- 它抓住了一个真实差异
- 但总结偏粗，容易把两篇论文的区别压缩成过于简单的“外部搜索 vs 内部模型执行”
- 更准确的说法应该更强调：
  - LongSeeker 关注长时搜索中的上下文编排
  - Executable World Models 关注可执行世界模型支撑的程序化推理

## Q13

**问题**

如果我更关心长时间搜索过程里的上下文管理，上一题里哪篇更相关？为什么？

**实际答案**

LongSeeker 更相关，因为它专门通过搜索增强长上下文推理，直接优化了长时搜索中的上下文管理能力。

**点评**

- 这题当前是 `correct`
- 能承接上一题的上下文
- 没有漂移到无关论文
- follow-up grounding 是成立的

## 总结

如果只看这 5 题，`ResearchAgent` 当前的回答质量可以概括成：

- 单论文事实题：已经基本可用
- 跨论文比较题：能答，而且方向通常对
- follow-up memory 题：已经能接住

当前最主要的弱点不是“完全不会答”，而是：

- 某些比较题会把真实差异总结得过于粗略
- 少数题的 final synthesis 还不够精细

所以更准确的判断不是“ResearchAgent 现在已经完美”，而是：

**它已经进入了可以做可信 benchmark 的阶段，但还没有到研究员级精细比较的稳定水平。**
