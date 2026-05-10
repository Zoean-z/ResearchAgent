# Benchmark Pilot Run 1

## Scope

本轮不是正式 benchmark，只是第一轮 live pilot。

目标：

1. 确认 `askRAg` 的 benchmark-only 语料是否可用
2. 确认 `research-agent` 在模型可用后，是否能围绕已导入论文稳定回答
3. 先用 5 个样本题看出主要问题出在哪一层

## Corpus Preparation

`askRAg` 侧已新增 benchmark-only 语料：

- `data/docs/benchmark_pilot/2605.05191-LongSeeker.md`
- `data/docs/benchmark_pilot/2605.05138-Executable-World-Models-ARC-AGI-3.md`
- `data/docs/benchmark_pilot/2605.06664-BAMI-GUI-Grounding.md`
- `data/docs/benchmark_pilot/2605.05017-SPINE-Privacy-Utility-Tradeoff.md`

索引重建后：

- documents: `13`
- chunks: `881`

## Runtime Notes

### askRAg

- OpenViking 健康检查通过
- `GET /ops/state` 显示 `openviking.status=ready`
- 运行时可正常访问 benchmark-only markdown 文档

### ResearchAgent

- 发现一个真实运行面问题：当前 shell 环境里的旧 `DEEPSEEK_API_KEY` 会覆盖 `.env`
- 需要显式用 `.env` 中的新 key 启动 benchmark 实例
- 这轮 live run 使用了 `8012` 端口的重启实例

## Sample Questions

本轮使用的 5 个样本题：

- `Q1` 在 `LongSeeker` 论文里，主要解决什么问题？提出的核心机制是什么？
- `Q3` 在 `BAMI` 论文里，想缓解哪几类 GUI grounding 偏差？为什么说它是 training-free？
- `Q7` `LongSeeker` 和 `Executable World Models for ARC-AGI-3` 都在讨论 agent。它们分别把能力提升押在什么地方？
- `Q12` 请总结 `LongSeeker` 和 `Executable World Models for ARC-AGI-3` 的核心区别。
- `Q13` 如果我更关心长时间搜索过程里的上下文管理，上一题里哪篇更相关？为什么？

## Results

### ResearchAgent

| Question | Result | Time | Notes |
| --- | --- | --- | --- |
| `Q1` | wrong | `33.7s` | 返回“未找到相关论文或信息” |
| `Q3` | wrong | `23.8s` | 返回“未找到相关论文或信息” |
| `Q7` | wrong | `26.9s` | 返回“未找到相关信息” |
| `Q12` | wrong | `28.5s` | 返回“未能找到可用信息” |
| `Q13` | wrong | `10.9s` | 延续上一题的“未找到信息” |

额外观察：

- 这轮查询全部 `status=200`，说明模型调用已恢复
- 但 `used_memory_citations` 基本为 `0`
- `source_reread_chunks` 也没有进入稳定可用状态
- 从结果上看，当前主要问题不再是 key，而是 **query-time grounding 没有稳定命中已导入论文**

### askRAg

| Question | Result | Time | Notes |
| --- | --- | --- | --- |
| `Q1` | correct | `95.6s` | 正确命中 `LongSeeker` 文档并概括问题与方法 |
| `Q3` | partial | `121.6s` | 正确命中 `BAMI`，但偏差类型解释不够完整 |
| `Q7` | wrong | `123.0s` | 只回答了 `ARC-AGI-3` 论文，没有做比较 |
| `Q12` | wrong | `107.9s` | 返回“无法可靠回答” |
| `Q13` | wrong | `38.9s` | follow-up 也没有接住上一轮上下文 |

额外观察：

- 单论文题已经可用，能稳定命中新建的 benchmark markdown
- 跨论文比较明显变弱
- follow-up / memory 类题目前没有体现出稳定优势

## Current Read

截至 Run 1，第一轮 pilot 已经能回答两个关键问题：

1. `askRAg` 的 benchmark-only markdown 语料路线是可行的
   - 至少单论文事实题已经能跑通

2. `research-agent` 当前更大的问题不是导入，而是 query 阶段没有稳定利用已导入论文
   - 环境问题已解决
   - 但回答层仍然没有围绕 session 内论文建立可靠 grounding

## What This Means

如果现在立刻扩大到完整 `16` 题或正式 `30-50` 题，结果会被 `research-agent` 当前的 query grounding 问题严重污染。

所以更合理的下一步不是直接扩 benchmark，而是先做一轮窄调试：

1. 查清 `research-agent` 为什么在已导入 session 中仍然频繁返回“未找到相关论文或信息”
2. 确认它的 query path 是否真正看到了 session papers / paper memories / source chunks
3. 在 `research-agent` query grounding 修到最小可用后，再重跑这 5 题

## Recommendation

下一步优先做：

- `research-agent` query grounding diagnosis

而不是：

- 继续扩论文数量
- 继续加更多 benchmark 问题
- 立刻写 README 的 Benchmark section

原因很简单：当前 Run 1 已经足够说明，benchmark 轨道跑起来了，但系统对比还不公平，因为 `research-agent` 处在“导入成功、查询失焦”的状态。
