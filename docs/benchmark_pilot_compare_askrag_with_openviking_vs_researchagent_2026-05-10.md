# Benchmark Pilot 对比：askRAg(OpenViking on) vs ResearchAgent

## 对比范围

本文件记录：

1. `askRAg` 在 **OpenViking 可用** 状态下的 5 题重跑结果
2. 它与 `ResearchAgent` 当前最后一轮可用 5 题结果的对比
3. 与 `askRAg` 的 OpenViking off 结果相比，是否真的有改善

对齐题目：

- `Q1`
- `Q3`
- `Q7`
- `Q12`
- `Q13`

其中 `Q12/Q13` 作为同一组 follow-up 连续提问。

这同样只是一轮**小样本 pilot**：4 篇论文、5 个问题。它更适合回答“OpenViking on/off 在这组题上有没有带来可见变化”，而不适合被外推成稳定的统计结论或更广泛的通用排名。

## 一、运行前状态

### askRAg

这轮 askRAg 的前提：

- chat: `deepseek-v4-flash`
- embeddings: GLM `embedding-3`
- web search: 关闭
- OpenViking: `ready / healthy`

验证点：

- `/ops/state` 显示：
  - `openviking.status = ready`
  - `openviking.healthy = true`

原始结果文件：

- `D:\py\askRAg\docs\benchmark_pilot_askrag_run_deepseekv4flash_openviking_on.json`

### ResearchAgent

`ResearchAgent` 仍使用前一轮最后可用结果作为对照基线：

- `Q1`: 修复 active-paper 锚定后的 rerun
- `Q3/Q7/Q12/Q13`: 来自 `docs/benchmark_pilot_run2.md`

## 二、askRAg（OpenViking on）逐题结果

### 耗时

- `Q1`: `123.93s`
- `Q3`: `108.16s`
- `Q7`: `112.88s`
- `Q12`: `101.50s`
- `Q13`: `27.19s`

平均耗时：

- `94.73s`

这比它的 OpenViking off 版本明显更慢。

### Q1

- 结果：`partial`
- 原因：
  - 仍然只答出了“LongSeeker 解决 long-horizon search 中的 context explosion / 效率低下”
  - 没有把核心机制清楚答成 `Context-ReAct`

### Q3

- 结果：`wrong`
- 原因：
  - 这次甚至比 OpenViking off 更弱
  - 几乎只剩一句 “BAMI 是 training-free 的 GUI grounding 偏差缓解方法”
  - 没有回答偏差类型，也没有回答为什么 training-free

### Q7

- 结果：`wrong`
- 原因：
  - 仍然没有完成 LongSeeker 和 `Executable World Models` 的比较
  - 还是说上下文里没有 LongSeeker

### Q12

- 结果：`wrong`
- 原因：
  - 和 OpenViking off 一样
  - 依然只抓住 `Executable World Models`
  - 继续声称 LongSeeker 不在上下文中

### Q13

- 结果：`wrong`
- 原因：
  - 依然拒答
  - 选中的 source 依旧漂到了无关的 `BAMI`

## 三、askRAg OpenViking on 小结

### 正确性

- `correct`: `0 / 5`
- `partial`: `1 / 5`
- `wrong`: `4 / 5`

### 耗时

- 平均：`94.73s`

## 四、与 askRAg OpenViking off 的比较

OpenViking off 结果是：

- `correct`: `0 / 5`
- `partial`: `2 / 5`
- `wrong`: `3 / 5`
- 平均：`36.00s`

### 对比结论

这次 OpenViking on 并没有让 askRAg 在这组 benchmark 上变好，反而呈现出：

1. **延迟显著变差**
   - 从 `36.00s` 平均上升到 `94.73s`

2. **回答质量没有改善**
   - `Q7/Q12/Q13` 依旧失败
   - `Q3` 甚至比 off 时还更弱

3. **主要失败模式没变**
   - 仍然是单文档摘要强于跨论文比较
   - 仍然缺少稳定的 follow-up paper anchoring
   - `Q13` 仍然 source drift 到 `BAMI`

所以这轮 OpenViking on 的安全表述只能是：

- OpenViking 在 askRAg 当前实现里已经真实参与了 memory context 层
- 但在这组 5 题论文 benchmark 上，它**没有证明自己提升了最终答案质量**
- 至少在这个小样本上，它带来的最明显变化是 **更高的延迟**

## 五、与 ResearchAgent 的直接对比

### ResearchAgent 基线

- `correct`: `4 / 5`
- `partial`: `1 / 5`
- `wrong`: `0 / 5`
- 平均：`53.63s`

### askRAg（OpenViking on）

- `correct`: `0 / 5`
- `partial`: `1 / 5`
- `wrong`: `4 / 5`
- 平均：`94.73s`

### 结果解读

与 `ResearchAgent` 相比，这轮 askRAg(OpenViking on) 没有缩小差距，反而在两个维度都更弱：

1. **回答质量更差**
   - `ResearchAgent` 已经能稳定处理：
     - `Q1` 单论文解释
     - `Q3` 单论文方法细节
     - `Q7` 跨论文比较
     - `Q13` follow-up grounding
   - askRAg(OpenViking on) 只在 `Q1` 上勉强保住一个 partial

2. **速度也更慢**
   - askRAg 原本 off 版的唯一明显优势是快
   - OpenViking on 之后，这个优势也消失了

## 六、当前结论

这轮结果说明：

- 在 askRAg 现有实现里，OpenViking 更像“长期记忆增强层”
- 但它并没有自动把 askRAg 变成一个更强的论文比较 / follow-up 系统

更准确地说，当前 askRAg 的主要瓶颈仍然是：

- paper-aware session grounding 不足
- cross-paper comparison 能力弱
- follow-up 持续锚定弱

而不是：

- “缺一个长期记忆后端”

## 七、这轮 benchmark 的综合判断

到目前为止，这个 pilot benchmark 可以得出 3 个相对稳定的结论：

1. `ResearchAgent` 当前在这组论文题上明显强于 askRAg
2. 把 askRAg 的聊天模型换成与 `ResearchAgent` 相同的 `deepseek-v4-flash`，并不能消除差距
3. 把 OpenViking 打开之后，askRAg 在这组 benchmark 上也没有显示出答案质量收益，反而明显增加了延迟

所以这轮最合理的总结是：

- `ResearchAgent` 的优势主要来自 paper-native ingest、session-grounded retrieval、memory-first + reread 结构
- askRAg 当前更像本地文档问答系统，长项不在跨论文比较和 follow-up paper reasoning
