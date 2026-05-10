# Benchmark Pilot Draft

## Goal

先做一轮小样本 pilot，验证下面三件事：

1. `research-agent` 和 `askRAg` 能否围绕同一批论文进入可比测试轨道
2. `PDF/arXiv -> askRAg markdown` 这条语料对齐路线是否够用
3. 三类问题是否能稳定区分：
   - 单论文事实检索
   - 跨论文比较/关联
   - 多轮 memory / follow-up

本轮先不追求完整正式 benchmark，不先写 README 里的最终结论。

## Systems Under Test

- `ResearchAgent`
  - 路径：原生 arXiv / PDF ingest
  - 特点：模型可调用 `search_arxiv`、`import_arxiv_paper`，查询期优先使用论文记忆

- `askRAg`
  - 路径：本地 `.md/.txt` 文档索引
  - 特点：固定检索/记忆流程更强，当前不是 arXiv/PDF-native

## Pilot Paper Set

第一轮固定用这 4 篇：

1. `LongSeeker: Elastic Context Orchestration for Long-Horizon Search Agents`
   - arXiv: `2605.05191`
   - 作用：搜索 agent / context orchestration

2. `Executable World Models for ARC-AGI-3 in the Era of Coding Agents`
   - arXiv: `2605.05138`
   - 作用：coding agent / executable world model

3. `BAMI: Training-Free Bias Mitigation in GUI Grounding`
   - arXiv: `2605.06664`
   - 作用：GUI agent / grounding / bias mitigation

4. `Position: Embodied AI Requires a Privacy-Utility Trade-off`
   - arXiv: `2605.05017`
   - 作用：position paper / privacy-utility framework

## Why This Set

- 都能从摘要和方法段读出核心论点，不强依赖大表格
- 主题有区分度，但仍然都能落在 “agent / reasoning / interaction / framework” 的邻近区域
- 适合先做文本型 benchmark，不先碰公式、图表、附录重题

## Corpus Alignment Rule

本轮必须使用同一批论文，但输入路径允许不同：

- `ResearchAgent`
  - 直接使用现有 arXiv ingest

- `askRAg`
  - 使用同一批论文转换后的 `.md`
  - 不要求先改造 askRAg 支持 PDF

结论解释时必须明确写出：

- `ResearchAgent` 使用原生 arXiv/PDF ingest
- `askRAg` 使用同源论文的预处理 markdown 语料

## Question Set

本轮先做 `16` 题，分 3 组。

### A. 单论文事实题

`Q1`
`LongSeeker` 主要解决什么问题？它提出的核心机制是什么？

目的：
验证能否正确提取论文目标与方法主线。

`Q2`
`Executable World Models for ARC-AGI-3` 中，作者所说的 “executable world model” 具体指什么？

目的：
验证对核心技术概念的解释能力。

`Q3`
`BAMI` 想缓解的是哪几类 GUI grounding 偏差？它的做法为什么是 training-free？

目的：
验证方法细节和关键术语理解。

`Q4`
`Position: Embodied AI Requires a Privacy-Utility Trade-off` 这篇文章的核心主张是什么？`SPINE` 是什么？

目的：
验证 position / framework 论文的概念提取能力。

`Q5`
`LongSeeker` 论文里为什么认为长程搜索 agent 会出现 context explosion？

目的：
验证对问题动机的定位能力。

`Q6`
`Executable World Models for ARC-AGI-3` 报告了哪些整体结果？不要背所有数值，概括主要结论即可。

目的：
验证是否能抓住结果层面的高层结论，而不是丢进细枝末节。

### B. 跨论文比较题

`Q7`
`LongSeeker` 和 `Executable World Models for ARC-AGI-3` 都在讨论 agent。它们分别把“能力提升”押在什么地方？

目的：
比较 context orchestration 与 world modeling 的不同侧重点。

`Q8`
`BAMI` 和 `LongSeeker` 的共同点是什么？它们的任务场景又有什么本质区别？

目的：
验证跨任务抽象与差异识别。

`Q9`
`SPINE` 这篇 position paper 和 `BAMI` 这种方法论文在输出形式上有什么不同？

目的：
看系统能否区分 framework / position paper 与 concrete method paper。

`Q10`
如果从 “agent 在复杂环境中犯错的原因” 这个角度看，`BAMI`、`LongSeeker`、`SPINE` 三篇分别关注了什么问题？

目的：
验证三篇以上的多论文归纳能力。

`Q11`
如果要把这 4 篇论文分成“更偏系统/agent runtime”和“更偏任务/应用或治理框架”两类，你会怎么分？

目的：
验证聚类式比较，而不是简单逐篇复述。

### C. 多轮 Memory / Follow-up 题

下面这组题必须在同一个 session 里顺序提问，不能每题新开会话。

`Q12`
先问：请总结 `LongSeeker` 和 `Executable World Models for ARC-AGI-3` 的核心区别。

目的：
给系统建立一个明确的会话内比较上下文。

`Q13`
紧接着追问：如果我更关心长时间搜索过程里的上下文管理，上一题里哪篇更相关？为什么？

目的：
验证系统是否复用上一轮比较结论，而不是重新从零检索。

`Q14`
再追问：那另一篇更适合什么问题场景？

目的：
验证对照对象是否还能被稳定指代。

`Q15`
换一组上下文：先问 `SPINE` 和 `BAMI` 的论文类型差异是什么？

目的：
建立第二组跨论文上下文。

`Q16`
紧接着追问：如果我接下来要做“隐私敏感场景下的 GUI agent”，刚才这两篇里哪篇更像治理框架，哪篇更像具体技术部件？

目的：
验证系统是否能把上一轮上下文和新的任务需求结合起来回答。

## Run Protocol

### ResearchAgent

每组测试使用一个新 session。

准备：

1. 在当前 session 中导入这 4 篇论文
2. 确认导入成功并生成 paper memory

执行：

1. A 组和 B 组题可以按单题独立记录
2. C 组题必须在同一 session 内连续提问
3. 记录每题：
   - 是否回答正确
   - `step_count`
   - 总响应时间
   - 是否明显使用了 memory / reread / source evidence

### askRAg

每组测试使用一个新 `conversation_id`。

准备：

1. 将同一批论文转成 `.md`
2. 放入 `data/docs/benchmark_pilot/`
3. 重建索引
4. 保持 `use_web_search=false`

执行：

1. A 组和 B 组题按单题独立记录
2. C 组题必须在同一 `conversation_id` 内连续提问
3. 记录每题：
   - 是否回答正确
   - 响应时间
   - trace 中是否使用了 memory context
   - 是否出现无关 source / 路由漂移

## Scoring

本轮先用人工 rubric，不做自动评分。

### Correct

满足下面两条即记为 `correct`：

- 主结论正确
- 没有关键性事实颠倒

### Partial

满足下面任一条记为 `partial`：

- 大方向对，但核心区别说反或混淆
- 只答到表层，没有覆盖问题要求的关键点

### Wrong

满足下面任一条记为 `wrong`：

- 论文张冠李戴
- 方法、任务、贡献主体混淆
- follow-up 没接住上一轮上下文

## Metrics

每题记录这几项：

- `system`
- `question_id`
- `paper_scope`
- `question_type`
- `correctness`
- `response_time_ms`
- `step_count`
- `used_memory`
- `used_source_reread`
- `notes`

说明：

- `ResearchAgent.step_count` 直接取运行时 step 数
- `askRAg.step_count` 本轮先不强行和 `ResearchAgent` 做 1:1 对齐
- 如果需要统一对比，第一轮先只比较：
  - 正确率
  - 平均响应时间
  - memory follow-up 成功率

## Success Criteria For Pilot

满足下面条件，就可以进入正式 benchmark 设计：

1. 这 4 篇论文都能被 askRAg 的 markdown 语料稳定回答单论文事实题
2. 至少一半跨论文题在两边都能得到可接受答案
3. 多轮 follow-up 题能明显区分两边的 memory / context handling 差异
4. 没有出现大面积因为语料转换失败而无法回答的情况

## Known Risks

- `askRAg` 的 markdown 语料可能丢失图表和细节结构
- `ResearchAgent` 与 `askRAg` 的 memory 语义并不完全同构
- 如果题目过于依赖数值表格，本轮结果会被语料格式差异污染
- `askRAg` 当前不是 paper-native pipeline，这一点必须在结论里写清楚

## Next Step After This Draft

1. 准备这 4 篇论文对应的 askRAg markdown 语料
2. 先跑 4-6 个小样本问题，验证语料是否够用
3. 如果语料质量可接受，再跑完整 16 题
4. pilot 通过后，再扩展到 10-20 篇和 30-50 题的正式 benchmark
