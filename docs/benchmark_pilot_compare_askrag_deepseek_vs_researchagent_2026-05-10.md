# Benchmark Pilot 对比

## 对比范围

本文件对比两组结果：

- `ResearchAgent` 当前最后一轮可用的 5 题 pilot 结果
- `askRAg` 在聊天模型切换到 `deepseek-v4-flash`、embedding 继续保留 GLM 后的 5 题重跑结果

对齐题目：

- `Q1`
- `Q3`
- `Q7`
- `Q12`
- `Q13`

其中 `Q12/Q13` 按 follow-up 连续问题处理。

这是一轮**小样本 pilot**，样本规模只有 4 篇论文、5 个问题，因此这里的结论更适合被理解为“系统路径差异的探索性证据”，不能直接外推成大规模正式 benchmark 或统计显著结论。

## 一、ResearchAgent 基线结果

这里采用的 `ResearchAgent` 基线是：

- `Q3/Q7/Q12/Q13`：来自 `docs/benchmark_pilot_run2.md`
- `Q1`：来自修复 `latest imported paper` 误锚定之后的单独 rerun

### 正确性

- `Q1`: `correct`
- `Q3`: `correct`
- `Q7`: `correct`
- `Q12`: `partial`
- `Q13`: `correct`

即：

- `correct`: `4 / 5`
- `partial`: `1 / 5`
- `wrong`: `0 / 5`

### 耗时

- `Q1`: `53.47s`
- `Q3`: `62.14s`
- `Q7`: `94.03s`
- `Q12`: `48.85s`
- `Q13`: `9.64s`
- 平均：`53.63s`

## 二、askRAg（DeepSeek chat + GLM embedding）结果

这轮 askRAg 的运行配置是：

- chat: `deepseek-v4-flash`
- chat base URL: `https://api.deepseek.com`
- embedding: GLM `embedding-3`
- web search: 关闭

原始结果文件：

- `D:\py\askRAg\docs\benchmark_pilot_askrag_run_deepseekv4flash.json`

### 耗时

- `Q1`: `36.70s`
- `Q3`: `35.44s`
- `Q7`: `36.50s`
- `Q12`: `36.83s`
- `Q13`: `34.53s`
- 平均：`36.00s`

### 逐题评分

#### Q1

- 结果：`partial`
- 判断理由：
  - 它答对了 LongSeeker 主要解决的是长时域搜索中的 context explosion / 低效率问题
  - 但没有把问题的第二部分答完整
  - 没有明确说出核心机制是 `Context-ReAct`

#### Q3

- 结果：`partial`
- 判断理由：
  - 它答对了 BAMI 是 training-free
  - 也提到了粗到细聚焦、候选框选择这类步骤
  - 但没有清楚列出题目要求的 GUI grounding 偏差类型

#### Q7

- 结果：`wrong`
- 判断理由：
  - 它没有完成两篇论文的比较
  - 直接说上下文里没有 LongSeeker
  - 实际只回答了 `Executable World Models` 一边

#### Q12

- 结果：`wrong`
- 判断理由：
  - 本题要求总结 LongSeeker 和 `Executable World Models` 的核心区别
  - 它依然只抓住了 `Executable World Models`
  - 继续错误声称 LongSeeker 不在上下文中

#### Q13

- 结果：`wrong`
- 判断理由：
  - 这是一个 follow-up / memory continuity 题
  - 它直接拒答
  - 选中的 source 还漂到了无关的 `BAMI`

### askRAg 小结

- `correct`: `0 / 5`
- `partial`: `2 / 5`
- `wrong`: `3 / 5`

## 三、实现差异

这轮 benchmark 很重要的一点是：**两边的差距并不只是模型差距。**

把 askRAg 的聊天模型对齐到 `deepseek-v4-flash` 之后，结果仍然明显落后，说明真正的差异更多来自系统实现。

### 1. 文档检索主干不同

`askRAg` 的文档主干很明确是 **Chroma 向量检索**：

- `app/rag.py` 里直接使用 `langchain_chroma.Chroma`
- `build_vector_store()` / `get_vector_store()` 会创建 Chroma collection
- `CompatibleEmbeddings` 负责真正调用 embedding API

见：

- [app/rag.py](D:/py/askRAg/app/rag.py:15)
- [app/rag.py](D:/py/askRAg/app/rag.py:173)
- [app/rag.py](D:/py/askRAg/app/rag.py:774)
- [app/rag.py](D:/py/askRAg/app/rag.py:789)

而 `ResearchAgent` 当前并不是“外部向量库主导”的实现。

它的主路径是：

- 论文按 chunk 持久化到 SQLite
- memory 也在本地表里维护
- 查询时先走 session/global memory
- 不够时再从 source chunks 做 reread

见：

- [backend/src/research_agent/adapters/storage/sqlite_content.py](D:/py/research-agent/backend/src/research_agent/adapters/storage/sqlite_content.py:117)
- [backend/src/research_agent/services/retrieval_service.py](D:/py/research-agent/backend/src/research_agent/services/retrieval_service.py:77)
- [backend/src/research_agent/tools/registry.py](D:/py/research-agent/backend/src/research_agent/tools/registry.py:288)

也就是说，`ResearchAgent` 现在更像：

- `SQLite chunk store + structured memory tables + selective reread`

而不是：

- `向量库召回一批 chunk，再让模型总结`

### 2. 论文对象语义不同

`ResearchAgent` 是 paper-native 的：

- 先 ingest arXiv / PDF
- 建立 `paper`、`artifact`、`chunk`、`session_document`
- 再围绕 paper 写 `paper_memory`、`open_question_memory`

所以它问答时天然知道“当前 session 里有哪些论文”。

而 `askRAg` 当前还是 document-native 的：

- 输入是 `.md/.txt`
- 检索对象首先是 document / chunk
- 并没有天然的 `paper_id -> session paper set -> paper memory bundle` 这一层

这也是为什么 askRAg 在 `Q7/Q12/Q13` 这类题上更容易退化成：

- 只抓住一篇文档
- 或者在 follow-up 时 source 漂到无关文档

### 3. memory 的角色不同

`ResearchAgent` 的 memory 是论文研究导向的：

- `paper_memory`
- `relation_memory`
- `open_question_memory`

其运行顺序也是显式设计好的：

1. session memory
2. global memory
3. source reread

而 `askRAg` 的 memory 更偏 assistant memory / task-state recall。

即使 OpenViking 参与，它更像“回答时的长期上下文增强层”，不是围绕论文对象组织的一套 memory schema。

所以在 benchmark 里：

- askRAg 的 memory 更容易帮助“记住最近说了什么”
- 但不等于它天然擅长“围绕两篇论文持续比较”

### 4. 工具/运行时结构不同

`ResearchAgent` 当前已经是显式 host runtime 驱动：

- session 里有导入论文
- 工具里有 `list_session_papers`
- 有 `get_paper_memory_bundle`
- 有 `search_source_chunks`
- 有 `read_source_passages`

虽然最近暂时下线了 `search_arxiv` 的正常 query 暴露，但它的 query path 已经是显式的 paper-grounded runtime。

askRAg 这边则还是更像：

- 固定流程检索
- 选 source
- 生成答案

所以它的优势主要是：

- 路径短
- 延迟更稳

但劣势就是：

- 跨 paper 对比弱
- follow-up paper anchoring 弱

## 四、结果解读

这轮结果非常清楚：

- **askRAg 更快**
- **ResearchAgent 答得更好**

而且更重要的是：

把 askRAg 的聊天模型换成和 `ResearchAgent` 一样的 `deepseek-v4-flash` 之后，差距依然还在。

这说明当前 5 题 slice 上，真正决定结果的主要不是聊天模型本身，而是：

- `ResearchAgent` 的 paper-native ingest + session-grounded retrieval
- `ResearchAgent` 的 memory-first + source-reread 路径
- askRAg 的 fixed-path document summarization 风格

## 五、当前结论

在当前这个 5 题 pilot 上：

- 如果问题只是单篇论文的简单摘要，askRAg 有时可以较快给出一个“部分可用”的回答
- 但一旦进入：
  - 跨论文比较
  - session 内 follow-up
  - 需要稳定保持论文锚定
  
  `ResearchAgent` 明显更强

因此，这轮 benchmark 的主结论可以写成：

- `ResearchAgent` 在当前 5 题 slice 上赢在回答质量
- `askRAg` 赢在延迟
- 聊天模型对齐之后，系统实现差异仍然是主要差异来源
