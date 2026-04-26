# OpenViking Memory-Routed Paper Agent Project Spec

## 项目目标

- 做一个能阅读 arXiv 论文或本地 PDF 的 agent。
- 将论文理解结果写入结构化研究记忆。
- 后续追问时优先使用记忆，只有记忆不足时才回读原始论文。
- 核心价值不是普通论文问答，而是“记忆改变后续决策”。

## MVP 范围

- 新论文资料输入只支持 arXiv 链接或本地 PDF。
- 在论文被读入某个 session 后，用户可以在该 session 中继续使用自然语言进行后续追问。
- 支持三类记忆：
  - `paper_memory`
  - `relation_memory`
  - `open_question_memory`
- 后续追问仅限于当前 session 中已读入的论文及其相关记忆。
- 处理追问时，必须先查和当前 session 相关的记忆，再查全局记忆，只有记忆不足时才回读原文。
- MVP 不加 `web_search`。
- 不做 `multi-agent`。
- 不要一开始就重度依赖 agent 框架。
- runtime 设计要尽量与框架解耦，方便后续迁移到 PydanticAI。

## 产品约束

- 记忆是全局共享的，但每个 session 只展示与自己相关的引用、事件和上下文。
- 第一版先用 SQLite 或 mock 存储，并为以后接 OpenViking 预留清晰的 adapter 边界。
- 第一版先手写 runtime loop，不要把核心业务逻辑绑死在 agent 框架里。
- trace 里的 `action` 和 `result` 必须来自真实执行状态。
- trace 里的 `reason` 和 `impact` 可以由模型生成，但要和系统原始记录分开存储。
- 系统中的消息类型至少要区分：
  - `ingest_arxiv`
  - `ingest_pdf`
  - `followup_query`

## 推荐目录结构

```text
research-agent/
  AGENTS.md
  README.md
  .gitignore
  backend/
    pyproject.toml
    src/research_agent/
      __init__.py
      domain/
        __init__.py
        models/
          __init__.py
        value_objects/
          __init__.py
        enums/
          __init__.py
        ports/
          __init__.py
        policies/
          __init__.py
      services/
        __init__.py
      tools/
        __init__.py
      runtime/
        __init__.py
      adapters/
        __init__.py
        storage/
          __init__.py
        sources/
          __init__.py
        llm/
          __init__.py
      api/
        __init__.py
        routes/
          __init__.py
        schemas/
          __init__.py
        deps.py
        app.py
    tests/
      unit/
        __init__.py
      integration/
        __init__.py
      contract/
        __init__.py
      fixtures/
        __init__.py
    migrations/
      .gitkeep
  frontend/
    package.json
    src/
      app/
      components/
      features/
        sessions/
        chat/
        ingest/
        timeline/
        trace/
        memory/
      lib/
        api/
        types/
      styles/
    tests/
  data/
    artifacts/
      .gitkeep
    sqlite/
      .gitkeep
  docs/
    project_spec.md
    api.md
```

## 核心领域模型

### Session

- `id`
- `title`
- `created_at`
- `updated_at`
- `status`

### SessionDocument

- `id`
- `session_id`
- `paper_id`
- `source_type`
- `artifact_id`
- `added_at`

### Message

- `id`
- `session_id`
- `type`
- `content`
- `created_at`
- `status`

### Paper

- `id`
- `canonical_key`
- `title`
- `authors`
- `abstract`
- `year`
- `arxiv_id`
- `pdf_fingerprint`

### Artifact

- `id`
- `kind`
- `uri_or_path`
- `checksum`
- `page_count`

### SourceRef

- `paper_id`
- `artifact_id`
- `page`
- `section`
- `chunk_id`
- `quote`

### PaperMemory

- `id`
- `paper_id`
- `problem`
- `method`
- `key_results`
- `limitations`
- `novelty_claim`
- `source_refs`
- `confidence`
- `updated_at`

### RelationMemory

- `id`
- `source_paper`
- `target_paper`
- `relation_type`
- `summary`
- `evidence`
- `confidence`
- `updated_at`

### OpenQuestionMemory

- `id`
- `unresolved_question`
- `related_papers`
- `why_open`
- `possible_followup`
- `confidence`
- `updated_at`

### TaskRun

- `id`
- `session_id`
- `message_id`
- `status`
- `step_count`
- `started_at`
- `finished_at`
- `finish_reason`

### TraceStep

- `id`
- `run_id`
- `action`
- `input_payload`
- `result_payload`
- `status`
- `started_at`
- `finished_at`

### TraceNarrative

- `trace_step_id`
- `reason_text`
- `impact_text`

### TimelineEvent

- `id`
- `session_id`
- `run_id`
- `event_type`
- `summary`
- `related_memory_ids`
- `related_paper_ids`
- `created_at`

## Service 边界

### SessionService

- 创建、列出、读取 session
- 管理 session 内消息和文档绑定关系

### PaperRegistryService

- 规范化论文身份
- 管理 canonical key、alias 和重复论文判定

### IngestionService

- 执行 arXiv / PDF 输入处理流程
- 组织文本提取与初始记忆写入

### MemoryService

- 管理三类 memory 的创建、合并、更新和查询

### RetrievalService

- 执行 follow-up 查询时的记忆优先检索与 sufficiency 判断

### QueryService

- 组织 follow-up 的上下文
- 调用回答组合逻辑

### TraceService

- 记录真实执行步骤和结果
- 在 run 结束后生成 narrative

### TimelineService

- 生成与 session 相关的研究时间线事件

### TaskRunService

- 管理 run 生命周期
- 控制 step limit、termination 和 finish_task

## Tool 列表和接口契约

### `resolve_arxiv_source`

- input: `{ arxiv_url: str }`
- output: `{ normalized_arxiv_id: str, title: str | null, pdf_url: str | null, metadata: dict }`

### `load_local_pdf`

- input: `{ file_path: str }`
- output: `{ artifact_id: str, checksum: str, page_count: int | null }`

### `extract_pdf_text`

- input: `{ artifact_id: str }`
- output: `{ pages: list, sections: list | null, chunks: list }`

### `register_paper`

- input: `{ metadata: dict, checksum: str | null, arxiv_id: str | null }`
- output: `{ paper_id: str, canonical_key: str, operation: 'created' | 'matched' }`

### `retrieve_session_memories`

- input: `{ session_id: str, query: str, top_k: int }`
- output: `{ memories: list, coverage_score: float }`

### `retrieve_global_memories`

- input: `{ query: str, related_paper_ids: list | null, top_k: int }`
- output: `{ memories: list, coverage_score: float }`

### `read_source_passages`

- input: `{ session_id: str, query: str, target_paper_ids: list, top_k: int }`
- output: `{ passages: list, citations: list }`

### `upsert_paper_memory`

- input: `{ paper_id: str, memory_payload: dict }`
- output: `{ memory_id: str, operation: 'created' | 'updated' }`

### `upsert_relation_memory`

- input: `{ relation_payload: dict }`
- output: `{ memory_id: str, operation: 'created' | 'updated' }`

### `upsert_open_question_memory`

- input: `{ open_question_payload: dict }`
- output: `{ memory_id: str, operation: 'created' | 'updated' }`

### `compose_answer`

- input: `{ query: str, memory_context: list, source_context: list | null }`
- output: `{ answer: str, citations: list, memory_influence: list }`

### `finish_task`

- input: `{ run_id: str, finish_reason: str, outcome: dict }`
- output: `{ status: 'finished' }`

### 工具契约统一规则

- 每个 tool 返回结构化结果。
- tool 不返回自由文本日志。
- 每个 tool 的真实 `action` / `result` 必须写入 `TraceStep`。
- `reason` / `impact` 不放入 tool 原始返回，而是单独进入 `TraceNarrative`。

## Runtime Loop 设计

### 基本运行模型

- 一个用户消息对应一个 `TaskRun`。
- `TaskRun` 状态机：
  - `pending`
  - `running`
  - `finished`
  - `failed`
  - `step_limit_reached`
- 宿主 runtime 维护 `max_steps`，MVP 固定为 `8`。

### `ingest_arxiv` / `ingest_pdf` 路径

1. 创建 `TaskRun`
2. 解析来源并注册 `Paper`
3. 获取或载入 PDF artifact
4. 提取文本和 chunks
5. 生成并 upsert `paper_memory`
6. 必要时生成并 upsert `relation_memory`
7. 必要时生成并 upsert `open_question_memory`
8. 写入 `TraceStep`、`TimelineEvent`
9. 调用 `finish_task`

### `followup_query` 路径

1. 创建 `TaskRun`
2. 查询当前 session 相关记忆
3. 查询全局记忆
4. 做 sufficiency 判断
5. 若不足才回读原始论文片段
6. 生成答案和引用
7. 必要时补充 open question 或 relation memory
8. 写入 trace 和 timeline
9. 调用 `finish_task`

### termination logic

以下情况必须终止 run：

- 已得到最终答案
- ingest 已完成并成功写入记忆
- 出现不可恢复错误
- 达到 step limit
- 无法获得足够上下文且不存在下一步恢复路径

### finish_task 机制

- `finish_task` 是显式 runtime 操作，不由模型自由决定。
- 宿主 runtime 负责更新 run 终态、计数、结果摘要和结束时间。
- step limit 必须由宿主 runtime 控制，不交给模型自由决定。

## Storage Adapter 设计

- 第一版先做 `SQLite` + `InMemoryMock`。
- 通过 repository ports 与领域层交互。

### repository ports

- `SessionRepositoryPort`
- `MessageRepositoryPort`
- `PaperRepositoryPort`
- `ArtifactRepositoryPort`
- `MemoryRepositoryPort`
- `TraceRepositoryPort`
- `TimelineRepositoryPort`
- `ChunkRepositoryPort`

### SQLite 结构要求

- 使用结构化表存储三类 memory。
- 提供 `chunks` 表保存切片。
- 使用 SQLite `FTS5` 做本地检索。
- `trace_steps` / `trace_narratives` 必须分表。
- memory 表需要支持 upsert。

### OpenViking 迁移方式

- 后续只新增 `OpenViking` storage adapter。
- `domain` / `services` / `runtime` 不直接依赖 OpenViking。
- 保持 repository port 不变，用 adapter 替换存储实现。

## API 设计

采用 `REST + SSE`。

### Sessions

- `POST /api/sessions`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `GET /api/sessions/{session_id}/messages`

### Ingest

- `POST /api/sessions/{session_id}/ingest/arxiv`
- `POST /api/sessions/{session_id}/ingest/pdf`

### Query

- `POST /api/sessions/{session_id}/queries`

### Timeline / Memory / Trace

- `GET /api/sessions/{session_id}/timeline`
- `GET /api/sessions/{session_id}/memory-snapshot`
- `GET /api/sessions/{session_id}/runs/{run_id}/trace`
- `GET /api/sessions/{session_id}/runs/{run_id}/events`

## 前端组件结构

前端采用三栏布局：

- 左栏：`SessionSidebar`
- 中栏：`ChatWorkspace`
- 右栏：
  - `ResearchTimelineTab`
  - `MemoryInfluenceTraceTab`
  - `MemorySnapshotPanel`

### 主要组件

- `SessionSidebar`
  - session 列表
  - 聊天历史管理

- `ChatWorkspace`
  - 消息区
  - 输入框
  - arXiv 输入
  - PDF 上传入口

- `ResearchTimelineTab`
  - 展示 session 相关论文读入、记忆生成、记忆更新和回答事件

- `MemoryInfluenceTraceTab`
  - 展示真实执行步骤与 narrative
  - 强调“先查记忆，再决定是否回读原文”

- `MemorySnapshotPanel`
  - 展示当前 session 相关的 paper/relation/open question 视图

### UI 目标

`Timeline`、`Trace`、`Snapshot` 三块必须一起证明“记忆改变了决策”。

## 测试计划

### 单元测试

- memory upsert 规则
- canonical key 规则
- sufficiency 判定
- termination logic
- session relevance 过滤

### 合约测试

- tool 输入输出契约
- repository port 行为一致性

### 集成测试

- arXiv ingest 流程
- 本地 PDF ingest 流程
- follow-up 在“记忆足够”和“需要回读原文”两条路径下的行为

### API 测试

- sessions
- ingest
- queries
- timeline
- memory snapshot
- trace
- SSE events

### 前端测试

- 三栏布局
- timeline / trace tab 切换
- run 与消息历史关联展示
- snapshot 渲染

### fixtures

- 示例 arXiv metadata
- 本地 PDF fixture
- mock LLM response fixture
- mock trace / timeline fixture

## 迁移到 PydanticAI 的路径

- 迁移目标是替换编排层，不是重写系统。
- 现有 `services` / `adapters` / `domain` 保持不动。
- tool input/output 维持 Pydantic model 风格。
- runtime 仍保留宿主侧 `step limit`、`finish_task`、`trace` 持久化。
- 后续将现有 tools 包装为 PydanticAI tools，逐步替换 orchestration 层。

## 补充约束

### 1. `relation_type` 枚举固定为

- `improves_on`
- `similar_to`
- `conflicts_with`
- `complements`
- `uses_same_benchmark`
- `compares_with`

### 2. `canonical_key` 规则固定为

- 有 arXiv id：`paper:arxiv:{id}`
- 无 arXiv id：`paper:pdf:{sha256}`
- 如果后续确认本地 PDF 对应某个 arXiv，只建立 alias，不直接修改主 `canonical_key`

### 3. memory upsert 规则

- `source_refs`：追加并去重
- `limitations` / `key_results` / `why_open`：允许增量补充
- `problem` / `method` / `novelty_claim`：默认保留旧值，只有新证据置信度更高时才覆盖
- 如果出现明显冲突，不直接覆盖，改为新增 relation memory 或 open question memory

### 4. follow-up 的记忆检索和 sufficiency 规则

- 先查 session memory，`top_k=5`
- 再查 global memory，`top_k=5`
- 如果命中至少 1 条相关 paper memory，且能覆盖问题核心对象，则优先直接回答
- 只有在以下情况才允许回读原文：
  - 缺少证据句
  - 缺少比较对象
  - memory confidence 低于阈值
  - memory 明显不足以回答问题

### 5. `open_question_memory` 的创建触发条件

- 论文明确给出 future work 或 limitation
- 用户问题无法由已有 `paper_memory` / `relation_memory` 充分回答
- 两篇论文之间存在未解决冲突、空白或待验证点

### 6. `TraceNarrative` 生成规则

- `TraceStep` 只存真实 `action` / `result` / `status` / `payload`
- `TraceNarrative` 单独存储 `reason_text` / `impact_text`
- 第一版在 run 结束后统一生成 `TraceNarrative`，不在每个 step 实时生成

### 7. SSE 事件格式固定为

事件类型：

- `run_started`
- `step_completed`
- `memory_updated`
- `timeline_updated`
- `run_finished`
- `run_failed`

统一字段：

- `event_type`
- `session_id`
- `run_id`
- `timestamp`
- `payload`

### 8. 前端历史显示规则

- 聊天区显示 message history
- assistant 消息可以关联和展开对应 run
- timeline / trace 默认看当前 session，也支持聚焦当前选中 run

## 当前阻塞信息

目前仍待确认：

- `LLM provider / model` 选型与凭据
- MVP 是否只支持可提取文本的 PDF，不做 OCR
