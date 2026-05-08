# Memory-Routed Paper Agent

> 一个论文阅读 Agent —— 模型自主决定是否检索、何时读原文、怎样回答，而非固定管线。

<p align="center">
  <img src="docs/demo.gif" width="800" alt="demo" />
</p>

---

## 核心功能

- **论文导入**：粘贴 arXiv 链接或上传本地 PDF，自动解析分块、提取结构化记忆（论文要点、关联知识、开放问题）
- **模型驱动的工具调用**：模型自主决定下一步动作 —— 搜索记忆、读取原文、直接回答，由宿主循环执行并校验
- **记忆系统**：基于 OpenViking 的长期记忆，支持论文记忆、关联记忆、开放问题记忆的存储与检索
- **实时推理流**：前端实时展示模型的思考过程、工具调用、中间结果，而不是等全部跑完才显示
- **追问与上下文**：自动注入最近对话上下文，支持追问时不重复检索
- **可观测性**：每一步的 planner 决策、工具输入输出、记忆引用都有 trace 记录，可追溯

<p align="center">
  <img src="docs/main.png" width="800" alt="主界面" />
</p>

## 和 RAG 的区别

| | RAG | 本项目 |
|---|---|---|
| 流程 | 固定管线：query → retrieve → generate | 模型自主决定：可直接回答，也可多次检索 |
| 检索 | 必须检索，不管需不需要 | 模型判断是否需要检索 |
| 工具 | 无工具调用 | 模型选择工具，宿主执行 |
| 上下文 | 检索结果拼接 | 分层上下文：记忆摘要 + 对话历史 + 原文片段 |

<p align="center">
  <img src="docs/memory.png" width="800" alt="记忆面板" />
</p>

## 踩过的坑

### 1. DeepSeek 返回空 body 但 HTTP 200

DeepSeek 限流时不返回标准的 429，而是返回 HTTP 200 + 空 body。

**解法**：在解析响应前先检查 body 是否为空，空则抛出限流异常，触发指数退避重试（5s/10s/15s）。

### 2. 模型返回 markdown 包裹的 JSON

模型有时返回 ` ```json\n{...}\n``` ` 或者先输出一段文字再附上 JSON，直接 `json.loads` 会失败。

**解法**：先尝试去掉 markdown 代码块标记，再用 `{` 和 `}` 的首尾位置提取 JSON 子串，两步容错。

### 3. max_tokens 不够导致 JSON 被截断

模型返回 10 个 chunk UUID + 中文理由时，640 token 不够，JSON 被截断导致解析失败。

**解法**：`choose_next_action` 的 `max_tokens` 从 640 提升到 2048。

### 4. 模型不读原文就编造论文内容

早期 prompt 鼓励"优先不检索直接回答"，导致模型凭记忆摘要编造论文细节。

**解法**：在 prompt 中加入 CRITICAL 指令 —— 回答论文具体内容（方法、模型、数据集、实验结果）时必须先读原文。

### 5. 单文件过大难以维护

`query_execution_service.py` 曾膨胀到 1988 行。

**解法**：拆分为 6 个模块 —— `query_execution_models`、`query_citation_builder`、`query_observation_builder`、`query_answer_composer`、`query_trace_writer` 和瘦身后的主服务。

## 快速开始

### 后端

```bash
cd backend
pip install -e .
cp .env.example .env  # 填入 DEEPSEEK_API_KEY
uvicorn research_agent.api.app:app --port 8011
```

### 前端

```bash
cd frontend
npm install
npm run build  # 构建后由后端同源服务
```

或使用一键脚本（Windows）：

```powershell
.\scripts\start-dev.ps1
```

启动后访问 `http://127.0.0.1:8011/`。

## 技术栈

- **后端**：Python / FastAPI / SQLite / Pydantic
- **前端**：React / TypeScript / Vite
- **LLM**：DeepSeek API
- **记忆层**：OpenViking（可选，有 Noop fallback）

<p align="center">
  <img src="docs/setting.png" width="800" alt="设置面板" />
</p>

## 项目结构

```
backend/          Python 服务、领域模型、运行时、适配器、API
frontend/         React 工作台：会话、对话、导入、时间线、记忆
data/             本地制品和 SQLite 存储
docs/             项目规格文档
scripts/          启动脚本
```

---

# Memory-Routed Paper Agent

> A paper-reading Agent that's NOT RAG — the model decides whether to retrieve, when to read the source, and how to answer, instead of following a fixed pipeline.

<p align="center">
  <img src="docs/demo.gif" width="800" alt="demo" />
</p>

---

## Core Features

- **Paper Import**: Paste an arXiv link or upload a local PDF — the system auto-parses, chunks, and extracts structured memories (key points, related knowledge, open questions)
- **Model-Driven Tool Calling**: The model autonomously decides the next action — search memory, read source passages, or answer directly — with the host loop executing and validating each step
- **Memory System**: Long-term memory powered by OpenViking, supporting paper memories, relation memories, and open-question memories
- **Real-Time Reasoning Stream**: The frontend live-displays the model's thinking process, tool calls, and intermediate results instead of waiting for completion
- **Follow-Up & Context**: Automatically injects recent conversation context so follow-up questions don't require redundant retrieval
- **Observability**: Every step's planner decision, tool input/output, and memory reference is recorded in trace for full auditability

<p align="center">
  <img src="docs/main.png" width="800" alt="Main interface" />
</p>

## How This Differs from RAG

| | RAG | This Project |
|---|---|---|
| Pipeline | Fixed: query → retrieve → generate | Model-driven: can answer directly or retrieve multiple times |
| Retrieval | Must retrieve, regardless of need | Model decides whether retrieval is needed |
| Tools | No tool calling | Model selects tools, host executes |
| Context | Retrieved chunks concatenated | Layered: memory summary + conversation history + source passages |

<p align="center">
  <img src="docs/memory.png" width="800" alt="Memory panel" />
</p>

## Pitfalls & Solutions

### 1. DeepSeek returns empty body with HTTP 200

DeepSeek's rate-limiting doesn't return a standard 429 — instead it returns HTTP 200 with an empty body.

**Fix**: Check for empty body before parsing; raise a rate-limit exception to trigger exponential backoff (5s/10s/15s).

### 2. Model returns markdown-wrapped JSON

The model sometimes returns ` ```json\n{...}\n``` ` or prose followed by JSON, causing `json.loads` to fail.

**Fix**: Strip markdown code block markers first, then extract the JSON substring between the first `{` and last `}` — two-layer fault tolerance.

### 3. max_tokens truncates JSON

When the model returns 10 chunk UUIDs + Chinese rationale, 640 tokens isn't enough — the JSON gets truncated.

**Fix**: Increased `choose_next_action` max_tokens from 640 to 2048.

### 4. Model fabricates paper content without reading source

Early prompts encouraged "prefer answering without retrieval," causing the model to invent paper details from memory summaries.

**Fix**: Added CRITICAL instructions in the prompt — when answering about specific paper content (methods, models, datasets, experiment results), the model MUST read the source first.

### 5. Single file too large to maintain

`query_execution_service.py` once ballooned to 1988 lines.

**Fix**: Split into 6 modules — `query_execution_models`, `query_citation_builder`, `query_observation_builder`, `query_answer_composer`, `query_trace_writer`, and the slimmed-down main service.

## Quick Start

### Backend

```bash
cd backend
pip install -e .
cp .env.example .env  # Add your DEEPSEEK_API_KEY
uvicorn research_agent.api.app:app --port 8011
```

### Frontend

```bash
cd frontend
npm install
npm run build  # Served by backend as same-origin
```

Or use the one-click script (Windows):

```powershell
.\scripts\start-dev.ps1
```

Then visit `http://127.0.0.1:8011/`.

## Tech Stack

- **Backend**: Python / FastAPI / SQLite / Pydantic
- **Frontend**: React / TypeScript / Vite
- **LLM**: DeepSeek API
- **Memory Layer**: OpenViking (optional, with Noop fallback)

<p align="center">
  <img src="docs/setting.png" width="800" alt="Settings panel" />
</p>

## Project Structure

```
backend/          Python service, domain models, runtime, adapters, API
frontend/         React workbench: sessions, chat, ingest, timeline, memory
data/             Local artifacts and SQLite storage
docs/             Project specification
scripts/          Startup scripts
```
