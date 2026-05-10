# Memory-Routed Paper Agent

> 一个面向论文研究场景的 Agent：先导入论文、写入长期记忆，再让后续问答优先使用记忆，而不是每次都从 PDF 重新开始。

## 核心架构选择

这个项目在架构上保留了 OpenViking 作为长期记忆 / 检索适配层的明确边界，而不是把它完全混进运行时逻辑里。

- **OpenViking 是什么**
  - 一个面向 Agent / 记忆 / 检索场景的基础设施项目
  - 在这个仓库里，它对应的是“可选的长期记忆与检索适配层”，而不是当前默认部署的硬依赖

- **为什么这里要选它**
  - 这个项目需要的不只是本地 chunk 检索，还需要一个清晰的“长期记忆系统”扩展方向
  - OpenViking 给了比较明确的 memory / retrieval 边界，适合把 session memory、global memory 和后续更强的长期记忆能力留出演进空间
  - 因此当前实现保留了 OpenViking 适配层，但把 runtime 的主语义继续掌握在本仓库自己手里

- **tradeoff**
  - 好处是：长期记忆边界更清楚，未来迁移和扩展空间更大
  - 代价是：本地启动和部署复杂度会上升，所以当前默认运行路径仍然推荐 `SQLite + noop`
  - 也就是说：架构上为 OpenViking 预留位置，但产品演示和默认部署不强依赖它

- **背景讨论**
  - RFC #1190: https://github.com/volcengine/OpenViking/discussions/1190

<p align="center">
  <img src="docs/demo.gif" width="800" alt="demo" />
</p>

## 项目定位

这不是一个普通的“论文 PDF 聊天工具”。

这个项目更关心的是：

- 论文导入之后，系统能否把理解结果沉淀成结构化研究记忆
- 后续追问时，系统能否先使用这些记忆，再决定是否回读原文
- 前端能否把这条决策链清楚展示出来

所以它的核心价值不是“也能回答论文问题”，而是“能看见长期记忆如何影响下一轮回答”。

## 和普通 RAG 的区别

| 对比项 | 标准 RAG | 本项目 |
|---|---|---|
| 问答路径 | 基本固定：query -> retrieve -> generate | 由模型决定是否检索、是否回读原文、是否直接回答 |
| 论文导入 | 常见做法是切 chunk 后直接检索 | 先导入，再提取并持久化结构化记忆 |
| 长期记忆 | 通常较弱或不存在 | 明确支持 `paper_memory`、`relation_memory`、`open_question_memory` |
| 可观测性 | 多数只看最终回答 | 前端展示 trace、timeline、memory drawer、工具调用与思考过程 |

<p align="center">
  <img src="docs/main.png" width="800" alt="main interface" />
</p>

## 项目能力

- **论文导入**
  - 支持直接粘贴 arXiv 链接
  - 支持上传本地 PDF
  - 导入后会下载 / 解析 PDF、切 chunk、注册 paper、写入记忆并生成摘要

- **arXiv 搜索与导入链路**
  - 支持通过官方 arXiv API 搜索候选论文
  - 搜索结果返回 `arxiv_id`、标题、作者、摘要、分类、abs URL、pdf URL
  - 模型可以先搜索，再显式调用 `import_arxiv_paper`
  - `import_arxiv_paper` 复用现有 ingest run，不额外发明一套 PDF URL 导入逻辑

- **长期记忆优先的问答**
  - 后续问题先查 session memory
  - 再查 global memory
  - 只有记忆不足时才回读原文 chunk

- **模型驱动工具调用**
  - 模型可以自己决定下一步是搜索 arXiv、导入论文、检索记忆、回读原文还是直接回答
  - 宿主 runtime 仍然控制生命周期、step limit、finish/fail、trace 和 timeline

- **失败兜底**
  - arXiv 搜索不到、网络失败或导入失败时，会返回结构化 `no_results`
  - 不会因为一次 arXiv 工具失败就让整轮问答直接崩掉

- **实时可视化**
  - 前端可以看到 step-by-step 的工具调用
  - 可以查看 timeline、trace reasoning、memory drawer
  - 能看到“为什么这一步用了记忆 / 为什么这一步回读了原文”

<p align="center">
  <img src="docs/memory.png" width="800" alt="memory panel" />
</p>

## Demo 与部署

### 静态 Demo

- 地址：`https://zoean-z.github.io/ResearchAgent/`
- 这是一个只读静态演示页
- 使用真实前端 + mock `/api` 数据
- 会自动播放多轮消息，展示：
  - 搜索 arXiv
  - 导入论文
  - 长期记忆如何影响后续回答
  - 与标准 RAG 的区别

### Docker 一键部署

最简单的真实运行方式是：

```bash
docker compose up --build
```

启动后访问：

- `http://127.0.0.1:8011/`

如果你只想把项目跑起来，Docker 部署只需要一个最小 `.env`：

```env
DEEPSEEK_API_KEY=你的_key
```

因为 `docker-compose.yml` 已经内置了这些默认项：

- `RESEARCH_AGENT_STORAGE_BACKEND=sqlite`
- `RESEARCH_AGENT_SQLITE_PATH=/app/data/sqlite/research_agent.sqlite3`
- `RESEARCH_AGENT_OPENVIKING_BACKEND=noop`
- `RESEARCH_AGENT_QUERY_AGENT_BACKEND=turn_adapter`

也就是说，**Docker 路径下通常不需要手动重复配置一大串 env**。  
只要给出 `DEEPSEEK_API_KEY`，就能用 SQLite 跑起当前主流程。

推荐做法：

1. 在仓库根目录新建 `.env`
2. 只写：

```env
DEEPSEEK_API_KEY=你的_key
```

3. 运行：

```bash
docker compose up --build
```

4. 打开：

```text
http://127.0.0.1:8011/
```

### OpenViking 说明

当前仓库里虽然保留了 OpenViking 相关边界和适配层，但**默认部署路径并不依赖 OpenViking**。

目前更实际的结论是：

- Docker 默认使用 `RESEARCH_AGENT_OPENVIKING_BACKEND=noop`
- 也就是：**默认不开 OpenViking**
- 当前最稳定、最推荐的运行方式是 **SQLite + noop**

如果你只是想演示、体验或部署这个项目，可以直接把 OpenViking 当作“当前不用配置的可选层”。

README 这里不再把它写成必需组件，避免让第一次运行的人误以为还要先搭一套 OpenViking 服务。

## 本地开发

如果你不用 Docker，也可以本地分别启动前后端。

后端：

```bash
cd backend
pip install -e .
uvicorn research_agent.api.app:app --host 0.0.0.0 --port 8011
```

前端：

```bash
cd frontend
npm install
npm run build
```

Windows 下仓库还提供了一个启动脚本：

```powershell
.\scripts\start-dev.ps1
```

但要注意：

- 这个脚本是早期本地开发路径的一部分
- 它会尝试走 embedded OpenViking 相关准备逻辑
- 如果你的目标只是“尽快跑起来”，优先推荐 Docker，而不是先折腾 OpenViking

## 最小环境变量

为了避免 `.env` 太冗长，当前建议按“最小可运行”原则理解：

```env
DEEPSEEK_API_KEY=你的_key
RESEARCH_AGENT_OPENVIKING_BACKEND=noop
```

其他大部分参数都有默认值，只有在你明确要改模型、改后端或做更细粒度调试时，才需要继续展开配置。

## 运行时主链路

当前系统的主要调用链大致是：

1. 用户输入问题 / arXiv 链接 / 上传 PDF
2. 前端或 query runtime 判断是普通问答还是论文导入
3. 导入路径解析 PDF、注册论文、写入 chunk 和结构化记忆
4. 后续问答优先使用 session / global memory
5. 只有记忆不足时才回读 source passages
6. 结果通过 trace、timeline 和 memory drawer 在前端展示

## 项目结构

```text
backend/    FastAPI API、runtime、services、tools、storage adapters
frontend/   React + TypeScript 前端工作台
data/       本地 artifacts 与 SQLite 数据
docs/       项目说明、接口文档、演示素材
scripts/    启动与辅助脚本
```

## 技术栈

- Backend: Python / FastAPI / SQLite / Pydantic
- Frontend: React / TypeScript / Vite
- LLM: DeepSeek API
- Memory Layer: SQLite 为当前主路径，OpenViking 保留为可选适配层

<p align="center">
  <img src="docs/setting.png" width="800" alt="settings panel" />
</p>

## 适合展示的点

如果这是一个求职或项目展示仓库，最值得看的不是“它能不能回答论文问题”，而是下面这些：

- 它把论文导入、长期记忆、追问和可视化 trace 串成了一条完整链路
- 它不是固定 RAG，而是模型驱动的工具调用和决策
- 它能展示“长期记忆如何影响下一轮回答”
- 它既有静态 demo，也有 Docker 一键部署入口

## 后续仍可扩展的方向

- 更完整的 token 级流式输出
- 更强的 ingest 质量控制与评估
- 更丰富的 memory editing / inspection 工具
- 在不改变核心 runtime 语义的前提下继续扩展工具面
