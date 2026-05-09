import type {
  MemoryBundles,
  Message,
  RuntimeStatus,
  Session,
  TaskRun,
  TimelineEvent,
  TraceNarrative,
  TraceStep,
} from "./types";

const SESSION_ID = "session-demo-memory-routed-agent";
const RUN_SEARCH_IMPORT = "run-demo-search-import-answer";
const RUN_MEMORY_FOLLOWUP = "run-demo-memory-followup";

const sessions: Session[] = [
  {
    id: SESSION_ID,
    title: "Memory-Routed CS Demo",
    created_at: "2026-05-08T10:00:00Z",
    updated_at: "2026-05-08T10:08:00Z",
    status: "active",
  },
];

const messages: Message[] = [
  {
    id: "msg-demo-user-1",
    session_id: SESSION_ID,
    role: "user",
    type: "followup_query",
    content: "先在 arXiv 搜索几篇 computer science 里关于 memory-routed paper agents 的论文，选一篇导入，再总结它的核心贡献。",
    created_at: "2026-05-08T10:00:10Z",
    status: "accepted",
  },
  {
    id: "msg-demo-assistant-1",
    session_id: SESSION_ID,
    role: "assistant",
    type: "followup_query",
    content: "我先用 search_arxiv 找到几篇相关论文，再显式导入其中一篇。当前 session 已导入 arXiv:2401.12345，对应论文《Memory-Routed Research Agents》。系统下载并解析 PDF，写入论文记忆和开放问题记忆。核心贡献是把“是否检索、何时回读原文”从固定 RAG 流程改成模型驱动决策，同时把结构化研究记忆作为后续问答的首要上下文。",
    created_at: "2026-05-08T10:01:24Z",
    status: "completed",
  },
  {
    id: "msg-demo-user-2",
    session_id: SESSION_ID,
    role: "user",
    type: "followup_query",
    content: "如果我下一轮继续追问，这些长期记忆会怎么影响回答？",
    created_at: "2026-05-08T10:06:05Z",
    status: "accepted",
  },
  {
    id: "msg-demo-assistant-2",
    session_id: SESSION_ID,
    role: "assistant",
    type: "followup_query",
    content: "后续回答会先检索当前 session 的论文记忆和开放问题记忆，再看全局记忆；只有这些记忆不足以覆盖你的问题时，系统才会重新回读原文 chunk。所以你会看到后续决策先被长期记忆改变，而不是每次都从 PDF 重新开始。",
    created_at: "2026-05-08T10:06:42Z",
    status: "completed",
  },
];

const runs: TaskRun[] = [
  {
    id: RUN_SEARCH_IMPORT,
    session_id: SESSION_ID,
    message_id: "msg-demo-user-1",
    status: "finished",
    step_count: 5,
    started_at: "2026-05-08T10:00:11Z",
    finished_at: "2026-05-08T10:01:23Z",
    finish_reason: "completed",
  },
  {
    id: RUN_MEMORY_FOLLOWUP,
    session_id: SESSION_ID,
    message_id: "msg-demo-user-2",
    status: "finished",
    step_count: 4,
    started_at: "2026-05-08T10:06:06Z",
    finished_at: "2026-05-08T10:06:41Z",
    finish_reason: "completed",
  },
];

const timeline: TimelineEvent[] = [
  {
    id: "evt-1",
    session_id: SESSION_ID,
    run_id: RUN_SEARCH_IMPORT,
    event_type: "run_started",
    summary: "问答已开始：先搜索候选论文，再决定是否导入。",
    related_memory_ids: [],
    related_paper_ids: [],
    created_at: "2026-05-08T10:00:11Z",
  },
  {
    id: "evt-2",
    session_id: SESSION_ID,
    run_id: RUN_SEARCH_IMPORT,
    event_type: "step_completed",
    summary: "search_arxiv 返回 3 篇候选论文。",
    related_memory_ids: [],
    related_paper_ids: [],
    created_at: "2026-05-08T10:00:26Z",
  },
  {
    id: "evt-3",
    session_id: SESSION_ID,
    run_id: RUN_SEARCH_IMPORT,
    event_type: "step_completed",
    summary: "import_arxiv_paper 导入 arXiv:2401.12345 并触发 ingest run。",
    related_memory_ids: [],
    related_paper_ids: ["paper:arxiv:2401.12345"],
    created_at: "2026-05-08T10:00:57Z",
  },
  {
    id: "evt-4",
    session_id: SESSION_ID,
    run_id: RUN_SEARCH_IMPORT,
    event_type: "step_completed",
    summary: "会话记忆检索命中新导入论文的 paper_memory 与 open_question_memory。",
    related_memory_ids: ["memory-paper-2401", "memory-open-2401"],
    related_paper_ids: ["paper:arxiv:2401.12345"],
    created_at: "2026-05-08T10:01:08Z",
  },
  {
    id: "evt-5",
    session_id: SESSION_ID,
    run_id: RUN_SEARCH_IMPORT,
    event_type: "assistant_message_committed",
    summary: "助手消息已写入，会话中保留导入摘要与研究结论。",
    related_memory_ids: ["memory-paper-2401", "memory-open-2401"],
    related_paper_ids: ["paper:arxiv:2401.12345"],
    created_at: "2026-05-08T10:01:23Z",
  },
  {
    id: "evt-6",
    session_id: SESSION_ID,
    run_id: RUN_MEMORY_FOLLOWUP,
    event_type: "run_started",
    summary: "追问开始：优先检索 session 与 global memory。",
    related_memory_ids: [],
    related_paper_ids: ["paper:arxiv:2401.12345"],
    created_at: "2026-05-08T10:06:06Z",
  },
  {
    id: "evt-7",
    session_id: SESSION_ID,
    run_id: RUN_MEMORY_FOLLOWUP,
    event_type: "step_completed",
    summary: "记忆足以回答，无需回读 PDF 原文。",
    related_memory_ids: ["memory-paper-2401", "memory-global-1"],
    related_paper_ids: ["paper:arxiv:2401.12345"],
    created_at: "2026-05-08T10:06:28Z",
  },
  {
    id: "evt-8",
    session_id: SESSION_ID,
    run_id: RUN_MEMORY_FOLLOWUP,
    event_type: "run_finished",
    summary: "问答完成，最终回答明确说明了长期记忆如何影响后续决策。",
    related_memory_ids: ["memory-paper-2401", "memory-open-2401", "memory-global-1"],
    related_paper_ids: ["paper:arxiv:2401.12345"],
    created_at: "2026-05-08T10:06:41Z",
  },
];

const traces: Record<string, { steps: TraceStep[]; narratives: TraceNarrative[] }> = {
  [RUN_SEARCH_IMPORT]: {
    steps: [
      {
        id: "trace-search-1",
        run_id: RUN_SEARCH_IMPORT,
        action: "search_arxiv",
        input_payload: {
          planner_decision: {
            selected_tool: "search_arxiv",
            rationale: "用户先要求搜索几篇候选论文，再决定导入。",
            fallback_used: false,
          },
          query: "computer science memory-routed paper agents",
          category: "cs",
        },
        result_payload: {
          success: true,
          query: "all:computer science memory-routed paper agents AND cat:cs",
          count: 3,
          papers: [
            { arxiv_id: "2401.12345", title: "Memory-Routed Research Agents" },
            { arxiv_id: "2402.23456", title: "Adaptive Retrieval for Paper Agents" },
            { arxiv_id: "2403.34567", title: "Session Memory for Scientific QA" },
          ],
        },
        status: "completed",
        started_at: "2026-05-08T10:00:12Z",
        finished_at: "2026-05-08T10:00:26Z",
      },
      {
        id: "trace-search-2",
        run_id: RUN_SEARCH_IMPORT,
        action: "import_arxiv_paper",
        input_payload: {
          planner_decision: {
            selected_tool: "import_arxiv_paper",
            rationale: "选取最贴近研究问题的一篇论文进入当前 session。",
            fallback_used: false,
          },
          arxiv_id_or_url: "2401.12345",
        },
        result_payload: {
          success: true,
          arxiv_id: "2401.12345",
          paper_id: "paper:arxiv:2401.12345",
          artifact_id: "artifact-arxiv-2401",
          chunk_count: 42,
          ingest_summary: "导入完成，已写入论文记忆与开放问题记忆。",
        },
        status: "completed",
        started_at: "2026-05-08T10:00:27Z",
        finished_at: "2026-05-08T10:00:57Z",
      },
      {
        id: "trace-search-3",
        run_id: RUN_SEARCH_IMPORT,
        action: "retrieve_session_memories",
        input_payload: {
          planner_decision: {
            selected_tool: "retrieve_session_memories",
            rationale: "导入之后先看这篇论文已经写下了哪些结构化记忆。",
            fallback_used: false,
          },
        },
        result_payload: {
          memory_ids: ["memory-paper-2401", "memory-open-2401"],
          coverage_score: 0.91,
          matched_query_terms: ["memory-routed", "agent", "decision"],
        },
        status: "completed",
        started_at: "2026-05-08T10:00:58Z",
        finished_at: "2026-05-08T10:01:08Z",
      },
      {
        id: "trace-search-4",
        run_id: RUN_SEARCH_IMPORT,
        action: "rerank_context_candidates",
        input_payload: {
          planner_decision: {
            selected_tool: "rerank_context_candidates",
            rationale: "优先用刚导入论文的 paper_memory，再补 open question。",
            fallback_used: false,
          },
        },
        result_payload: {
          selected_memory_ids: ["memory-paper-2401", "memory-open-2401"],
          selection_source: "session_memory_first",
          fallback_used: false,
        },
        status: "completed",
        started_at: "2026-05-08T10:01:09Z",
        finished_at: "2026-05-08T10:01:15Z",
      },
      {
        id: "trace-search-5",
        run_id: RUN_SEARCH_IMPORT,
        action: "compose_mock_answer",
        input_payload: {
          planner_decision: {
            selected_tool: "compose_mock_answer",
            rationale: "已有论文记忆足够回答核心贡献，无需再回读 PDF chunk。",
            fallback_used: false,
          },
        },
        result_payload: {
          answer_preview: "模型先搜索 arXiv，再导入一篇最相关论文，并基于新写入的论文记忆总结核心贡献。",
          memory_citations: [
            { memory_id: "memory-paper-2401" },
            { memory_id: "memory-open-2401" },
          ],
          source_reread_chunks: [],
        },
        status: "completed",
        started_at: "2026-05-08T10:01:16Z",
        finished_at: "2026-05-08T10:01:23Z",
      },
    ],
    narratives: [
      {
        trace_step_id: "trace-search-1",
        reason_text: "先拓展候选论文范围，避免只凭先验挑论文。",
        impact_text: "后续导入动作来自真实搜索结果，而不是硬编码 ID。",
      },
      {
        trace_step_id: "trace-search-2",
        reason_text: "导入沿用现有 ingest run，因此会触发真正的 PDF 下载、解析、记忆写入和摘要生成。",
        impact_text: "这一步把候选论文变成当前 session 可检索的长期记忆。",
      },
    ],
  },
  [RUN_MEMORY_FOLLOWUP]: {
    steps: [
      {
        id: "trace-follow-1",
        run_id: RUN_MEMORY_FOLLOWUP,
        action: "retrieve_session_memories",
        input_payload: {
          planner_decision: {
            selected_tool: "retrieve_session_memories",
            rationale: "追问直接依赖当前 session 的论文记忆。",
            fallback_used: false,
          },
        },
        result_payload: {
          memory_ids: ["memory-paper-2401", "memory-open-2401"],
          coverage_score: 0.95,
          matched_query_terms: ["长期记忆", "后续回答", "影响"],
        },
        status: "completed",
        started_at: "2026-05-08T10:06:07Z",
        finished_at: "2026-05-08T10:06:16Z",
      },
      {
        id: "trace-follow-2",
        run_id: RUN_MEMORY_FOLLOWUP,
        action: "retrieve_global_memories",
        input_payload: {
          planner_decision: {
            selected_tool: "retrieve_global_memories",
            rationale: "补充一条跨论文的全局经验记忆。",
            fallback_used: false,
          },
        },
        result_payload: {
          memory_ids: ["memory-global-1"],
          coverage_score: 0.62,
          matched_query_terms: ["memory-first", "research qa"],
        },
        status: "completed",
        started_at: "2026-05-08T10:06:17Z",
        finished_at: "2026-05-08T10:06:23Z",
      },
      {
        id: "trace-follow-3",
        run_id: RUN_MEMORY_FOLLOWUP,
        action: "rerank_context_candidates",
        input_payload: {
          planner_decision: {
            selected_tool: "rerank_context_candidates",
            rationale: "当前问题更像机制解释，不需要回读原文。",
            fallback_used: false,
          },
        },
        result_payload: {
          selected_memory_ids: ["memory-paper-2401", "memory-global-1"],
          selection_source: "session_then_global",
          fallback_used: false,
        },
        status: "completed",
        started_at: "2026-05-08T10:06:24Z",
        finished_at: "2026-05-08T10:06:28Z",
      },
      {
        id: "trace-follow-4",
        run_id: RUN_MEMORY_FOLLOWUP,
        action: "direct_final_answer",
        input_payload: {
          retrieval_skipped: true,
          planner_decision: {
            selected_tool: "direct_final_answer",
            rationale: "session 和 global memory 已经足够回答。",
            fallback_used: false,
          },
        },
        result_payload: {
          answer_preview: "后续问答会先命中论文记忆和开放问题记忆，只有不足时才会回读原文。",
        },
        status: "completed",
        started_at: "2026-05-08T10:06:29Z",
        finished_at: "2026-05-08T10:06:41Z",
      },
    ],
    narratives: [
      {
        trace_step_id: "trace-follow-4",
        reason_text: "这个问题问的是系统如何决策，不是论文细节，因此记忆已经足够。",
        impact_text: "你可以直接看到 memory-first 路径如何改变后续行为。",
      },
    ],
  },
};

const sessionMemoryBundles: MemoryBundles = {
  papers: [
    {
      paper: {
        paper_id: "paper:arxiv:2401.12345",
        title: "Memory-Routed Research Agents",
        file_name: "2401.12345.pdf",
        created_at: "2026-05-08T10:00:57Z",
        updated_at: "2026-05-08T10:01:12Z",
        memory_count: 2,
      },
      paper_memories: [
        {
          id: "memory-paper-2401",
          memory_type: "paper_memory",
          content:
            "The paper proposes a memory-routed research agent that decides whether to answer from structured memory, search additional memory, or reread source passages instead of following a fixed retrieve-then-generate pipeline.",
          created_at: "2026-05-08T10:01:02Z",
          updated_at: "2026-05-08T10:01:12Z",
          paper_id: "paper:arxiv:2401.12345",
          source_paper: null,
          target_paper: null,
          relation_direction: null,
          relation_type: null,
          related_papers: [],
          source_chunk_ids: ["chunk-2401-4", "chunk-2401-9"],
          evidence_count: 2,
        },
      ],
      open_question_memories: [
        {
          id: "memory-open-2401",
          memory_type: "open_question_memory",
          content:
            "Open question: how stable are the memory-routing decisions when the session accumulates many partially overlapping papers and the model must avoid stale summaries?",
          created_at: "2026-05-08T10:01:05Z",
          updated_at: "2026-05-08T10:01:12Z",
          paper_id: "paper:arxiv:2401.12345",
          source_paper: null,
          target_paper: null,
          relation_direction: null,
          relation_type: null,
          related_papers: ["paper:arxiv:2401.12345"],
          source_chunk_ids: ["chunk-2401-31"],
          evidence_count: 1,
        },
      ],
      relation_memories: [],
      source_chunks: [
        {
          chunk_id: "chunk-2401-4",
          paper_id: "paper:arxiv:2401.12345",
          page: 2,
          section: "Introduction",
          excerpt: "We replace fixed retrieval ladders with model-routed decisions over session memory, global memory, and source reread.",
        },
        {
          chunk_id: "chunk-2401-31",
          paper_id: "paper:arxiv:2401.12345",
          page: 8,
          section: "Limitations",
          excerpt: "Routing quality may degrade as the session accumulates overlapping summaries and unresolved questions.",
        },
      ],
      source_chunk_count: 2,
      empty_fields: [],
    },
  ],
  unscoped_memories: [],
};

const globalMemoryBundles: MemoryBundles = {
  papers: [
    {
      paper: {
        paper_id: "paper:arxiv:2310.56789",
        title: "Persistent Memory for Scientific QA",
        file_name: "2310.56789.pdf",
        created_at: "2026-04-30T08:10:00Z",
        updated_at: "2026-05-01T09:20:00Z",
        memory_count: 1,
      },
      paper_memories: [
        {
          id: "memory-global-1",
          memory_type: "paper_memory",
          content:
            "Global lesson: scientific agents answer more consistently when session memory is checked before any fresh passage reread.",
          created_at: "2026-05-01T09:20:00Z",
          updated_at: "2026-05-01T09:20:00Z",
          paper_id: "paper:arxiv:2310.56789",
          source_paper: null,
          target_paper: null,
          relation_direction: null,
          relation_type: null,
          related_papers: [],
          source_chunk_ids: ["chunk-2310-7"],
          evidence_count: 1,
        },
      ],
      open_question_memories: [],
      relation_memories: [],
      source_chunks: [
        {
          chunk_id: "chunk-2310-7",
          paper_id: "paper:arxiv:2310.56789",
          page: 3,
          section: "Approach",
          excerpt: "Memory-first routing reduces unnecessary rereads and makes later answers auditable.",
        },
      ],
      source_chunk_count: 1,
      empty_fields: [],
    },
  ],
  unscoped_memories: [],
};

const runtimeStatus: RuntimeStatus = {
  app_name: "OpenViking Memory-Routed Paper Agent",
  storage_backend: "sqlite",
  sqlite_path: "./data/sqlite/research_agent.sqlite3",
  query_agent_backend: "turn_adapter",
  query_agent_provider: "deepseek",
  query_agent_model: "deepseek-v4-flash",
  ingest_extraction_backend: "model_adapter",
  ingest_extraction_provider: "deepseek",
  ingest_extraction_model: "deepseek-v4-flash",
  openviking_backend: "embedded",
  openviking_data_path: "./data/openviking",
  openviking_url: null,
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
    },
  });
}

function match(pathname: string, pattern: RegExp): RegExpMatchArray | null {
  return pathname.match(pattern);
}

function readOnlyResponse(): Response {
  return jsonResponse({ detail: "Demo is read-only. Use Docker deployment for the live app." }, 405);
}

function handleApiRequest(method: string, pathname: string): Response | null {
  if (method !== "GET") {
    return readOnlyResponse();
  }

  if (pathname === "/api/sessions") {
    return jsonResponse({ items: sessions });
  }
  if (pathname === `/api/sessions/${SESSION_ID}/messages`) {
    return jsonResponse({ items: messages });
  }
  if (pathname === `/api/sessions/${SESSION_ID}/runs`) {
    return jsonResponse({ items: runs });
  }
  if (pathname === `/api/sessions/${SESSION_ID}/timeline`) {
    return jsonResponse({ items: timeline });
  }
  if (pathname === `/api/sessions/${SESSION_ID}/memory-bundles`) {
    return jsonResponse(sessionMemoryBundles);
  }
  if (pathname === "/api/memories/global-bundles") {
    return jsonResponse(globalMemoryBundles);
  }
  if (pathname === "/api/system/runtime") {
    return jsonResponse(runtimeStatus);
  }

  const traceMatch = match(pathname, new RegExp(`^/api/sessions/${SESSION_ID}/runs/([^/]+)/trace$`));
  if (traceMatch) {
    return jsonResponse(traces[traceMatch[1]] ?? { steps: [], narratives: [] });
  }

  const eventsMatch = match(pathname, new RegExp(`^/api/sessions/${SESSION_ID}/runs/([^/]+)/events$`));
  if (eventsMatch) {
    return jsonResponse({
      items: timeline.filter((event) => event.run_id === eventsMatch[1]),
    });
  }

  if (pathname === `/api/sessions/${SESSION_ID}/memory-snapshot`) {
    return jsonResponse({
      paper_memories: [],
      relation_memories: [],
      open_question_memories: [],
    });
  }

  return jsonResponse({ detail: "Demo route not found." }, 404);
}

export function installDemoApi(): void {
  if (typeof window === "undefined") {
    return;
  }
  const originalFetch = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const request = input instanceof Request ? input : null;
    const method = init?.method ?? request?.method ?? "GET";
    const rawUrl = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    const url = new URL(rawUrl, window.location.origin);

    if (!url.pathname.startsWith("/api/")) {
      return originalFetch(input, init);
    }

    const response = handleApiRequest(method.toUpperCase(), url.pathname);
    if (response) {
      return response;
    }
    return jsonResponse({ detail: "Demo route not found." }, 404);
  };
}
