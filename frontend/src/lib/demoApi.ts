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

type DemoRunDefinition = {
  id: string;
  messageId: string;
  userVisibleAt: number;
  assistantVisibleAt: number;
  stepCount: number;
  startedAt: string;
  finishedAt: string;
};

type DemoTrace = {
  steps: TraceStep[];
  narratives: TraceNarrative[];
};

const sessions: Session[] = [
  {
    id: SESSION_ID,
    title: "记忆路由研究 Agent 演示",
    created_at: "2026-05-08T10:00:00Z",
    updated_at: "2026-05-08T10:12:30Z",
    status: "active",
  },
];

const allMessages: Message[] = [
  {
    id: "msg-demo-user-1",
    session_id: SESSION_ID,
    role: "user",
    type: "followup_query",
    content:
      "先在 arXiv 搜索几篇计算机领域关于记忆路由研究 Agent 的论文，导入其中一篇，再总结它的核心贡献。",
    created_at: "2026-05-08T10:00:10Z",
    status: "accepted",
  },
  {
    id: "msg-demo-assistant-1",
    session_id: SESSION_ID,
    role: "assistant",
    type: "followup_query",
    content:
      "我先搜索了 arXiv，选中了 arXiv:2401.12345，并通过现有 ingest 链路完成导入。这篇论文的核心贡献，是把固定的“检索后生成”流程改成由模型在会话记忆、全局记忆和原文回读之间自主决策，因此后续回答会先受长期研究记忆影响，而不是每次都从 PDF 重新开始。",
    created_at: "2026-05-08T10:01:24Z",
    status: "completed",
  },
  {
    id: "msg-demo-user-2",
    session_id: SESSION_ID,
    role: "user",
    type: "followup_query",
    content: "如果我继续追问，长期记忆会怎样影响下一轮回答？",
    created_at: "2026-05-08T10:03:05Z",
    status: "accepted",
  },
  {
    id: "msg-demo-assistant-2",
    session_id: SESSION_ID,
    role: "assistant",
    type: "followup_query",
    content:
      "下一轮回答会先使用导入时写下的会话记忆，再检查相关的全局记忆，只有这些记忆不足时才会回读原文 chunk。这意味着后续决策会明显体现“记忆优先”，而不是“PDF 优先”。",
    created_at: "2026-05-08T10:03:42Z",
    status: "completed",
  },
  {
    id: "msg-demo-user-3",
    session_id: SESSION_ID,
    role: "user",
    type: "followup_query",
    content: "再用一小段话，把这个 Agent 和标准 RAG 工作流做个对比。",
    created_at: "2026-05-08T10:06:10Z",
    status: "accepted",
  },
  {
    id: "msg-demo-assistant-3",
    session_id: SESSION_ID,
    role: "assistant",
    type: "followup_query",
    content:
      "标准 RAG 往往要求每个问题都先走同一条检索链路，这个 Agent 不是这样。它可以直接回答，也可以检索会话记忆、检索全局记忆，或者回读原文片段，具体走哪条路径由模型判断是否必要，trace 里还能看到它为什么这么选。",
    created_at: "2026-05-08T10:06:34Z",
    status: "completed",
  },
  {
    id: "msg-demo-user-4",
    session_id: SESSION_ID,
    role: "user",
    type: "followup_query",
    content: "导入之后，长期记忆里具体会存下什么？",
    created_at: "2026-05-08T10:09:15Z",
    status: "accepted",
  },
  {
    id: "msg-demo-assistant-4",
    session_id: SESSION_ID,
    role: "assistant",
    type: "followup_query",
    content:
      "长期记忆层会存结构化的论文记忆、开放问题，以及支撑这些记忆的原文 chunk。在这个演示里你还能看到一条相关的全局论文记忆，它让后续会话可以复用稳定结论，而不必每次重新导入同一篇论文。",
    created_at: "2026-05-08T10:09:44Z",
    status: "completed",
  },
];

const runDefinitions: DemoRunDefinition[] = [
  {
    id: "run-demo-search-import-answer",
    messageId: "msg-demo-user-1",
    userVisibleAt: 1,
    assistantVisibleAt: 2,
    stepCount: 5,
    startedAt: "2026-05-08T10:00:11Z",
    finishedAt: "2026-05-08T10:01:23Z",
  },
  {
    id: "run-demo-memory-followup",
    messageId: "msg-demo-user-2",
    userVisibleAt: 3,
    assistantVisibleAt: 4,
    stepCount: 4,
    startedAt: "2026-05-08T10:03:06Z",
    finishedAt: "2026-05-08T10:03:41Z",
  },
  {
    id: "run-demo-rag-comparison",
    messageId: "msg-demo-user-3",
    userVisibleAt: 5,
    assistantVisibleAt: 6,
    stepCount: 4,
    startedAt: "2026-05-08T10:06:11Z",
    finishedAt: "2026-05-08T10:06:33Z",
  },
  {
    id: "run-demo-memory-snapshot",
    messageId: "msg-demo-user-4",
    userVisibleAt: 7,
    assistantVisibleAt: 8,
    stepCount: 3,
    startedAt: "2026-05-08T10:09:16Z",
    finishedAt: "2026-05-08T10:09:43Z",
  },
];

const fullTraces: Record<string, DemoTrace> = {
  "run-demo-search-import-answer": {
    steps: [
      buildStep("trace-search-1", "run-demo-search-import-answer", "search_arxiv", {
        planner_decision: {
          selected_tool: "search_arxiv",
          rationale: "先搜索候选论文，再决定把哪一篇导入当前会话。",
          fallback_used: false,
        },
        query: "computer science memory-routed research agents",
        category: "cs",
      }, {
        success: true,
        query: "all:computer science memory-routed research agents AND cat:cs",
        count: 3,
        papers: [
          { arxiv_id: "2401.12345", title: "Memory-Routed Research Agents" },
          { arxiv_id: "2402.23456", title: "Adaptive Retrieval for Paper Agents" },
          { arxiv_id: "2403.34567", title: "Session Memory for Scientific QA" },
        ],
      }, "2026-05-08T10:00:12Z", "2026-05-08T10:00:26Z"),
      buildStep("trace-search-2", "run-demo-search-import-answer", "import_arxiv_paper", {
        planner_decision: {
          selected_tool: "import_arxiv_paper",
          rationale: "Import the most relevant candidate through the existing arXiv ingest run.",
          fallback_used: false,
        },
        arxiv_id_or_url: "2401.12345",
      }, {
        success: true,
        arxiv_id: "2401.12345",
        paper_id: "paper:arxiv:2401.12345",
        artifact_id: "artifact-arxiv-2401",
        chunk_count: 42,
        ingest_summary: "导入完成，论文记忆和开放问题记忆已经写入。",
      }, "2026-05-08T10:00:27Z", "2026-05-08T10:00:57Z"),
      buildStep("trace-search-3", "run-demo-search-import-answer", "retrieve_session_memories", {
        planner_decision: {
          selected_tool: "retrieve_session_memories",
          rationale: "先使用刚写入的会话记忆，再决定是否回读原文。",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-paper-2401", "memory-open-2401"],
        coverage_score: 0.91,
        matched_query_terms: ["memory-routed", "agent", "decision"],
      }, "2026-05-08T10:00:58Z", "2026-05-08T10:01:08Z"),
      buildStep("trace-search-4", "run-demo-search-import-answer", "rerank_context_candidates", {
        planner_decision: {
          selected_tool: "rerank_context_candidates",
          rationale: "优先使用这篇论文的 paper memory，并保留 open question 作为辅助信号。",
          fallback_used: false,
        },
      }, {
        selected_memory_ids: ["memory-paper-2401", "memory-open-2401"],
        selection_source: "session_memory_first",
        fallback_used: false,
      }, "2026-05-08T10:01:09Z", "2026-05-08T10:01:15Z"),
      buildStep("trace-search-5", "run-demo-search-import-answer", "compose_mock_answer", {
        planner_decision: {
          selected_tool: "compose_mock_answer",
          rationale: "现有论文记忆已经足以总结核心贡献。",
          fallback_used: false,
        },
      }, {
        answer_preview:
          "Agent 先搜索 arXiv，再导入一篇论文，并用新写入的记忆总结核心贡献。",
        memory_citations: [{ memory_id: "memory-paper-2401" }, { memory_id: "memory-open-2401" }],
        source_reread_chunks: [],
      }, "2026-05-08T10:01:16Z", "2026-05-08T10:01:23Z"),
    ],
    narratives: [
      {
        trace_step_id: "trace-search-1",
        reason_text: "先扩展候选论文集合，避免在导入前就把目标写死。",
        impact_text: "导入选择来自真实搜索结果，而不是硬编码的 paper id。",
      },
      {
        trace_step_id: "trace-search-2",
        reason_text: "导入复用了现有 ingest 链路，而不是额外造一条独立的 PDF 路径。",
        impact_text: "这篇论文会变成可持续影响后续回答的会话记忆。",
      },
    ],
  },
  "run-demo-memory-followup": {
    steps: [
      buildStep("trace-follow-1", "run-demo-memory-followup", "retrieve_session_memories", {
        planner_decision: {
          selected_tool: "retrieve_session_memories",
          rationale: "追问应该先从当前会话的论文记忆开始。",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-paper-2401", "memory-open-2401"],
        coverage_score: 0.95,
        matched_query_terms: ["长期记忆", "下一轮回答", "追问"],
      }, "2026-05-08T10:03:07Z", "2026-05-08T10:03:16Z"),
      buildStep("trace-follow-2", "run-demo-memory-followup", "retrieve_global_memories", {
        planner_decision: {
          selected_tool: "retrieve_global_memories",
          rationale: "先补充一条可复用的全局经验，再决定是否还需要原文。",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-global-1"],
        coverage_score: 0.62,
        matched_query_terms: ["memory-first", "research qa"],
      }, "2026-05-08T10:03:17Z", "2026-05-08T10:03:23Z"),
      buildStep("trace-follow-3", "run-demo-memory-followup", "rerank_context_candidates", {
        planner_decision: {
          selected_tool: "rerank_context_candidates",
          rationale: "这是机制解释类问题，记忆通常已经足够。",
          fallback_used: false,
        },
      }, {
        selected_memory_ids: ["memory-paper-2401", "memory-global-1"],
        selection_source: "session_then_global",
        fallback_used: false,
      }, "2026-05-08T10:03:24Z", "2026-05-08T10:03:28Z"),
      buildStep("trace-follow-4", "run-demo-memory-followup", "direct_final_answer", {
        retrieval_skipped: true,
        planner_decision: {
          selected_tool: "direct_final_answer",
          rationale: "会话记忆和全局记忆已经足以覆盖答案。",
          fallback_used: false,
        },
      }, {
        answer_preview: "后续回答会先命中论文记忆和开放问题，再考虑是否回读原文。",
      }, "2026-05-08T10:03:29Z", "2026-05-08T10:03:41Z"),
    ],
    narratives: [
      {
        trace_step_id: "trace-follow-4",
        reason_text: "这个问题问的是系统如何运作，而不是具体论文细节。",
        impact_text: "因此不用回读 PDF，也能清楚展示记忆优先路径。",
      },
    ],
  },
  "run-demo-rag-comparison": {
    steps: [
      buildStep("trace-rag-1", "run-demo-rag-comparison", "retrieve_session_memories", {
        planner_decision: {
          selected_tool: "retrieve_session_memories",
          rationale: "对比时先复用刚导入论文的记忆。",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-paper-2401"],
        coverage_score: 0.88,
        matched_query_terms: ["rag", "workflow", "compare"],
      }, "2026-05-08T10:06:12Z", "2026-05-08T10:06:18Z"),
      buildStep("trace-rag-2", "run-demo-rag-comparison", "retrieve_global_memories", {
        planner_decision: {
          selected_tool: "retrieve_global_memories",
          rationale: "再补一条关于记忆优先路由的全局经验。",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-global-1"],
        coverage_score: 0.57,
        matched_query_terms: ["memory-first", "retrieval"],
      }, "2026-05-08T10:06:19Z", "2026-05-08T10:06:23Z"),
      buildStep("trace-rag-3", "run-demo-rag-comparison", "rerank_context_candidates", {
        planner_decision: {
          selected_tool: "rerank_context_candidates",
          rationale: "这个对比只需要长期记忆，不需要回读原文。",
          fallback_used: false,
        },
      }, {
        selected_memory_ids: ["memory-paper-2401", "memory-global-1"],
        selection_source: "comparison_memory_pack",
        fallback_used: false,
      }, "2026-05-08T10:06:24Z", "2026-05-08T10:06:27Z"),
      buildStep("trace-rag-4", "run-demo-rag-comparison", "direct_final_answer", {
        retrieval_skipped: true,
        planner_decision: {
          selected_tool: "direct_final_answer",
          rationale: "这是概念性回答，仅靠记忆证据就能完成。",
          fallback_used: false,
        },
      }, {
        answer_preview: "标准 RAG 默认强制检索；这个 Agent 会先判断是否真的需要检索。",
      }, "2026-05-08T10:06:28Z", "2026-05-08T10:06:33Z"),
    ],
    narratives: [
      {
        trace_step_id: "trace-rag-4",
        reason_text: "这里比较的是编排策略，而不是缺失的原文事实。",
        impact_text: "所以回答可以保持简短，同时仍体现记忆驱动决策。",
      },
    ],
  },
  "run-demo-memory-snapshot": {
    steps: [
      buildStep("trace-snapshot-1", "run-demo-memory-snapshot", "retrieve_session_memories", {
        planner_decision: {
          selected_tool: "retrieve_session_memories",
          rationale: "先列出当前会话在导入后已经存下了什么。",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-paper-2401", "memory-open-2401"],
        coverage_score: 0.93,
        matched_query_terms: ["stored", "long-term memory", "import"],
      }, "2026-05-08T10:09:17Z", "2026-05-08T10:09:24Z"),
      buildStep("trace-snapshot-2", "run-demo-memory-snapshot", "retrieve_global_memories", {
        planner_decision: {
          selected_tool: "retrieve_global_memories",
          rationale: "顺便补充全局可复用记忆这一层。",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-global-1"],
        coverage_score: 0.44,
        matched_query_terms: ["global memory", "reusable lesson"],
      }, "2026-05-08T10:09:25Z", "2026-05-08T10:09:31Z"),
      buildStep("trace-snapshot-3", "run-demo-memory-snapshot", "direct_final_answer", {
        retrieval_skipped: true,
        planner_decision: {
          selected_tool: "direct_final_answer",
          rationale: "记忆清单本身已经结构化，不需要再回读原文。",
          fallback_used: false,
        },
      }, {
        answer_preview: "系统会存论文记忆、开放问题以及支撑它们的原文 chunk。",
      }, "2026-05-08T10:09:32Z", "2026-05-08T10:09:43Z"),
    ],
    narratives: [
      {
        trace_step_id: "trace-snapshot-3",
        reason_text: "memory drawer 本身就已经直接暴露了这些存储结构。",
        impact_text: "因此可以不重新打开原文，也能解释持久化层存了什么。",
      },
    ],
  },
};

const allTimelineEvents: Array<TimelineEvent & { visibleAt: number }> = [
  buildEvent("evt-1", "run-demo-search-import-answer", "run_started", "问答已开始：先搜索候选论文，再决定导入。", 1, [], []),
  buildEvent("evt-2", "run-demo-search-import-answer", "step_completed", "search_arxiv 返回了 3 篇候选论文。", 2, [], []),
  buildEvent("evt-3", "run-demo-search-import-answer", "step_completed", "import_arxiv_paper 通过 ingest run 导入了 arXiv:2401.12345。", 2, [], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-4", "run-demo-search-import-answer", "assistant_message_committed", "导入后的论文已经可以作为会话记忆参与后续回答。", 2, ["memory-paper-2401", "memory-open-2401"], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-5", "run-demo-memory-followup", "run_started", "追问已开始：先查 session memory，再决定是否回读原文。", 3, [], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-6", "run-demo-memory-followup", "run_finished", "这轮回答直接由会话记忆和全局记忆完成。", 4, ["memory-paper-2401", "memory-global-1"], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-7", "run-demo-rag-comparison", "run_started", "对比问答开始：用长期记忆说明它和 RAG 的区别。", 5, [], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-8", "run-demo-rag-comparison", "run_finished", "这次对比回答没有回读原文。", 6, ["memory-paper-2401", "memory-global-1"], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-9", "run-demo-memory-snapshot", "run_started", "记忆说明问答开始：解释长期记忆层实际存了什么。", 7, [], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-10", "run-demo-memory-snapshot", "run_finished", "记忆清单回答已经基于现有存储结构完成。", 8, ["memory-paper-2401", "memory-open-2401", "memory-global-1"], ["paper:arxiv:2401.12345"]),
];

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
            "这篇论文提出了一种记忆路由研究 Agent：它不会固定走 retrieve-then-generate，而是根据问题决定直接用结构化记忆回答、继续检索记忆，还是回读原文片段。",
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
            "开放问题：当一个 session 累积了许多部分重叠的论文时，模型如何稳定地做出记忆路由决策，并避免依赖过时摘要？",
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
          excerpt:
            "我们用模型驱动的路由决策，替代了固定的检索梯子，让系统在 session memory、global memory 和原文回读之间自主选择。",
        },
        {
          chunk_id: "chunk-2401-31",
          paper_id: "paper:arxiv:2401.12345",
          page: 8,
          section: "Limitations",
          excerpt:
            "当 session 中堆积越来越多重叠摘要和未解决问题时，路由质量可能会下降。",
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
            "全局经验：科学研究型 Agent 如果先检查 session memory，再决定是否回读原文，回答通常会更稳定。",
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
          excerpt: "记忆优先路由可以减少不必要的原文回读，也让后续回答更可审计。",
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

const playbackScheduleMs = [700, 2300, 4200, 5800, 7600, 9300, 11100, 12900];
const listeners = new Set<() => void>();
let visibleMessageCount = 0;
let playbackStarted = false;

function buildStep(
  id: string,
  runId: string,
  action: string,
  inputPayload: Record<string, unknown>,
  resultPayload: Record<string, unknown>,
  startedAt: string,
  finishedAt: string,
): TraceStep {
  return {
    id,
    run_id: runId,
    action,
    input_payload: inputPayload,
    result_payload: resultPayload,
    status: "completed",
    started_at: startedAt,
    finished_at: finishedAt,
  };
}

function buildEvent(
  id: string,
  runId: string,
  eventType: string,
  summary: string,
  visibleAt: number,
  relatedMemoryIds: string[],
  relatedPaperIds: string[],
): TimelineEvent & { visibleAt: number } {
  return {
    id,
    session_id: SESSION_ID,
    run_id: runId,
    event_type: eventType,
    summary,
    related_memory_ids: relatedMemoryIds,
    related_paper_ids: relatedPaperIds,
    created_at: new Date(visibleAt * 1000).toISOString(),
    visibleAt,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
    },
  });
}

function readOnlyResponse(): Response {
  return jsonResponse({ detail: "当前是静态演示，只读不可写。真实交互请使用 Docker 部署版本。" }, 405);
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function notify(): void {
  for (const listener of listeners) {
    listener();
  }
}

function startPlayback(): void {
  if (playbackStarted || typeof window === "undefined") {
    return;
  }
  playbackStarted = true;
  playbackScheduleMs.forEach((delay, index) => {
    window.setTimeout(() => {
      visibleMessageCount = Math.max(visibleMessageCount, index + 1);
      notify();
    }, delay);
  });
}

function getVisibleMessages(): Message[] {
  return allMessages.slice(0, visibleMessageCount);
}

function getVisibleRuns(): TaskRun[] {
  return [...runDefinitions]
    .filter((run) => visibleMessageCount >= run.userVisibleAt)
    .map((run) => {
      const finished = visibleMessageCount >= run.assistantVisibleAt;
      const progress = Math.max(1, visibleMessageCount - run.userVisibleAt + 1);
      return {
        id: run.id,
        session_id: SESSION_ID,
        message_id: run.messageId,
        status: finished ? "finished" : "running",
        step_count: finished ? run.stepCount : Math.min(run.stepCount - 1, progress),
        started_at: run.startedAt,
        finished_at: finished ? run.finishedAt : null,
        finish_reason: finished ? "completed" : null,
      };
    })
    .sort((left, right) => right.started_at.localeCompare(left.started_at));
}

function getVisibleTimeline(): TimelineEvent[] {
  return allTimelineEvents
    .filter((event) => visibleMessageCount >= event.visibleAt)
    .map(({ visibleAt: _visibleAt, ...event }) => event)
    .sort((left, right) => left.created_at.localeCompare(right.created_at));
}

function getVisibleTrace(runId: string): DemoTrace {
  const definition = runDefinitions.find((run) => run.id === runId);
  const trace = fullTraces[runId];
  if (!definition || !trace) {
    return { steps: [], narratives: [] };
  }
  if (visibleMessageCount >= definition.assistantVisibleAt) {
    return trace;
  }
  if (visibleMessageCount < definition.userVisibleAt) {
    return { steps: [], narratives: [] };
  }
  const visibleStepCount = Math.max(1, Math.min(trace.steps.length - 1, visibleMessageCount - definition.userVisibleAt + 1));
  return {
    steps: trace.steps.slice(0, visibleStepCount),
    narratives: trace.narratives,
  };
}

function match(pathname: string, pattern: RegExp): RegExpMatchArray | null {
  return pathname.match(pattern);
}

function handleApiRequest(method: string, pathname: string): Response | null {
  if (method !== "GET") {
    return readOnlyResponse();
  }

  if (pathname === "/api/sessions") {
    return jsonResponse({ items: sessions });
  }
  if (pathname === `/api/sessions/${SESSION_ID}/messages`) {
    return jsonResponse({ items: getVisibleMessages() });
  }
  if (pathname === `/api/sessions/${SESSION_ID}/runs`) {
    return jsonResponse({ items: getVisibleRuns() });
  }
  if (pathname === `/api/sessions/${SESSION_ID}/timeline`) {
    return jsonResponse({ items: getVisibleTimeline() });
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
    return jsonResponse(getVisibleTrace(traceMatch[1]));
  }

  const eventsMatch = match(pathname, new RegExp(`^/api/sessions/${SESSION_ID}/runs/([^/]+)/events$`));
  if (eventsMatch) {
    return jsonResponse({
      items: getVisibleTimeline().filter((event) => event.run_id === eventsMatch[1]),
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

  visibleMessageCount = 0;
  (globalThis as { __RESEARCH_AGENT_DEMO__?: { subscribe: typeof subscribe } }).__RESEARCH_AGENT_DEMO__ = {
    subscribe,
  };

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

  startPlayback();
}
