import type {
  IngestExecution,
  MemorySnapshot,
  Message,
  MessageSubmission,
  QueryExecution,
  RuntimeStatus,
  Session,
  TaskRun,
  TimelineEvent,
  TraceNarrative,
  TraceStep,
} from "./types";

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail ?? detail;
    } catch {
      // Ignore JSON parsing failures and fall back to status text.
    }
    throw new Error(detail || `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    headers: {
      Accept: "application/json",
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
    ...init,
  });
  return readJson<T>(response);
}

export const api = {
  listSessions: () => request<{ items: Session[] }>("/api/sessions"),
  createSession: (title: string) =>
    request<Session>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ title }),
    }),
  deleteSession: (sessionId: string) =>
    request(`/api/sessions/${sessionId}`, {
      method: "DELETE",
    }),
  listMessages: (sessionId: string) => request<{ items: Message[] }>(`/api/sessions/${sessionId}/messages`),
  listRuns: (sessionId: string) => request<{ items: TaskRun[] }>(`/api/sessions/${sessionId}/runs`),
  listTimeline: (sessionId: string) => request<{ items: TimelineEvent[] }>(`/api/sessions/${sessionId}/timeline`),
  getMemorySnapshot: (sessionId: string) =>
    request<MemorySnapshot>(`/api/sessions/${sessionId}/memory-snapshot`),
  getTrace: (sessionId: string, runId: string) =>
    request<{ steps: TraceStep[]; narratives: TraceNarrative[] }>(`/api/sessions/${sessionId}/runs/${runId}/trace`),
  getRunEvents: (sessionId: string, runId: string) =>
    request<{ items: TimelineEvent[] }>(`/api/sessions/${sessionId}/runs/${runId}/events`),
  getRuntimeStatus: () => request<RuntimeStatus>("/api/system/runtime"),
  submitText: (sessionId: string, text: string) =>
    request<MessageSubmission>(`/api/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
  submitArxiv: (sessionId: string, arxivUrl: string) =>
    request<MessageSubmission>(`/api/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ arxiv_url: arxivUrl }),
    }),
  startQueryRun: (sessionId: string, runId: string) =>
    request<TaskRun>(`/api/sessions/${sessionId}/queries/${runId}/start`, {
      method: "POST",
    }),
  startIngestRun: (sessionId: string, runId: string) =>
    request<TaskRun>(`/api/sessions/${sessionId}/ingest/${runId}/start`, {
      method: "POST",
    }),
  uploadPdf: async (sessionId: string, file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<MessageSubmission>(`/api/sessions/${sessionId}/uploads/pdf`, {
      method: "POST",
      body: formData,
    });
  },
  executeQuery: (sessionId: string, runId: string) =>
    request<QueryExecution>(`/api/sessions/${sessionId}/queries/${runId}/execute`, {
      method: "POST",
    }),
  executeIngest: (sessionId: string, runId: string) =>
    request<IngestExecution>(`/api/sessions/${sessionId}/ingest/${runId}/execute`, {
      method: "POST",
    }),
  deleteMemory: (sessionId: string, memoryKind: string, memoryId: string) =>
    request(`/api/sessions/${sessionId}/memories/${memoryKind}/${memoryId}`, {
      method: "DELETE",
    }),
};
