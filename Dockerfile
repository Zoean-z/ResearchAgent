FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RESEARCH_AGENT_STORAGE_BACKEND=sqlite
ENV RESEARCH_AGENT_SQLITE_PATH=/app/data/sqlite/research_agent.sqlite3
ENV RESEARCH_AGENT_OPENVIKING_BACKEND=noop
ENV RESEARCH_AGENT_QUERY_AGENT_BACKEND=turn_adapter

WORKDIR /app

COPY backend/pyproject.toml /app/backend/pyproject.toml
COPY backend/src /app/backend/src
COPY backend/migrations /app/backend/migrations
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

RUN pip install --no-cache-dir /app/backend

RUN mkdir -p /app/data/artifacts /app/data/sqlite

EXPOSE 8000

CMD ["uvicorn", "research_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
