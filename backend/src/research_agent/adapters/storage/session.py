"""In-memory session and message repositories."""

from __future__ import annotations

from collections import defaultdict

from research_agent.domain.models import Message, Session, SessionDocument, utc_now
from research_agent.domain.ports import MessageRepositoryPort, SessionRepositoryPort


class InMemorySessionRepository(SessionRepositoryPort):
    """Simple in-memory storage for sessions and session documents."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._documents_by_session: defaultdict[str, list[SessionDocument]] = defaultdict(list)

    def save(self, session: Session) -> Session:
        self._sessions[session.id] = session
        return session

    def get_by_id(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list_all(self) -> list[Session]:
        return sorted(self._sessions.values(), key=lambda session: session.created_at)

    def delete(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        deleted = session.model_copy(update={"status": "deleted", "updated_at": utc_now()})
        self._sessions[session_id] = deleted
        return deleted

    def save_document(self, session_document: SessionDocument) -> SessionDocument:
        documents = [
            document
            for document in self._documents_by_session[session_document.session_id]
            if document.id != session_document.id and document.paper_id != session_document.paper_id
        ]
        documents.append(session_document)
        documents.sort(key=lambda document: document.added_at)
        self._documents_by_session[session_document.session_id] = documents
        return self.get_document(session_document.session_id, session_document.paper_id) or session_document

    def list_documents(self, session_id: str) -> list[SessionDocument]:
        return list(self._documents_by_session.get(session_id, []))

    def list_all_documents(self) -> list[SessionDocument]:
        documents = [document for items in self._documents_by_session.values() for document in items]
        documents.sort(key=lambda document: document.added_at)
        return documents

    def get_document(self, session_id: str, paper_id: str) -> SessionDocument | None:
        return next((document for document in self._documents_by_session.get(session_id, []) if document.paper_id == paper_id), None)

    def delete_documents(self, session_id: str) -> int:
        documents = self._documents_by_session.pop(session_id, [])
        return len(documents)


class InMemoryMessageRepository(MessageRepositoryPort):
    """Simple in-memory storage for session messages."""

    def __init__(self) -> None:
        self._messages: dict[str, Message] = {}
        self._messages_by_session: defaultdict[str, list[Message]] = defaultdict(list)

    def save(self, message: Message) -> Message:
        self._messages[message.id] = message
        messages = [existing for existing in self._messages_by_session[message.session_id] if existing.id != message.id]
        messages.append(message)
        messages.sort(key=lambda item: item.created_at)
        self._messages_by_session[message.session_id] = messages
        return message

    def get_by_id(self, message_id: str) -> Message | None:
        return self._messages.get(message_id)

    def list_by_session(self, session_id: str) -> list[Message]:
        return list(self._messages_by_session.get(session_id, []))

    def delete_by_session(self, session_id: str) -> int:
        messages = self._messages_by_session.pop(session_id, [])
        for message in messages:
            self._messages.pop(message.id, None)
        return len(messages)
