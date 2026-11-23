"""Persistent conversation catalogue and LangGraph checkpoints in SQLite."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver


@dataclass(frozen=True)
class Conversation:
    """Metadata used to identify and display one conversation."""

    thread_id: str
    title: str
    created_at: str
    updated_at: str


class ConversationStore:
    """Manage conversation metadata using an open SQLite connection."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(self, title: str = "Nouvelle discussion") -> Conversation:
        """Create and immediately persist a conversation with a unique ID."""
        title = title.strip()
        if not title:
            raise ValueError("Conversation title must not be empty.")

        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        conversation = Conversation(str(uuid4()), title, now, now)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO conversations (thread_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    conversation.thread_id,
                    conversation.title,
                    conversation.created_at,
                    conversation.updated_at,
                ),
            )
        return conversation

    def list(self) -> list[Conversation]:
        """Return conversations ordered by their most recent activity."""
        rows = self.connection.execute(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC, thread_id ASC
            """
        ).fetchall()
        return [Conversation(*row) for row in rows]

    def get(self, thread_id: str) -> Conversation:
        """Return one conversation, raising KeyError when it does not exist."""
        row = self.connection.execute(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM conversations
            WHERE thread_id = ?
            """,
            (thread_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown conversation: {thread_id}")
        return Conversation(*row)

    def rename(self, thread_id: str, title: str) -> None:
        """Persist a new title and activity date for an existing conversation."""
        title = title.strip()
        if not title:
            raise ValueError("Conversation title must not be empty.")

        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self.connection:
            cursor = self.connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE thread_id = ?
                """,
                (title, now, thread_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown conversation: {thread_id}")

    def touch(self, thread_id: str) -> None:
        """Update activity after a turn, raising KeyError for an unknown ID."""
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE thread_id = ?",
                (now, thread_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown conversation: {thread_id}")


@contextmanager
def open_storage(
    database_path: str | Path = "runtime/conversations.sqlite",
) -> Iterator[tuple[ConversationStore, SqliteSaver]]:
    """Open file-backed storage; keep the context open while using the graph.

    The catalogue and the checkpointer use separate connections to the same
    file so that each component manages its own transactions. Catalogue calls
    belong to the thread that opened this context, such as the terminal loop.
    """
    if str(database_path) == ":memory:":
        raise ValueError("Persistent storage requires a file path.")
    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(path), timeout=5.0)
    try:
        store = ConversationStore(connection)
        with SqliteSaver.from_conn_string(str(path)) as checkpointer:
            checkpointer.setup()
            yield store, checkpointer
    finally:
        connection.close()
