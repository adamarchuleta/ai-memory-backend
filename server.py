

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


APP_TITLE = "AI Memory Backend"
DB_PATH = Path(os.getenv("MEMORY_DB_PATH", "./memory.db"))
MAX_RESULTS = 20


app = FastAPI(title=APP_TITLE)


# ---------- Data models ----------
class MemoryMetadata(BaseModel):
    source: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class SaveMemoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    topic: Optional[str] = None
    importance: int = Field(default=3, ge=1, le=5)
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)


class SaveMemoryResponse(BaseModel):
    memory_id: str
    saved: bool
    topic: str
    created_at: str


class SearchMemoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    topic: Optional[str] = None
    limit: int = Field(default=5, ge=1, le=MAX_RESULTS)


class RecallMemoryRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    topic: Optional[str] = None
    limit: int = Field(default=5, ge=1, le=MAX_RESULTS)


class MemoryRecord(BaseModel):
    id: str
    user_id: str
    topic: str
    content: str
    importance: int
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    score: Optional[float] = None


class MemoryListResponse(BaseModel):
    memories: list[MemoryRecord]
    count: int


class ChatContextRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    topic: Optional[str] = None
    limit: int = Field(default=5, ge=1, le=MAX_RESULTS)


class ChatContextResponse(BaseModel):
    inferred_topic: str
    memories: list[MemoryRecord]
    context_block: str


@dataclass
class SearchTokenSet:
    query_tokens: set[str]
    content_tokens: set[str]
    topic_tokens: set[str]


# ---------- Database ----------
def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(get_connection()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                importance INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_topic ON memories(user_id, topic)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user_updated ON memories(user_id, updated_at DESC)"
        )
        conn.commit()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


# ---------- Helpers ----------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_topic(value: Optional[str], *, fallback_text: Optional[str] = None) -> str:
    if value:
        cleaned = " ".join(value.strip().lower().split())
        if cleaned:
            return cleaned.replace(" ", "-")

    if fallback_text:
        guess = infer_topic_from_text(fallback_text)
        if guess:
            return guess

    return "general"


def infer_topic_from_text(text: str) -> str:
    lowered = text.lower()

    topic_keywords = {
        "work": ["meeting", "project", "deadline", "client", "manager", "job"],
        "personal": ["birthday", "family", "friend", "favorite", "love", "home"],
        "health": ["doctor", "medication", "workout", "sleep", "diet"],
        "travel": ["flight", "hotel", "trip", "travel", "airport", "vacation"],
        "preferences": ["prefer", "favorite", "likes", "dislikes", "usually"],
    }

    for topic, keywords in topic_keywords.items():
        if any(keyword in lowered for keyword in keywords):
            return topic

    words = [w for w in tokenize(text) if len(w) > 3]
    return words[0] if words else "general"


def tokenize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in cleaned.split() if token]


def score_memory(query: str, memory_row: sqlite3.Row) -> float:
    token_set = SearchTokenSet(
        query_tokens=set(tokenize(query)),
        content_tokens=set(tokenize(memory_row["content"])),
        topic_tokens=set(tokenize(memory_row["topic"])),
    )

    if not token_set.query_tokens:
        return float(memory_row["importance"])

    overlap_content = len(token_set.query_tokens & token_set.content_tokens)
    overlap_topic = len(token_set.query_tokens & token_set.topic_tokens)
    importance_bonus = float(memory_row["importance"]) * 0.35
    exact_phrase_bonus = 2.0 if query.lower() in memory_row["content"].lower() else 0.0

    return (overlap_content * 2.0) + (overlap_topic * 1.5) + importance_bonus + exact_phrase_bonus


def row_to_memory(row: sqlite3.Row, score: Optional[float] = None) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        user_id=row["user_id"],
        topic=row["topic"],
        content=row["content"],
        importance=row["importance"],
        metadata=json.loads(row["metadata_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        score=score,
    )


def format_context_block(memories: list[MemoryRecord]) -> str:
    if not memories:
        return "No relevant memories found."

    lines = ["Relevant memories:"]
    for memory in memories:
        lines.append(
            f"- [{memory.topic}] {memory.content}"
        )
    return "\n".join(lines)


# ---------- Routes ----------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": APP_TITLE}


@app.post("/memory/save", response_model=SaveMemoryResponse)
def save_memory(payload: SaveMemoryRequest) -> SaveMemoryResponse:
    memory_id = str(uuid4())
    now = utc_now_iso()
    topic = normalize_topic(payload.topic, fallback_text=payload.content)

    with closing(get_connection()) as conn:
        conn.execute(
            """
            INSERT INTO memories (
                id, user_id, topic, content, importance, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                payload.user_id,
                topic,
                payload.content.strip(),
                payload.importance,
                payload.metadata.model_dump_json(),
                now,
                now,
            ),
        )
        conn.commit()

    return SaveMemoryResponse(
        memory_id=memory_id,
        saved=True,
        topic=topic,
        created_at=now,
    )


@app.post("/memory/recall", response_model=MemoryListResponse)
def recall_memories(payload: RecallMemoryRequest) -> MemoryListResponse:
    topic = normalize_topic(payload.topic) if payload.topic else None

    with closing(get_connection()) as conn:
        if topic:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND topic = ?
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (payload.user_id, topic, payload.limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ?
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (payload.user_id, payload.limit),
            ).fetchall()

    memories = [row_to_memory(row) for row in rows]
    return MemoryListResponse(memories=memories, count=len(memories))


@app.post("/memory/search", response_model=MemoryListResponse)
def search_memories(payload: SearchMemoryRequest) -> MemoryListResponse:
    topic = normalize_topic(payload.topic) if payload.topic else None

    with closing(get_connection()) as conn:
        if topic:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? AND topic = ?",
                (payload.user_id, topic),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id = ?",
                (payload.user_id,),
            ).fetchall()

    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        score = score_memory(payload.query, row)
        if score > 0:
            scored.append((score, row))

    scored.sort(key=lambda item: (item[0], item[1]["importance"], item[1]["updated_at"]), reverse=True)
    top_rows = scored[: payload.limit]
    memories = [row_to_memory(row, score=score) for score, row in top_rows]
    return MemoryListResponse(memories=memories, count=len(memories))


@app.post("/chat/context", response_model=ChatContextResponse)
def build_chat_context(payload: ChatContextRequest) -> ChatContextResponse:
    inferred_topic = normalize_topic(payload.topic, fallback_text=payload.message)

    search_payload = SearchMemoryRequest(
        user_id=payload.user_id,
        query=payload.message,
        topic=inferred_topic,
        limit=payload.limit,
    )
    search_result = search_memories(search_payload)

    if not search_result.memories:
        recall_result = recall_memories(
            RecallMemoryRequest(
                user_id=payload.user_id,
                topic=inferred_topic,
                limit=payload.limit,
            )
        )
        memories = recall_result.memories
    else:
        memories = search_result.memories

    return ChatContextResponse(
        inferred_topic=inferred_topic,
        memories=memories,
        context_block=format_context_block(memories),
    )


@app.get("/memory/{memory_id}", response_model=MemoryRecord)
def get_memory(memory_id: str) -> MemoryRecord:
    with closing(get_connection()) as conn:
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")

    return row_to_memory(row)