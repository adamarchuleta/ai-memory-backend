

# 🧠 AI Memory Backend

A minimal, production-style backend for building AI systems with persistent memory.

This service lets you store, search, and recall user-specific memories — and generate structured context for LLMs.

---

## 🚀 Features

- Save structured user memories
- Search memories with relevance scoring
- Recall memories by topic
- Automatic topic inference
- Context generation for AI conversations
- Lightweight FastAPI + SQLite setup

---

## 📦 API Overview

- `POST /memory/save` — store a memory
- `POST /memory/search` — search memories by query
- `POST /memory/recall` — recall top memories
- `POST /chat/context` — build memory-aware context for AI
- `GET /memory/{memory_id}` — fetch a memory
- `GET /health` — health check

---

## ⚡ Quick Start

```bash
git clone <your-repo>
cd ai-memory-backend

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
uvicorn server:app --reload
```

Open:
http://127.0.0.1:8000/docs

---

## 🧠 Example

Save a memory:

```json
{
  "user_id": "user_123",
  "content": "My favorite food is sushi",
  "importance": 4
}
```

Search memories:

```json
{
  "user_id": "user_123",
  "query": "what do I like to eat?"
}
```

---

## 🎯 Use Cases

- AI assistants with long-term memory
- Personal AI companions
- Memory-augmented chatbots
- RAG-style systems without heavy infrastructure
- Prototyping AI memory systems quickly

---

## 🧩 Notes

- Uses SQLite by default (easy local setup)
- Can be swapped for Postgres / vector DB later
- Designed to be simple, hackable, and extensible

---

## ⭐️

If this helped you, consider starring the repo!