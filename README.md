# KaStack3 - Conversation Memory Backend

KaStack3 is a FastAPI backend for an AI assistant memory pipeline. It ingests conversation data, creates chronological checkpoints, stores/retrieves memory, routes questions through the right RAG strategy, and supports encrypted day-wise backup storage.

## Hackathon Scope

The project is built around the required stack:

- API: FastAPI
- DB/Auth: Supabase
- Backup Storage: Cloudflare R2/S3-compatible storage
- Vector DB: Qdrant
- Observability: Comet Opik
- Encryption: encrypted backup payloads derived from user credentials

## Current Modules

- `app/main.py`
  - Main FastAPI app.
  - Provides `GET /health`, `POST /upload`, and `POST /ask`.
  - Builds checkpoints, encrypts backup payloads, optionally syncs to Supabase/R2, and returns routed answers.

- `app/retrieval/`
  - Yashas's RAG retrieval layer.
  - Includes semantic search, keyword search, hybrid merge logic, Qdrant vector adapter, event-memory retrieval, and route decision logic.

- `checkpointing.py`
  - Builds topic checkpoints, 100-message checkpoints, day-wise checkpoint metadata, and mood checkpoints.
  - Stores checkpoint rows in Supabase tables.

- `schema.sql`
  - Supabase schema for topic, message, day, mood, event memory, and query log tables.
  - Includes row-level security policies.

- `run_server.py`
  - Simple server runner for the FastAPI app.

- `test_live_api.py`
  - Small live API smoke test for `/upload`.

## Python Version

Use Python 3.11 or 3.12.

The pinned team dependency set uses `numpy==1.26.4`, which can fail on Python 3.13.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and fill credentials as needed.

## Environment Variables

Core retrieval/Qdrant:

```bash
USE_SENTENCE_TRANSFORMERS=1
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=conversation_memory
```

Supabase:

```bash
SUPABASE_URL=
SUPABASE_KEY=
SUPABASE_ANON_KEY=
```

Cloudflare R2/S3-compatible backup storage:

```bash
R2_BUCKET=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=auto
```

Opik:

```bash
OPIK_API_KEY=
OPIK_WORKSPACE=
OPIK_PROJECT_NAME=kastack3-memory-backend
```

Dataset:

```bash
CSV_PATH=conversations.csv
```

## Run

Recommended:

```bash
uvicorn app.main:app --reload --port 8000
```

Alternative:

```bash
python run_server.py
```

Health check:

```bash
GET /health
```

## API

### POST `/upload`

Ingests a conversation, creates checkpoints, extracts events, encrypts backup payloads, and optionally syncs to Supabase/R2.

Example:

```json
{
  "user_id": "demo-user",
  "password": "demo-pass",
  "conversation": [
    {
      "role": "user",
      "content": "I had a great day at work and I am excited about the project."
    },
    {
      "role": "assistant",
      "content": "That sounds wonderful."
    },
    {
      "role": "user",
      "content": "I need to buy groceries later this evening."
    }
  ]
}
```

Returns checkpoint counts, encrypted backup round-trip status, and backup storage info.

### POST `/ask`

Routes a question through keyword, semantic, or hybrid retrieval and returns an answer with sources.

Example:

```json
{
  "user_id": "demo-user",
  "password": "demo-pass",
  "question": "What did the user say about groceries?",
  "top_k": 3
}
```

Returns:

- `answer`
- `retrieval_path`
- `reason`
- `sources`

## Checkpoint Design

- Topic checkpoints: split conversations by topic-like segments and summarize each segment.
- 100-message checkpoints: summarize every 100-message chunk.
- Day-wise checkpoints: group messages by day, summarize, encrypt, and prepare for R2 backup.
- Mood checkpoints: assign mood label and score per segment/day.
- Events memory: extract discrete facts/events such as grocery mentions or mood drops.

## RAG Router Logic

The retrieval layer supports:

- Keyword search for exact terms, names, lists, places, and explicit mentions.
- Semantic search for meaning, preferences, mood, summaries, and inference-style questions.
- Hybrid search for ambiguous questions or when both exact and semantic signals are useful.

The router logs the selected path and reason so Opik can trace retrieval decisions.

## Supabase Setup

Run `schema.sql` in Supabase SQL editor to create:

- `topic_checkpoints`
- `message_checkpoints`
- `day_checkpoints`
- `mood_checkpoints`
- `events_memory`
- `query_logs`

The schema includes RLS policies using Supabase Auth user IDs.

## Demo Checklist

- Run `/upload` with a sample conversation.
- Ask one keyword question.
- Ask one semantic question.
- Ask one ambiguous question.
- Show route decision and sources in `/ask` response.
- Show encrypted backup round-trip status from `/upload`.
- Show Supabase tables and Opik traces if credentials are configured.
