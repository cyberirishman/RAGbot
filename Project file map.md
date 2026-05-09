
## Project File System and Process Maps

```
Chatbotv2/
├── chatbot-16G-venv/            ← Python virtual environment (never edit manually)
├── ingest/                      ← DROP DOCUMENTS HERE to add to the knowledge base
│   └── processed/               ← Documents moved here automatically after ingestion
├── chroma_db/                   ← ChromaDB data (auto-managed, never edit manually)
├── logs/                        ← Log files written here automatically
│   └── chatbot.log              ← Main log (rotates at 5MB, keeps 5 backups)
├── src/
│   ├── __init__.py              ← Package marker — REQUIRED for imports to work (0 bytes)
│   ├── ingest.py                ← Run to load documents into ChromaDB
│   ├── chat.py                  ← RAG query engine (imported by app.py)
│   ├── app.py                   ← Run to start the browser chat interface
│   └── logging_setup.py         ← Shared logging config (used by all scripts)
├── config.py                    ← ALL settings live here — the only file admins need to edit
└── requirements.txt             ← Full dependency list — every package pinned to a known-good version
```

## Shared Infrastructure
```
SHARED INFRASTRUCTURE
                  (used by both pipelines above)

  ┌─────────────────────────────────────────────────────────────────┐
  │  Ollama  (http://0.0.0.0:11434)                                 │
  │  one local HTTP service that loads and serves both AI models:   │
  │   • nomic-embed-text  ← embeddings  (text → vector)             │
  │   • gemma4:e4b        ← chat answers  AND  image description    │
  │  models stay in RAM for OLLAMA_KEEP_ALIVE (default "24h")       │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  ChromaDB  (folder: chroma_db/, embedded mode — no separate     │
  │  server process). Built once by ingest.py, queried every time   │
  │  the user asks a question. Persisted to disk between sessions   │
  │  so the knowledge base survives reboots.                        │
  └─────────────────────────────────────────────────────────────────┘
```


## Ingest pipeline
```
INGEST PIPELINE
              (runs once per document — populates ChromaDB)

  Your Documents
  (PDF, DOCX,
   MD, TXT)
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Docling                                                │
  │  parses each file → markdown text + tables + image list │
  └─────────────────────────────────────────────────────────┘
        │
        │ (text path)                       (image path)
        ▼                                          ▼
  ┌──────────────────────┐         ┌────────────────────────────┐
  │ Two-stage chunker    │         │ gemma4:e4b (vision)        │
  │  1. split on H1/H2/  │         │ "describe this picture"    │
  │     H3 markdown      │         │ (multimodal — same model   │
  │     headings         │         │  used for chat answers)    │
  │  2. split each       │         └────────────────────────────┘
  │     section into     │                       │
  │     1800-char        │                       │ (description text
  │     overlapping      │                       │  becomes its own chunk)
  │     pieces           │                       │
  └──────────────────────┘                       │
        │                                        │
        ├────────────────────────────────────────┘
        ▼
  ┌────────────────────────────────────────────────┐
  │ nomic-embed-text  (embedder)                   │
  │ each chunk → 768-dimensional vector            │
  │ (a numeric "fingerprint" of its meaning)       │
  └────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────┐
  │ ChromaDB  (vector database — on disk)          │
  │ stores vectors + chunk text + metadata         │
  │ collection name: "enterprise_kb"               │
  └────────────────────────────────────────────────┘
        │
        ▼
  Source file moved to ingest/processed/
  (so re-running ingest.py never duplicates chunks)
```


## Query Pipleine

```
QUERY PIPELINE
            (runs once per user message — answers from ChromaDB)

  ┌──────────────────────┐
  │  Gradio (browser)    │  ← user types:
  │  http://localhost:   │     "what is the torque spec for the input shaft?"
  │   7860               │
  └──────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────┐
  │ src/app.py — chat_handler (Gradio generator)   │
  │ receives message + per-tab history             │
  └────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────┐
  │ src/chat.py — LCEL chain (LangChain)           │
  │ orchestrates: embed → retrieve → prompt → LLM  │
  └────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────┐
  │ nomic-embed-text  (embedder)                   │
  │ user's question → 768-dim vector               │
  └────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────┐
  │ ChromaDB similarity search                     │
  │ returns top-K (5) most relevant chunks         │
  │ from the knowledge base built by ingest        │
  └────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────┐
  │ Prompt assembly (LCEL)                         │
  │  • system instructions ("answer from context") │
  │  • last HISTORY_TURNS messages (conversation)  │
  │  • retrieved chunks (the grounded context)     │
  │  • the user's current question                 │
  └────────────────────────────────────────────────┘
        │
        ▼
  ┌────────────────────────────────────────────────┐
  │ gemma4:e4b  (chat)                             │
  │ reads the assembled prompt,                    │
  │ streams answer tokens back                     │
  └────────────────────────────────────────────────┘
        │
        ▼ (tokens stream back through LangChain → Gradio)
  ┌──────────────────────┐
  │  Gradio (browser)    │  ← answer appears word-by-word
  │                      │     in the chat bubble
  └──────────────────────┘

```
