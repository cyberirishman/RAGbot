
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


```
Your Documents (PDF, DOCX, MD, TXT)
        ↓
[ Docling ] ← extracts text, tables, and images
        ↓
[ gemma4:e4b ] ← reads charts/graphs, writes plain-English descriptions
        |          (multimodal — handles images AND text natively)
        ↓
[ nomic-embed-text ] ← converts all text into numbers (embeddings)
        ↓
[ ChromaDB ] ← stores and searches those numbers (vector database)
        ↓
[ LangChain ] ← when you ask a question, finds the right chunks and asks the LLM
        ↓
[ gemma4:e4b via Ollama ] ← reads the retrieved context, writes the answer
        ↓
[ Gradio ] ← the browser chat window you type into
```

