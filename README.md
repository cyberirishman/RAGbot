# RAG
Build a RAG from start - from a virgin MacOS silicon install so we cover ALL dependancies
RAG Chatbot — Local Buildout Plan & Lesson Guide (16GB Teaching Edition)
Machine target: Apple Silicon Mac with 16GB+ RAM (Mac Mini M-series, MacBook Air M-series, or similar) | macOS 14+
User: any user with admin (sudo) rights — does not need to be root


This project is designed to be *idiotproof* as much as I am capable of
The project assumes NO dependancies are already installed on your Mac silicon amchine
We go step by step installing everything and explaining why
Brew, Git , Python , a venv virtual environment , all dependancies using pip to install a requirements.txt , Ollama , the required models...everything! 



Goal: Build a locally-hosted RAG chatbot that can ingest PDF, Word, Markdown, and text files 
— including understanding graphs and charts 
— and answer questions through a browser-based chat interface,
on consumer-class Apple Silicon hardware (16GB RAM minimum)

## The only document you absolutely need to download is the RAGbot_instructions.md file.
## ALL step by instructions and line of code is inside this single document 
## The completed phython / config files are also in this repo but for best learning experience
## GO LINE BY LINE from the RAGbot.md file

## How to Read The RAGbot_instructions.md Document

Every step follows this format:

```
command to run
```
> **What it does:** Plain-language explanation of the command.  
> **Why we need it:** How it fits into the chatbot system.

Read the explanation before running the command. Understanding *why* matters more than just copy-pasting.

---

## Architecture Overview (Before We Start)

Our chatbot is made up of six layers that work together like an assembly line:

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

** 2-model design :
** `gemma4:e4b` is Google's "effective 4 billion parameter" multimodal model — it natively understands both text and images. This means it handles image description during ingest AND generates chat answers from a single loaded model, eliminating the need for a separate vision model entirely.
** `nomic-embed-text` is the dedicated embedding model — purpose-built for semantic search and far more accurate for RAG retrieval than a general LLM. Stated capabilities of Gemma 4 e4b: vision, tools, thinking, audio, and cloud-deployable.

**Vision quality disclosure:** The diagram and chart description quality of `gemma4:e4b` is meaningfully lower than the `gemma4:26b` model used in v1 of this project. For a teaching PoC and most text-heavy field-service Q&A, this trade-off is acceptable. If your end use case depends on rich diagram/chart understanding (e.g. parts-explosion drawings, schematic reading).

**The key concept — RAG (Retrieval-Augmented Generation):**  
Instead of asking the LLM a question cold (where it might guess or hallucinate), we first *retrieve* the most relevant passages from your documents, then *augment* the prompt with that context, then *generate* a grounded answer. The LLM only speaks from evidence it was handed.

**RAM footprint (16GB Apple Silicon Mac):**

| Model | Role | RAM (approx, during active use) |
|---|---|---|
| `nomic-embed-text` | Embeddings | ~0.4 GB |
| `gemma4:e4b` | Vision (ingest) + Chat (queries) | ~10–12 GB |
| System + ChromaDB + Python overhead | — | ~3–4 GB |
| **Total** | | **~14–16 GB of 16 GB** |

Tight but workable on a 16GB machine, comfortable on 24GB+. Two practical recommendations on 16GB: close other RAM-hungry apps (browsers with many tabs, IDEs) before running ingest, and consider lowering `OLLAMA_KEEP_ALIVE` from `24h` to `5m` if you do other work between chatbot sessions — the model will reload more slowly afterwards but RAM is reclaimed when idle.
