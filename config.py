# =============================================================================
# config.py — Central configuration for the RAG Chatbot
# =============================================================================
# Every setting the chatbot uses is defined here.
# To change a model, path, or behaviour: edit this file ONLY.
# Never hardcode these values inside src/ scripts.
# =============================================================================

import os

# --- Ollama Connection ---
# The address where Ollama is listening.
# localhost means "on this same machine" — explicit, portable, works on macOS and Linux.
OLLAMA_BASE_URL = "http://0.0.0.0:11434"

# --- Ollama Model Keep-Alive ---
# How long Ollama keeps a model resident in RAM after its last use.
# Default is 5 minutes — too short when alternating between vision and embedding
# models during a multi-document ingest. "24h" keeps all models loaded for the
# full working day, preventing slow reload delays between documents.
# Override at runtime: export OLLAMA_KEEP_ALIVE=5m  (to reduce memory use)
OLLAMA_KEEP_ALIVE = "24h"

# --- Model Names ---
# The exact model name as it appears in `ollama list`
# 2-model design (v2.0, inherited from v1.4 architectural decision):
# nomic-embed-text + gemma4:e4b. The chat/vision model is the e4b variant
# of Gemma 4 (~10GB on disk, "effective 4 billion parameters"), chosen so
# the project runs comfortably on a 16GB Mac. It is multimodal — handles
# both image description during ingest AND chat responses, so we still
# need only two models loaded.
EMBED_MODEL    = "nomic-embed-text"    # Dedicated embedding model — converts text to vectors.
                                        # Purpose-built for semantic search; far more accurate
                                        # for RAG retrieval than using a general LLM as embedder.
CHAT_MODEL     = "gemma4:e4b"          # Multimodal LLM — generates chat answers AND describes
                                        # images/charts/diagrams during document ingestion.
VISION_MODEL   = CHAT_MODEL            # Alias — same model used for image description at ingest.
                                        # Kept as a separate config key so the code is self-
                                        # documenting and easy to split back if ever needed.

# --- File Paths ---
# os.path.dirname(__file__) gets the folder this config.py file lives in.
# This makes all paths relative to the project root — works regardless of where
# you run the script from.
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
INGEST_DIR     = os.path.join(PROJECT_ROOT, "ingest")     # Drop documents here to ingest
CHROMA_DIR     = os.path.join(PROJECT_ROOT, "chroma_db")  # ChromaDB writes its data here

# --- ChromaDB Settings ---
CHROMA_COLLECTION = "enterprise_kb"  # Name of the collection (like a table in a database)

# --- RAG Chunking Settings ---
# When we ingest a document, we split it into smaller pieces called "chunks".
# The LLM has a limited context window — it can't read a 200-page manual all at once.
# Instead, we retrieve only the most relevant chunks and feed those to the LLM.
CHUNK_SIZE     = 1800   # Maximum characters per chunk — raised from 1000 (U-17).
                        # nomic-embed-text supports ~30k chars per input, so 1000 was
                        # far below capacity and fragmented multi-step procedures and
                        # spec tables unnecessarily. 1800 gives richer, more complete chunks
                        # while still fitting well within the embedding model's window.
CHUNK_OVERLAP  = 200    # Overlap raised proportionally from 150 (U-17).
                        # Prevents answers from being cut off at chunk boundaries.

# --- Retrieval Settings ---
RETRIEVAL_TOP_K = 5     # How many chunks to retrieve per query
                        # Higher = more context for the LLM, but slower and may dilute relevance

# --- Conversation Memory Settings ---
# How many of the most recent messages from the chat history are injected
# into the prompt so the LLM can resolve follow-up questions and pronouns
# (e.g. "how many are black?" right after "tell me about the inventory").
# 6 = roughly the last 3 exchanges (user + assistant pairs).
#
# Privacy: this history exists only inside the user's browser tab.
# The server (Python process, Ollama, ChromaDB) never persists it between
# sessions — closing the tab discards the conversation entirely.
#
# Referenced by src/chat.py format_history() — without this constant,
# follow-up queries crash with AttributeError. Was missing from the
# canonical heredoc through v2.4 (back-ported in v2.5 — see U-30).
HISTORY_TURNS = 6

# --- Vision Model Context Window ---
# How many characters of surrounding document text to send to the vision model
# alongside an image, to help it interpret the picture in context.
# Increase this if descriptions seem to lack document context; decrease to speed up calls.
VISION_CONTEXT_CHARS = 500

# --- Gradio UI Settings ---
GRADIO_HOST    = "0.0.0.0"  # Makes the UI accessible from any device on the local network
GRADIO_PORT    = 7860        # The port the Gradio web interface listens on
                             # Access it at http://localhost:7860 in your browser

# --- Supported File Extensions ---
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".md", ".txt"]
