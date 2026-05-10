"""
chat.py — RAG Query Engine
============================
This module handles the retrieval-augmented generation (RAG) pipeline.
It is imported by app.py (the Gradio UI) and handles:
  1. Embedding the user's question using nomic-embed-text
  2. Searching ChromaDB for the most relevant document chunks
  3. Building a context-rich prompt using LCEL (LangChain Expression Language)
  4. Calling gemma4:e4b via Ollama to generate a grounded answer
  5. Returning the answer and the sources it used — streaming token-by-token
     (via query_stream) or all-at-once (via query)

v1.3 changes:
  - Replaced deprecated RetrievalQA.from_chain_type with an LCEL chain (U-07).
    LCEL (LangChain Expression Language) uses the | pipe operator to compose
    steps — much like Unix pipes. It is the modern LangChain approach and
    supports streaming natively.
  - Added query_stream() generator for token-by-token streaming output (U-05).
    app.py uses this so answers appear word-by-word rather than all at once
    after a 20-60 second wait.
"""

import os
import sys
from operator import itemgetter

# LangChain components for the LCEL RAG chain (U-07)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import all settings from central config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Shared logging setup — writes to both terminal and logs/chatbot.log
from src.logging_setup import setup_logging
log = setup_logging(__name__)


# =============================================================================
# STEP 1: Initialise connections (done once when the module is imported)
# =============================================================================

# Embedding function — same model used during ingestion.
# Using the same model for both ingestion and querying is critical:
# the search only works if the question vector "lives in the same space"
# as the document vectors. Switching models would make all stored embeddings useless.
#
# IMPORTANT — Phase 8 Known Fix backported here in v1.5.1 (2026-05-03):
# We do NOT pass `keep_alive` to OllamaEmbeddings. langchain-ollama 0.2.2
# uses Pydantic with `extra_forbidden`, so any field outside its schema
# (including `keep_alive`) raises:
#   pydantic_core._pydantic_core.ValidationError: 1 validation error for OllamaEmbeddings
#   keep_alive  Extra inputs are not permitted [type=extra_forbidden]
# That crash happens at module import time, so it blocks every importer of
# src.chat — including src/app.py — and the whole UI fails to start.
# `keep_alive` IS supported by ChatOllama and the direct ollama client, so
# we keep it on the chat LLM below. Same fix is applied in src/ingest.py.
embedding_function = OllamaEmbeddings(
    model=config.EMBED_MODEL,
    base_url=config.OLLAMA_BASE_URL,
)

# Connect to the existing ChromaDB collection (read-only from chat's perspective)
vector_store = Chroma(
    collection_name=config.CHROMA_COLLECTION,
    embedding_function=embedding_function,
    persist_directory=config.CHROMA_DIR
)

# Create a retriever — this is the component that searches ChromaDB.
# k=RETRIEVAL_TOP_K means "return the top K most similar chunks to the query"
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": config.RETRIEVAL_TOP_K}
)

# Create the LLM connection — this is the model that writes the final answer.
# keep_alive keeps the chat model resident between user questions.
llm = ChatOllama(
    model=config.CHAT_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=0.1,    # Low temperature = more factual, less creative.
                        # 0.0 = fully deterministic, 1.0 = very creative/random.
                        # For a technical knowledge base, 0.1 keeps answers grounded.
    keep_alive=config.OLLAMA_KEEP_ALIVE,
)

log.info("chat.py initialised — retriever ready (top_k=%d)", config.RETRIEVAL_TOP_K)


# =============================================================================
# STEP 2: Define the prompt template
# =============================================================================

# This is the exact instruction we give the LLM every time it answers a question.
# {context} is replaced with the retrieved document chunks.
# {question} is replaced with the user's actual question.
# Explicit instructions to say "I don't know" prevent hallucination.

RAG_PROMPT_TEMPLATE = """You are an expert technical assistant for an enterprise field service company.
You answer questions based ONLY on the information provided in the context below.
If the answer is not contained in the context, say clearly: "I don't have information about that in the current knowledge base."
Do not guess, invent, or use knowledge outside the provided context.
If the context includes descriptions of charts or diagrams, use that visual information in your answer.
If the conversation history below contains relevant prior exchanges, use them to understand follow-up questions and pronouns.

Conversation history (most recent last):
{history}

Context from knowledge base:
{context}

Question: {question}

Answer (based only on the context above):"""

prompt = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["history", "context", "question"]
)


# =============================================================================
# STEP 3: Build the LCEL RAG chain (U-07 — replaces deprecated RetrievalQA)
# =============================================================================
#
# LCEL uses the | (pipe) operator to chain steps left-to-right, just like
# Unix pipes. Each step receives the output of the previous step.
#
# Reading this chain left-to-right:
#   1. The input dict {"query": question} arrives.
#   2. itemgetter("query") | retriever pulls the question and searches ChromaDB,
#      returning a list of matching Document objects.
#   3. format_docs joins those documents into one context string.
#   4. itemgetter("query") passes the original question through unchanged.
#   5. The prompt template fills in {context} and {question}.
#   6. llm generates the answer token-by-token.
#   7. StrOutputParser() strips any message wrapper and returns a plain string.
#
# Why LCEL over RetrievalQA:
#   - RetrievalQA.from_chain_type is deprecated in LangChain 0.2+ and will be
#     removed in a future release. It emits deprecation warnings on every query.
#   - LCEL chains support .stream() natively — this is required for U-05
#     (word-by-word streaming in Gradio). RetrievalQA has no .stream() method.
#   - LCEL is more explicit and easier to debug — every step is visible.

def format_docs(docs):
    """Concatenate retrieved document chunks into one context string for the prompt.
    Each chunk is separated by a blank line so the LLM can distinguish sections."""
    return "\n\n".join(d.page_content for d in docs)

# Build the LCEL chain
qa_chain = (
    {
        # Left branch: take the query, run it through the retriever, format the docs
        "context":  itemgetter("query") | retriever | format_docs,
        # Middle branch: pass the query through unchanged as the question
        "question": itemgetter("query"),
        # Right branch: pass the pre-formatted history string through unchanged.
        # format_history() is called in query()/query_stream() in Python land,
        # so by the time itemgetter("history") fires, this value is already a
        # ready-to-use plain-text block.
        "history":  itemgetter("history"),
    }
    | prompt            # Fill {history}, {context} and {question} into the prompt template
    | llm               # Send the completed prompt to gemma4:e4b
    | StrOutputParser() # Parse the LLM response to a plain string
)


# =============================================================================
# STEP 3.5: Conversation history formatter (v1.5)
# =============================================================================

def format_history(history: list) -> str:
    """
    Convert Gradio's history list into a plain-text conversation block for the prompt.

    Gradio passes history as a list of dicts:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    We trim to the last HISTORY_TURNS messages and format as readable lines like:
        User: ...
        Assistant: ...
        User: ...

    If history is empty or None, returns a placeholder so the {history} prompt
    variable is never blank — keeps the prompt template clean and signals to
    the LLM that there is no prior context to consider.

    The empty-assistant placeholder appended by app.py's chat_handler mid-stream
    is filtered out defensively, even though chat_handler passes history[:-2]
    which already excludes it.

    Privacy note: this history exists only inside the user's browser tab.
    The server (Ollama, Python process, ChromaDB) never persists it between
    sessions — closing the tab discards the conversation entirely.
    """
    if not history:
        return "No prior conversation."
    # Take only the last HISTORY_TURNS messages to avoid bloating the prompt
    recent = history[-config.HISTORY_TURNS:]
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        # Skip the empty assistant placeholder that appears mid-stream
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


# =============================================================================
# STEP 4: Public query functions (called by app.py)
# =============================================================================

def query(question: str, history: list = None) -> dict:
    """
    Non-streaming query — runs the full RAG pipeline and returns when complete.
    Returns a dict with:
        answer   : the LLM's complete answer (str)
        sources  : deduplicated list of source document names (list[str])
        chunks   : the raw retrieved text chunks (list[str])

    Used by: smoke tests, the check_knowledge_base utility, and any caller
    that needs the complete answer before doing anything with it.
    For the live chat UI, use query_stream() instead.

    history : optional list of prior Gradio messages-format dicts (v1.5).
              Defaults to None, which format_history() turns into the
              "No prior conversation." placeholder. Backward-compatible
              with all existing callers that don't pass history.

    NOTE — double retrieval: retriever.invoke() runs once here to capture
    source document names, and again inside qa_chain.invoke() to build context.
    This is an intentional trade-off for simplicity at PoC scale — two fast
    local vector searches add negligible time compared to LLM inference.
    """
    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "chunks": []
        }

    log.info("Query received: %s", question[:100])

    # Run retrieval first to capture source document references
    docs = retriever.invoke(question)
    sources = sorted({d.metadata.get("source", "Unknown source") for d in docs})
    chunks  = [d.page_content for d in docs]

    # Run the full LCEL chain — returns a plain string (via StrOutputParser).
    # history is pre-formatted into a plain-text block before being passed
    # into the chain (the chain's itemgetter("history") just plucks it out).
    answer = qa_chain.invoke({
        "query":   question,
        "history": format_history(history),
    })

    log.info("Query answered — sources used: %s", sources)

    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  chunks,
    }


def query_stream(question: str, history: list = None):
    """
    Streaming query — yields partial answer dicts as the LLM produces tokens.
    Used by app.py's chat_handler so words appear in the UI progressively
    rather than making the user wait for the full response.

    Each yielded dict has the same shape as query()'s return value:
        answer   : the answer so far (grows with each yield)
        sources  : source document names (available from the first yield)
        chunks   : raw retrieved chunks (available from the first yield)

    history : optional list of prior Gradio messages-format dicts (v1.5).
              Browser-tab-scoped only — the server never remembers it
              between sessions.

    Why streaming matters for a 25B model on M4:
        Full-response wait time is 20–60 seconds depending on answer length.
        Streaming makes the first tokens appear within ~1–2 seconds, which
        dramatically improves the perceived responsiveness of the chatbot.
        This is also the prerequisite for any future voice-output feature —
        TTS can begin reading the first sentence while the rest is still
        being generated.

    NOTE — double retrieval: same as query() above. Two local vector searches,
    negligible cost compared to LLM streaming time.
    """
    if not question or not question.strip():
        yield {"answer": "Please enter a question.", "sources": [], "chunks": []}
        return

    log.info("Stream query received: %s", question[:100])

    # Retrieve source documents up front — we need them for citations
    # and they're available immediately, before streaming begins.
    docs    = retriever.invoke(question)
    sources = sorted({d.metadata.get("source", "Unknown source") for d in docs})
    chunks  = [d.page_content for d in docs]

    # Format the conversation history once, up front (v1.5).
    # Logged at DEBUG so we can verify in logs/chatbot.log that history
    # actually reached the model during follow-up-question testing.
    formatted_history = format_history(history)
    log.debug(
        "History block sent to LLM (%d chars): %s",
        len(formatted_history),
        formatted_history[:200].replace("\n", " | ")
    )

    # Stream tokens from the LCEL chain.
    # qa_chain.stream() yields one string token at a time.
    # We accumulate them into `partial` and yield after each token so the
    # Gradio UI updates the chat bubble in real time.
    partial = ""
    for token in qa_chain.stream({
        "query":   question,
        "history": formatted_history,
    }):
        partial += token
        yield {
            "answer":  partial,
            "sources": sources,
            "chunks":  chunks,
        }

    log.info("Stream query complete — sources used: %s", sources)


def check_knowledge_base() -> int:
    """
    Returns the number of chunks currently stored in ChromaDB.
    Used by app.py to warn the user if the knowledge base is empty.
    """
    try:
        collection = vector_store._collection
        count = collection.count()
        log.info("Knowledge base check: %d chunks in collection '%s'",
                 count, config.CHROMA_COLLECTION)
        return count
    except Exception as e:
        log.error("Could not query knowledge base: %s", e)
        return 0
