"""
app.py — Gradio Chat Interface
================================
Launches the browser-based chat UI for the RAG chatbot.
This is the file you run to start the chatbot.

Usage (from project root, with venv activated):
    python src/app.py

Then open your browser to: http://localhost:7860
"""

import os
import sys
import platform
import subprocess
import gradio as gr

# Import version modules for the startup banner
import langchain
import chromadb as chromadb_pkg

# Import our RAG query engine.
# query_stream() is the streaming generator used by the live chat handler (U-05).
# check_knowledge_base() reports how many chunks are in ChromaDB for the status bar.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chat import query_stream, check_knowledge_base
import config

# Shared logging setup — writes to both terminal and logs/chatbot.log
from src.logging_setup import setup_logging
log = setup_logging(__name__)


# =============================================================================
# STARTUP: Log the software environment
# =============================================================================

def log_versions():
    """Print and log all key software versions at startup."""
    log.info("[ENV] Python    : %s (%s %s)",
             sys.version.split()[0], platform.system(), platform.machine())
    log.info("[ENV] Gradio    : %s", gr.__version__)
    log.info("[ENV] LangChain : %s", langchain.__version__)
    log.info("[ENV] ChromaDB  : %s", chromadb_pkg.__version__)
    try:
        ollama_ver = subprocess.check_output(
            ["ollama", "--version"], text=True
        ).strip()
        log.info("[ENV] Ollama    : %s", ollama_ver)
    except Exception:
        log.warning("[ENV] Ollama    : (could not query — is Ollama running?)")

log_versions()


# =============================================================================
# Chat handler — called by Gradio every time the user sends a message (U-05/U-08)
# =============================================================================

def chat_handler(message: str, history: list):
    """
    Streaming chat handler — a Python generator that yields partial responses
    as the LLM produces tokens. Gradio detects that this function uses 'yield'
    and automatically switches to streaming mode, updating the chat bubble
    in real time rather than waiting for the full response.

    message : the user's current question (str)
    history : list of previous message dicts in Gradio 5 messages format (U-08):
              [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    Yields: (cleared_input_box, updated_history) after each token.

    Why a generator instead of a regular function:
        With gemma4:e4b on a 16GB Apple Silicon Mac, a complete answer typically
        takes 2-8 seconds (substantially faster than v1's gemma4:26b at 20-60s,
        thanks to the smaller active parameter count). A regular function makes
        the user stare at a spinner for the whole time, even if it is short.
        A generator streams tokens as they are produced — the first words appear
        within ~0.5-1 seconds, and the answer builds up progressively.
        This is also the foundation for any future voice-output feature.
    """
    if not message.strip():
        yield "", history
        return

    log.info("User message: %s", message[:100])

    # Immediately add the user's message and an empty assistant placeholder
    # to history, then yield so Gradio shows the user's message right away
    # without waiting for the LLM to start responding.
    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield "", history

    # Stream tokens from the RAG pipeline.
    # Each iteration of query_stream() yields a dict with the answer-so-far,
    # sources, and chunks. We update the last history entry (the assistant
    # placeholder) with the growing partial answer.
    #
    # v1.5 — Pass conversation history into query_stream() for follow-up
    # question handling (pronoun resolution, "what else does it say about
    # that?", etc.). We slice off the last 2 entries because we just appended
    # the current user message and an empty assistant placeholder above —
    # those aren't answered yet and would confuse the model.
    #
    # Privacy: this history exists only inside the user's browser tab.
    # The server never remembers it between sessions — closing the tab
    # discards the conversation. Open a second tab and you start fresh.
    prior_history = history[:-2]
    final_sources = []
    for partial in query_stream(message, history=prior_history):
        history[-1]["content"] = partial["answer"]
        final_sources = partial["sources"]
        yield "", history

    # After streaming is complete, append the source citations to the final answer.
    # This is done as a single update at the end so the citation block doesn't
    # flicker into view character-by-character during streaming.
    if final_sources:
        source_list = "\n".join(f"  • {s}" for s in final_sources)
        history[-1]["content"] += f"\n\n---\n**Sources used:**\n{source_list}"
        yield "", history


# =============================================================================
# Build the Gradio interface
# =============================================================================

def build_ui():
    """Constructs and returns the Gradio Blocks interface."""

    # Check if the knowledge base has any documents loaded
    kb_count = check_knowledge_base()
    if kb_count == 0:
        status_msg = (
            "⚠️ Knowledge base is empty. "
            "Drop documents into the ingest/ folder and run: python src/ingest.py"
        )
    else:
        status_msg = f"✅ Knowledge base ready — {kb_count:,} chunks loaded from your documents."

    with gr.Blocks(
        title="Enterprise RAG Chatbot",
        theme=gr.themes.Soft()
    ) as demo:

        gr.Markdown("# 🤖 Enterprise Knowledge Base Chatbot")
        gr.Markdown(
            "Ask questions about your uploaded documents. "
            "Answers are grounded in your knowledge base only — "
            "the system will tell you if it doesn't have the information."
        )
        gr.Markdown(f"**Status:** {status_msg}")

        # The main chat window.
        # type="messages" is required in Gradio 5+ (U-08).
        # The old default ("tuples") used [user_text, assistant_text] pairs —
        # deprecated in Gradio 5 and will be removed in a future release.
        # "messages" format uses {"role": "user"|"assistant", "content": "..."} dicts,
        # which matches OpenAI's chat format and is the Gradio 5 standard.
        chatbot = gr.Chatbot(
            label="Conversation",
            height=500,
            show_copy_button=True,
            type="messages",     # Required in Gradio 5+ — removes deprecation warning
        )

        # The message input row
        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Ask a question about your documents...",
                label="Your question",
                scale=8,
                lines=2
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        # Clear button to reset the conversation
        clear_btn = gr.Button("Clear conversation", variant="secondary")

        # Wire up the send action (clicking Send or pressing Enter)
        send_btn.click(
            fn=chat_handler,
            inputs=[msg_box, chatbot],
            outputs=[msg_box, chatbot]
        )
        msg_box.submit(
            fn=chat_handler,
            inputs=[msg_box, chatbot],
            outputs=[msg_box, chatbot]
        )

        # Wire up the clear action
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

        gr.Markdown(
            "---\n"
            f"**Models:** `{config.EMBED_MODEL}` (embeddings) · "
            f"`{config.CHAT_MODEL}` (chat + vision)  \n"
            f"**Vector store:** ChromaDB · Collection: `{config.CHROMA_COLLECTION}`"
        )

    return demo


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    log.info("Starting Enterprise RAG Chatbot UI...")
    log.info("Chat/vision model: %s", config.CHAT_MODEL)
    log.info("Embed model      : %s", config.EMBED_MODEL)
    log.info("ChromaDB    : %s", config.CHROMA_DIR)
    log.info("Open your browser at: http://localhost:%d", config.GRADIO_PORT)

    demo = build_ui()
    demo.launch(
        server_name=config.GRADIO_HOST,
        server_port=config.GRADIO_PORT,
        share=False    # share=True would create a public Gradio link — keep False for enterprise
    )
