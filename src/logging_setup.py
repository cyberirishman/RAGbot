"""
logging_setup.py — Centralised logging configuration for the RAG chatbot.
==========================================================================
All scripts (ingest.py, chat.py, app.py) import from here so that log
formatting is consistent across the whole system.

What this gives you:
  - Timestamped log lines in the terminal (same as print, but structured)
  - A rotating log file at logs/chatbot.log (max 5MB × 5 files = 25MB total)
  - Log levels: INFO for normal progress, WARNING for recoverable problems,
    ERROR for failures that need attention.

Usage in any script:
    from src.logging_setup import setup_logging
    log = setup_logging(__name__)
    log.info("Connecting to %s", config.OLLAMA_BASE_URL)
    log.warning("Vision model failed: %s", error_message)
    log.error("Failed to process %s: %s", filename, error)
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
import sys

# Add the project root to sys.path so we can import config.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def setup_logging(name: str) -> logging.Logger:
    """
    Creates and returns a logger with the given name.
    Call this once at the top of each script:
        log = setup_logging(__name__)

    __name__ is a Python built-in that gives the current module's name
    (e.g. 'src.ingest', 'src.chat'). Using it as the logger name means
    log entries are tagged with the script they came from.
    """

    # Ensure the logs/ directory exists
    # exist_ok=True means "don't error if it already exists"
    log_dir = Path(config.PROJECT_ROOT) / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplicate log lines
    # if this function is called more than once (e.g. during testing)
    logger.handlers.clear()

    # --- Log format ---
    # Example output: 2026-04-30 14:23:01 [INFO] src.ingest: Parsing manual_v3.pdf
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- File handler (rotating) ---
    # maxBytes=5_000_000 → rotate when file hits 5MB
    # backupCount=5       → keep 5 old files (chatbot.log.1 through chatbot.log.5)
    # Total disk usage: at most 25MB of logs, then oldest are deleted automatically
    fh = RotatingFileHandler(
        log_dir / "chatbot.log",
        maxBytes=5_000_000,
        backupCount=5
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # --- Console (terminal) handler ---
    # This is the replacement for print() — same output, but with timestamp and level tag
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
