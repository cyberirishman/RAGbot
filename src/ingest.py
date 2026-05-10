"""
ingest.py — Document Ingestion Pipeline
========================================
Run this script whenever you add new documents to the ingest/ folder.
It will process every supported file, extract text and images,
describe any images using the vision model, and store everything in ChromaDB.

After successful ingestion, each processed file is moved to ingest/processed/
so it cannot be accidentally re-ingested.

Usage (from project root, with venv activated):
    python src/ingest.py
"""

import os
import sys
import base64
import hashlib      # For generating deterministic chunk IDs (prevents duplicates)
import platform     # For the startup version banner
import subprocess   # For querying Ollama's version
import importlib.metadata  # For reading installed package versions reliably.
                            # We use this for Docling because the docling package
                            # does NOT expose a top-level `__version__` attribute
                            # (unlike LangChain, ChromaDB, Gradio, etc.).
                            # `importlib.metadata.version("docling")` reads the
                            # version from pip's installed-package metadata, which
                            # works for ANY installed package regardless of whether
                            # it exposes `__version__` on its module object.
from pathlib import Path
from io import BytesIO

# tqdm gives us a progress bar so we can see how far through a large batch we are
from tqdm import tqdm

# Pillow (PIL) handles image loading and format conversion
from PIL import Image

# Ollama Python client — used directly here for vision model calls (image input)
import ollama as ollama_client

# Docling document converter with explicit pipeline options
# We must use PdfPipelineOptions and set generate_picture_images=True
# or Docling will silently skip all image extraction from PDFs
#
# Note: there is intentionally NO bare `import docling` here. An earlier
# version of this script had `import docling` followed by `docling.__version__`
# in the startup banner — but Docling 2.14.0 does not expose `__version__` on
# its top-level module, so that line raised AttributeError on every run.
# The version is now read via `importlib.metadata.version("docling")` (see the
# `import importlib.metadata` line in the stdlib block above), which works
# regardless of whether the package exposes `__version__`. The submodule
# imports below are absolute imports — they do not require a bare `import
# docling`, because Python initialises the `docling` package automatically
# whenever any submodule of it is imported.
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# PictureItem is the correct Docling class for image elements
# The old doc.body.children approach did not match Docling's actual API
from docling_core.types.doc import PictureItem

# LangChain packages
import langchain
import chromadb as chromadb_pkg

# LangChain text splitters — two-stage chunking pipeline (U-16):
#   Stage 1: MarkdownHeaderTextSplitter splits on headings first, preserving
#            section context (h1/h2/h3) as metadata on every resulting chunk.
#   Stage 2: RecursiveCharacterTextSplitter then cuts each header section down
#            to character-bounded chunks that fit the embedding model's window.
# Docling exports all documents as markdown, so heading-aware splitting applies
# even to PDFs and DOCX files — section names travel with every chunk.
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ChromaDB via LangChain — stores and retrieves our embeddings
from langchain_chroma import Chroma

# Ollama embeddings via LangChain — calls nomic-embed-text to vectorise text
from langchain_ollama import OllamaEmbeddings

# Import all settings from our central config file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import our shared logging setup — replaces print() throughout this script
from src.logging_setup import setup_logging
log = setup_logging(__name__)


# =============================================================================
# STARTUP: Log the software environment for forensics / reproducibility
# =============================================================================
# When something breaks months from now, this banner tells you which exact
# versions of Python, LangChain, ChromaDB, Docling, and Ollama were running.
# Without this, debugging is guesswork.

def log_versions():
    """Print and log all key software versions at startup."""
    log.info("[ENV] Python    : %s (%s %s)",
             sys.version.split()[0], platform.system(), platform.machine())
    # LangChain and ChromaDB both expose a top-level `__version__` attribute
    # on their root module — that is the simplest way to read their versions.
    log.info("[ENV] LangChain : %s", langchain.__version__)
    log.info("[ENV] ChromaDB  : %s", chromadb_pkg.__version__)
    # Docling does NOT expose `__version__` on its top-level module
    # (verified against docling 2.14.0). Reading `docling.__version__`
    # raises AttributeError. Instead we use the standard-library
    # `importlib.metadata.version()` which queries pip's installed-package
    # metadata — that always returns the version of an installed package.
    log.info("[ENV] Docling   : %s", importlib.metadata.version("docling"))
    try:
        ollama_ver = subprocess.check_output(
            ["ollama", "--version"], text=True
        ).strip()
        log.info("[ENV] Ollama    : %s", ollama_ver)
    except Exception:
        log.warning("[ENV] Ollama    : (could not query — is Ollama running?)")

log_versions()


# =============================================================================
# STEP 1: Set up connections to our tools
# =============================================================================

log.info("Connecting to Ollama at %s", config.OLLAMA_BASE_URL)
log.info("Embedding model : %s", config.EMBED_MODEL)
log.info("Chat/vision model: %s", config.CHAT_MODEL)
# VISION_MODEL is an alias for CHAT_MODEL in the 2-model design,
# so logging CHAT_MODEL once is sufficient.
log.info("ChromaDB path   : %s", config.CHROMA_DIR)
log.info("Ingest folder   : %s", config.INGEST_DIR)

# Create the embedding function — this is what converts text into vectors.
#
# IMPORTANT — Phase 8 Known Fix (also see same pattern in Phase 9 chat.py):
# We do NOT pass `keep_alive` to OllamaEmbeddings. langchain-ollama 0.2.2
# uses Pydantic with `extra_forbidden`, so any field outside its schema
# (including `keep_alive`) raises:
#   pydantic_core._pydantic_core.ValidationError: 1 validation error for OllamaEmbeddings
#   keep_alive  Extra inputs are not permitted [type=extra_forbidden]
# That crash happens at module import time, so the entire ingest run fails
# before processing a single document. `keep_alive` IS supported by ChatOllama
# and the direct `ollama` client (see describe_image() below) — we use it there.
# In practice, this means the embedding model may be unloaded by Ollama after
# its idle timeout. For typical PoC ingest sizes (<100 MB) the reload cost is
# negligible. If it ever becomes an issue, set OLLAMA_KEEP_ALIVE via the
# environment for the whole Ollama server: `OLLAMA_KEEP_ALIVE=24h ollama serve`.
embedding_function = OllamaEmbeddings(
    model=config.EMBED_MODEL,
    base_url=config.OLLAMA_BASE_URL,
)

# Connect to (or create) the ChromaDB collection.
# persist_directory tells ChromaDB to save its data to disk in our chroma_db/ folder.
# If the collection already exists, we add to it. If not, it is created fresh.
vector_store = Chroma(
    collection_name=config.CHROMA_COLLECTION,
    embedding_function=embedding_function,
    persist_directory=config.CHROMA_DIR
)

# --- Stage 1: Header-aware splitter (U-16) ---
# Splits the document first on markdown headings. Each resulting section carries
# its heading path (h1/h2/h3) as metadata. This means a chunk from
# "Section 7.2 — Input Shaft Specifications" will have metadata:
#   {"h1": "Chapter 7 — Drive Train", "h2": "7.2 Input Shaft Specifications"}
# That context travels into ChromaDB and makes retrieval far more precise for
# technical documents with deep section hierarchies.
# strip_headers=False keeps the heading text inside the chunk content as well,
# so the LLM also sees the heading when it reads the retrieved chunk.
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#",   "h1"),
        ("##",  "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,
)

# --- Stage 2: Character-bounded splitter (U-16 + U-17) ---
# After header splitting, each section is cut into fixed-size chunks.
# CHUNK_SIZE is now 1800 (up from 1000 in v1.2) — large enough to keep
# multi-step procedures and specification tables intact in a single chunk.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    # Prefer cutting at paragraph breaks, then sentence ends, then word boundaries
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Initialise Docling with explicit PDF pipeline options.
#
# WHY we set generate_picture_images=True:
#   By default, Docling does NOT materialise PIL images for embedded pictures.
#   Without this flag, every image is silently skipped — the vision pipeline is
#   a no-op and you have no way of knowing. This was a critical bug in v1.0.
#
# images_scale=2.0: Higher resolution = better vision-model input quality.
#   The vision model can read fine print and chart labels more accurately.
#
# do_ocr=True: Enables OCR for scanned PDFs (photographed pages, old manuals).
#   Without OCR, scanned pages produce empty text chunks. With it, Docling runs
#   text recognition on each page image before processing.
pdf_pipeline_options = PdfPipelineOptions()
pdf_pipeline_options.generate_picture_images = True
pdf_pipeline_options.images_scale = 2.0
pdf_pipeline_options.do_ocr = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)
    }
)


# =============================================================================
# STEP 2: Vision model helper — describe an image in plain English
# =============================================================================

def describe_image(image: Image.Image, context: str = "") -> str:
    """
    Takes a PIL Image object, sends it to the vision model via Ollama,
    and returns a plain-English description of what the image shows.

    image   : a PIL.Image object extracted from the document by Docling
    context : optional surrounding text from the document — helps the vision
              model understand what kind of chart or diagram this likely is.
              Length is controlled by config.VISION_CONTEXT_CHARS.
    """
    # Convert the image to PNG bytes, then encode as base64.
    # Ollama's vision API expects images as base64-encoded strings.
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Build the prompt — giving the model context improves description quality.
    prompt = (
        "You are analysing an image extracted from an enterprise technical document. "
        "Describe what this image shows in precise detail. "
        "If it is a chart or graph: state the chart type, what the axes represent, "
        "the key data values, trends, and any conclusions the chart supports. "
        "If it is a diagram or schematic: describe the components and their relationships. "
        "If it is a table: summarise the data it contains. "
        "Be factual and specific — your description will be stored as searchable text. "
    )
    if context:
        # VISION_CONTEXT_CHARS controls how much surrounding text we send.
        # Tune this in config.py — more context helps but increases token use.
        prompt += f"\n\nSurrounding document context: {context[:config.VISION_CONTEXT_CHARS]}"

    try:
        response = ollama_client.chat(
            model=config.VISION_MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_b64]
            }],
            keep_alive=config.OLLAMA_KEEP_ALIVE,
        )
        return response["message"]["content"].strip()
    except Exception as e:
        # If vision model call fails, log it but don't crash the whole ingestion.
        # The image will be stored as "[Image: description unavailable]" so the
        # chunk is preserved and the rest of the document continues processing.
        log.warning("Vision model failed for image: %s", e)
        return "[Image: description unavailable]"


# =============================================================================
# STEP 3: Generate a deterministic chunk ID
# =============================================================================

def make_chunk_id(metadata: dict, text: str) -> str:
    """
    Creates a unique, reproducible ID for a chunk based on its source file,
    position within that file, and text content.

    WHY deterministic IDs matter:
        ChromaDB's add_texts() by default generates random IDs on every call.
        This means re-running ingest.py on a file that is already in ChromaDB
        will DUPLICATE all its chunks — doubling (or tripling, or worse) the
        number of times each chunk appears. Duplicate chunks silently degrade
        retrieval quality because the top-K results get dominated by identical
        entries from the same source.

        By using deterministic IDs, ChromaDB's upsert behaviour kicks in:
        same ID = overwrite the existing entry rather than create a new one.
        Re-running ingest.py on an unchanged file is now a safe no-op.
    """
    h = hashlib.sha1(
        f"{metadata['source']}|{metadata['chunk_index']}|{text}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{metadata['source']}::{metadata['chunk_index']}::{h}"


# =============================================================================
# STEP 4: Process a single document file
# =============================================================================

def process_document(file_path: Path) -> list[dict]:
    """
    Takes a file path, parses it with Docling, extracts text and images,
    and returns a list of text chunks with metadata ready to store in ChromaDB.

    Each chunk is a dict with:
        text     : the text content of the chunk
        metadata : source file, chunk index, chunk type, etc.
    """
    log.info("Parsing: %s", file_path.name)
    chunks = []

    try:
        # Docling converts the document into a structured internal format.
        # It handles PDFs, DOCX, MD, and TXT automatically based on file extension.
        result = converter.convert(str(file_path))
        doc = result.document

        # --- Extract main text content ---
        # export_to_markdown() gives us a clean, structured text representation
        # including headings, paragraphs, and table contents.
        full_text = doc.export_to_markdown()

        if full_text.strip():
            # --- Two-stage chunking (U-16) ---
            #
            # Stage 1: Split on markdown headings.
            # header_splitter returns a list of Document objects, each with
            # page_content = the text under that heading section, and
            # metadata = {"h1": "...", "h2": "...", "h3": "..."} for whichever
            # heading levels were present above that section.
            header_docs = header_splitter.split_text(full_text)

            # Stage 2: Within each header section, split into character chunks.
            # We collect all sub-chunks together with their inherited heading metadata.
            staged_chunks = []
            for hdoc in header_docs:
                sub_chunks = text_splitter.split_text(hdoc.page_content)
                for sc in sub_chunks:
                    staged_chunks.append({
                        "text":    sc,
                        "headers": hdoc.metadata,   # e.g. {"h1": "...", "h2": "..."}
                    })

            log.info("Text: %d chunks from %d characters (via 2-stage split)",
                     len(staged_chunks), len(full_text))

            for i, ch in enumerate(staged_chunks):
                chunks.append({
                    "text": ch["text"],
                    "metadata": {
                        "source":       file_path.name,
                        "source_path":  str(file_path),
                        "chunk_index":  i,
                        "chunk_type":   "text",
                        "total_chunks": len(staged_chunks),
                        # Spread heading keys (h1/h2/h3) into metadata.
                        # These become searchable fields in ChromaDB and help
                        # the retriever surface the right section for specific queries.
                        **ch["headers"],
                    }
                })

        # --- Extract and describe images ---
        # CORRECT Docling traversal: doc.iterate_items() yields (element, depth) tuples
        # for every element in the document tree. We check isinstance(element, PictureItem)
        # to identify image elements.
        #
        # The old pattern (doc.body.children + element.image.pil_image) was incorrect —
        # it did not match Docling's actual API and silently extracted zero images.
        image_count = 0
        for element, _level in doc.iterate_items():
            if isinstance(element, PictureItem):
                try:
                    # get_image() returns a PIL.Image or None
                    pil_image = element.get_image(doc)
                    if pil_image is None:
                        continue

                    # caption_text() returns the figure caption if one exists,
                    # giving the vision model useful context about what to look for
                    context = (
                        element.caption_text(doc)
                        if hasattr(element, "caption_text")
                        else ""
                    )

                    log.info("Image %d: sending to vision model...", image_count + 1)
                    description = describe_image(pil_image, context=context)

                    # Wrap the description as a chunk with clear labelling
                    image_text = f"[VISUAL CONTENT — {file_path.name}]\n{description}"

                    chunks.append({
                        "text": image_text,
                        "metadata": {
                            "source":      file_path.name,
                            "source_path": str(file_path),
                            "chunk_index": f"img_{image_count}",
                            "chunk_type":  "image_description",
                        }
                    })
                    image_count += 1

                except Exception as e:
                    log.warning("Could not process image element: %s", e)

        if image_count:
            log.info("Images: %d described by vision model", image_count)

    except Exception as e:
        log.error("Failed to process %s: %s", file_path.name, e)

    return chunks


# =============================================================================
# STEP 5: Main ingestion loop — process everything in the ingest/ folder
# =============================================================================

def run_ingestion():
    """
    Scans the ingest/ folder for all supported documents,
    processes each one, and loads the results into ChromaDB.

    After each file is successfully processed, it is moved to ingest/processed/
    so it will not be accidentally re-ingested on the next run.
    """
    ingest_path = Path(config.INGEST_DIR)

    # Create the processed/ subfolder if it does not exist yet.
    # Files are moved here after ingestion — a simple, visible audit trail.
    processed_dir = ingest_path / "processed"
    processed_dir.mkdir(exist_ok=True)

    # Find all files with supported extensions in the ingest/ root only
    # (not in processed/ — those are already done)
    files_to_process = [
        f for f in ingest_path.iterdir()
        if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
    ]

    if not files_to_process:
        log.info("No supported files found in %s", config.INGEST_DIR)
        log.info("Supported types: %s", ", ".join(config.SUPPORTED_EXTENSIONS))
        log.info("Drop documents into the ingest/ folder and re-run.")
        return

    log.info("Found %d document(s) to process.", len(files_to_process))

    all_texts    = []
    all_metadata = []

    # Process each file
    for file_path in tqdm(files_to_process, desc="Processing documents", unit="file"):
        chunks = process_document(file_path)
        for chunk in chunks:
            all_texts.append(chunk["text"])
            all_metadata.append(chunk["metadata"])

        # Move the file to processed/ after successful extraction.
        # If extraction produced zero chunks (e.g. empty file), we still move it
        # to avoid an infinite retry loop on a broken file.
        try:
            file_path.rename(processed_dir / file_path.name)
            log.info("Moved %s → ingest/processed/", file_path.name)
        except Exception as e:
            log.warning("Could not move %s to processed/: %s", file_path.name, e)

        print()  # blank line between documents for readability

    if not all_texts:
        log.warning("No text was extracted from any documents. Check file contents.")
        return

    log.info("Total chunks ready to embed: %d", len(all_texts))
    log.info("Sending to ChromaDB via %s ...", config.EMBED_MODEL)
    log.info("(This may take a few minutes for large document sets)")

    # --- Generate deterministic chunk IDs ---
    # Same source file + same chunk position + same text = same ID.
    # ChromaDB will OVERWRITE existing chunks with the same ID instead of
    # creating duplicates. This makes re-ingestion a safe, idempotent operation.
    chunk_ids = [make_chunk_id(m, t) for m, t in zip(all_metadata, all_texts)]

    # Add all chunks to ChromaDB in one call.
    # ChromaDB will call the embedding function (nomic-embed-text) on each chunk
    # and store both the text and its vector representation.
    # The ids= parameter enables upsert behaviour — existing IDs are overwritten.
    vector_store.add_texts(
        texts=all_texts,
        metadatas=all_metadata,
        ids=chunk_ids,
    )

    # --- Post-ingestion assertion: make image extraction visible ---
    # A count of zero means the vision pipeline ran as a silent no-op.
    # This surfaces the problem immediately rather than only discovering it
    # when the chatbot fails to answer chart-related questions.
    image_chunks = sum(
        1 for m in all_metadata
        if m.get("chunk_type") == "image_description"
    )
    log.info("Image-description chunks created: %d", image_chunks)
    if image_chunks == 0:
        log.warning(
            "No image descriptions were generated. "
            "If your documents contain charts or diagrams, verify that "
            "pdf_pipeline_options.generate_picture_images = True and that "
            "the vision model (%s) is reachable at %s.",
            config.VISION_MODEL, config.OLLAMA_BASE_URL
        )

    log.info("SUCCESS: %d chunks stored in ChromaDB.", len(all_texts))
    log.info("Collection '%s' in %s", config.CHROMA_COLLECTION, config.CHROMA_DIR)
    log.info("Your knowledge base is ready. Run app.py to start the chatbot.")


# Run ingestion when this script is executed directly
if __name__ == "__main__":
    run_ingestion()
