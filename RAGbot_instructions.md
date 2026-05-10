 # RAG Chatbot (RAGbot)— Local Buildout Plan & Lesson Guide (16GB Teaching Edition)

**Machine target:** Apple Silicon Mac with **16GB+ RAM** (Mac Mini M-series, MacBook Air M-series, or similar) | macOS 14+  
**User:** any user with admin (sudo) rights — does not need to be root  
**Date:** 2026-05-07  
**Version:** 2.7 — 2026-05-07  
**Goal:** Build a locally-hosted RAG chatbot that can ingest PDF, Word, Markdown, and text files — including understanding graphs and charts — and answer questions through a browser-based chat interface, on **consumer-class Apple Silicon hardware (16GB RAM minimum)**. This is the *teaching edition* of the project — the architecture and codebase are otherwise identical to the production-class v1 reference (which targets 64GB hardware and the heavier `gemma4:26b` model).


---

## How to Read This Document

Every step follows this format:

```
command to run
```
> **What it does:** Plain-language explanation of the command.  
> **Why we need it:** How it fits into the chatbot system.

Read the explanation before running the command. Understanding *why* matters more than just copy-pasting.

---

## Architecture Overview (Before We Start)

The chatbot has two distinct pipelines that share two pieces of infrastructure. The **ingest pipeline** runs once per document (slow, batch — typically minutes per document) and populates ChromaDB. The **query pipeline** runs once per user message (fast, online — typically 2–8 seconds per answer) and reads from the ChromaDB that ingest built. The shared infrastructure is **Ollama** (which serves both AI models over a local HTTP API) and **ChromaDB** (the on-disk vector database). The diagrams below break this apart so you can see exactly what runs when.

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

**2 AI models design  - both running on ollama**

** `gemma4:e4b` is Google's "effective 4 billion parameter" multimodal model — it natively understands both text and images. This means it handles image description during ingest AND generates chat answers from a single loaded model, eliminating the need for a separate vision model entirely.

** `nomic-embed-text` remains the dedicated embedding model — purpose-built for semantic search and far more accurate for RAG retrieval than a general LLM. Stated capabilities of Gemma 4 e4b: vision, tools, thinking, audio, and cloud-deployable.

**Vision quality disclosure:** The diagram and chart description quality of `gemma4:e4b` is meaningfully lower than the `gemma4:26b` model used in v1 of this project. For a teaching PoC and most text-heavy field-service Q&A, this trade-off is acceptable. If your end use case depends on rich diagram/chart understanding (e.g. parts-explosion drawings, schematic reading), consider the v1 architecture or run ingest *once* on a 64GB machine using `gemma4:26b` and ship the resulting `chroma_db/` to a v2 deployment.

**The key concept — RAG (Retrieval-Augmented Generation):**  
Instead of asking the LLM a question cold (where it might guess or hallucinate), we first *retrieve* the most relevant passages from your documents, then *augment* the prompt with that context, then *generate* a grounded answer. The LLM only speaks from evidence it was handed.

**RAM footprint (16GB Apple Silicon Mac):**

| Model | Role | RAM (approx, during active use) |
|---|---|---|
| `nomic-embed-text` | Embeddings | ~0.4 GB |
| `gemma4:e4b` | Vision (ingest) + Chat (queries) | ~10–12 GB |
| System + ChromaDB + Python overhead | — | ~3–4 GB |
| **Total** | | **~14–16 GB of 16 GB** |

Tight but workable on a 16GB machine, comfortable on 24GB+. Two practical recommendations on 16GB: close other RAM-hungry apps (browsers with many tabs, IDEs) before running ingest, and consider lowering `OLLAMA_KEEP_ALIVE` from `24h` to `5m` if you do other work between chatbot sessions — the model will reload more slowly afterwards but RAM is reclaimed when idle. **For comparison**, v1's `gemma4:26b` configuration uses ~17 GB for the chat/vision model alone — about double — which is why v1 targets 32GB+ machines.


---
We Assume NO packages or dependencies are installed

## Phase 0 — Install Homebrew (macOS Package Manager)

Homebrew is the standard package manager for macOS. A package manager is a tool that lets you install, update, and remove software from the terminal with a single command — instead of going to a website, downloading an installer, clicking through a wizard, and managing updates manually. On Linux, the equivalent tools are `apt` (Ubuntu/Debian) or `yum` (RedHat/CentOS). On macOS, Homebrew fills that role.

Almost everything we install in this project — Python, and any system-level tools — comes through Homebrew. It installs software into `/opt/homebrew/` (on Apple Silicon Macs) which keeps it completely separate from the files macOS itself depends on.

**Step 0.1 — Check if Homebrew is already installed**

```bash
brew --version
```
> **What it does:** Asks Homebrew to report its version number.  
> **What to look for:** If you see something like `Homebrew 4.x.x` then Homebrew is already installed — **skip Step 0.2 and go straight to Step 0.3**. If you see `command not found: brew` then Homebrew is not installed and you must complete Step 0.2.

**Step 0.2 — Install Homebrew (only if Step 0.1 showed "command not found")**

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
> **What it does:** Downloads and runs the official Homebrew installer script directly from the Homebrew project on GitHub.  
> **Breaking down the command:**
> - `/bin/bash -c "..."` — runs the quoted text as a shell script using the built-in bash interpreter
> - `curl` — a tool for downloading content from URLs (built into macOS)
> - `-fsSL` — flags that tell curl to: follow redirects (`L`), fail silently on errors (`f`), suppress progress bars (`s`), and show errors if they occur (`S`)
> - The URL points to Homebrew's official install script
>
> **What to expect:** The installer will ask for your password (it needs admin rights to create the `/opt/homebrew/` folder), then download and install Homebrew. This takes 2–5 minutes depending on your internet speed.  
> **After install on Apple Silicon (M1/M2/M3/M4):** The installer will print instructions to add Homebrew to your PATH. Run the two lines it shows you — they will look like:
> ```bash
> echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /var/root/.zprofile
> eval "$(/opt/homebrew/bin/brew shellenv)"
> ```
> **Why this step:** Without adding Homebrew to your PATH, your terminal won't know where to find the `brew` command or any software installed through it.

**Step 0.3 — Verify Homebrew is working correctly**

```bash
brew doctor
```
> **What it does:** Homebrew runs a self-diagnosis and reports any problems with its installation.  
> **What to look for:** You should see `Your system is ready to brew.` at the end. Minor warnings (shown in yellow) are usually safe to ignore. Errors (shown in red) should be resolved before continuing — they typically include a suggested fix command.

**Step 0.4 — Check if Git is already installed**

```bash
git --version
```
> **What it does:** Asks Git to report its version number.  
> **What to look for:** If you see something like `git version 2.x.x`, Git is already installed — this often happens because macOS bundles a copy of Git with the Xcode Command Line Tools. **You can skip Step 0.5 and go straight to Step 0.6 to verify.** If you see `command not found: git`, continue to Step 0.5.

**Step 0.5 — Install Git via Homebrew (only if Step 0.4 showed "command not found")**

```bash
brew install git
```
> **What it does:** Tells Homebrew to download and install the latest stable version of Git from its package repository. Git is the industry-standard *version control system* — a tool that tracks every change you make to text files (typically code and configuration), lets you roll back to any prior version, and lets you sync those changes between machines through a remote repository (GitHub, GitLab, Bitbucket, etc.).  
> **Why we need it:** Three reasons. (1) **Production migration** — when we eventually move from this Mac Mini to the VPS in production, the cleanest way to ship the project's source code, configuration, and documentation is via a Git repository: push from the Mac, pull on the VPS. (2) **Pip occasionally needs it** — some Python packages (and their transitive dependencies) install more cleanly when `git` is on the PATH; pip will sometimes fetch a specific commit from a Git URL. (3) **Forensic history** — going forward, every meaningful edit to the buildout (config tweaks, prompt changes, new ingest scripts) should be committed so we have a record of what changed and when, instead of relying on memory.  
> **Why Homebrew over the bundled Xcode copy:** macOS's bundled Git lags behind by 1–2 years and only updates when you update Xcode. Homebrew's copy upgrades cleanly with `brew upgrade git` whenever you want it to, and it keeps Git on the same management surface as everything else in this project.  
> **What to expect:** Homebrew downloads Git and a small set of dependencies. Takes 30–60 seconds.

**Step 0.6 — Verify Git is working correctly**

```bash
git --version
```
> **What it does:** Same command as Step 0.4 — but now we expect a real version number instead of "command not found".  
> **What to look for:** A line like `git version 2.45.x` (or newer). To confirm you are running the Homebrew-installed copy rather than the older Xcode-bundled one, also run `which git`. On Apple Silicon Macs, Homebrew's Git lives at `/opt/homebrew/bin/git`. If `which git` shows `/usr/bin/git`, that's the Xcode copy — usable, but not the one we just installed; close and reopen your terminal so it picks up Homebrew's PATH order.  
> **If you still see "command not found":** your shell's PATH may not yet include Homebrew's binary directory. The simplest fix is to close the terminal window and open a fresh one — login shells re-read your shell profile on launch. If that doesn't work, re-run the two `eval` commands the Homebrew installer printed at the end of Step 0.2.

---

## Phase 1 — Install and Start Ollama

Ollama is the local server that loads and runs our two AI models (`nomic-embed-text` and `gemma4:e4b`). Think of it as a lightweight web service that lives on your Mac: it listens on `http://0.0.0.0:11434`, accepts HTTP requests from our Python code, and returns model output. Without it running, every other part of the chatbot has nothing to talk to.

This phase has five steps: check whether Ollama is already installed, install it via Homebrew if not, start the background service, verify it's reachable, and learn how to manage it day-to-day. The model downloads come in the next phase (Phase 2) — this phase is about the runtime that hosts the models.

**Step 1.1 — Check if Ollama is already installed**

```bash
ollama --version
```
> **What it does:** Asks Ollama to print its installed version number.  
> **What to look for:** If you see something like `ollama version is 0.x.x`, Ollama is already installed — **skip Step 1.2 and go straight to Step 1.3** (we still need to make sure the background service is started). If you see `command not found: ollama`, continue to Step 1.2.

**Step 1.2 — Install Ollama via Homebrew (only if Step 1.1 showed "command not found")**

```bash
brew install ollama
```
> **What it does:** Tells Homebrew to download and install Ollama's CLI runtime and background daemon. This is the **formula** install — a single system-level binary at `/opt/homebrew/bin/ollama` (Apple Silicon path) plus a launchd-compatible daemon. No GUI, no menu-bar app.  
>
> **Why the formula and not the cask:** Homebrew offers two ways to install Ollama. `brew install ollama` (the formula) installs a system-level binary that any user on the Mac can talk to over `http://0.0.0.0:11434`. `brew install --cask ollama` installs a `.app` version that runs as a menu-bar app *for the currently-logged-in user only*. If a second user logs in and starts the cask version, the two instances fight over port 11434 and may mutate each other's model store. This project's `config.py` deliberately sets `OLLAMA_BASE_URL = "http://0.0.0.0:11434"` — the `0.0.0.0` binding means "serve every user on this machine, not just me" — and the formula install is what makes that work cleanly. The cask install also locks you into macOS only; the formula install is the same management surface (`brew install`, `brew upgrade`, `brew services`) you'll use on a future Linux VPS in production.  
>
> **If you already have the cask version:** uninstall it first with `brew uninstall --cask ollama`, then `brew install ollama` (the formula). Models you already pulled will still be in `~/.ollama/models/` and won't need to be re-downloaded.  
>
> **What to expect:** Homebrew downloads Ollama and a small set of dependencies. Takes 30–90 seconds.

**Step 1.3 — Start the Ollama background service**

```bash
brew services start ollama
```
> **What it does:** Registers Ollama with macOS's launchd (the system-wide service manager) via Homebrew's `brew services` wrapper. The daemon starts now and auto-starts every time you log in — set-and-forget. The Linux equivalent is `systemctl enable --now ollama`; same idea, different command.  
>
> **Why this is the right default:** Ollama's idle overhead is small — a few MB of RAM with no model loaded. Models only consume their full RAM when actually serving a request, and `OLLAMA_KEEP_ALIVE` in `config.py` (Phase 7) controls how long they stay resident after the last request. Auto-starting on login means a student picking up the project after a reboot doesn't have to remember to start anything before the chatbot will work.  
>
> **What to expect:** A line confirming the service started, e.g. `Successfully started ollama`. The service runs in the background — there is no visible window or terminal.  
>
> **How to confirm it's running right now:** `brew services list | grep ollama` — you should see status `started`.  
>
> **Teaching tip — watch live traffic when you want to:**
>
> ```bash
> tail -f /opt/homebrew/var/log/ollama.log
> ```
>
> `tail` prints the end of a file. The `-f` flag (for "follow") makes it keep printing new lines as they're written, so you see every API call from the chatbot in real time as it happens. Press `Ctrl+C` to stop watching — this only stops the `tail` command, not Ollama itself. When students ask the chatbot a question, this log shows the embedding request, the chat completion, and the streaming tokens as they come out of the model. Makes the otherwise-invisible HTTP layer concrete. There is also a companion error log at `/opt/homebrew/var/log/ollama.err.log` — same `tail -f` trick — that is the first place to look when something is failing silently.

**Step 1.4 — Verify the Ollama daemon is reachable**

```bash
ollama list
```
> **What it does:** Asks the running Ollama daemon to list every model currently installed. This double-checks both that Ollama is installed AND that the background service we just started is actually answering on port 11434.  
>
> **What to look for:** A small table with column headers (`NAME`, `ID`, `SIZE`, `MODIFIED`). On a brand-new install, the rows under the header may be empty — that is expected; Phase 2 will populate them with `nomic-embed-text` and `gemma4:e4b`.  
>
> **If you see "Error: could not connect to ollama app, is it running?":** the daemon didn't start. Check the service status with `brew services list | grep ollama`. If it shows `error` or `stopped`, run `brew services restart ollama` and try `ollama list` again. If the restart fails, the error logs in `/opt/homebrew/var/log/ollama.err.log` will tell you why (`tail /opt/homebrew/var/log/ollama.err.log` to read the most recent ~10 lines).

**Step 1.5 — How to manage Ollama in future sessions (reference, not an action)**

Once Phase 1 is done, you do not need to start Ollama manually again — `brew services start ollama` registered it with launchd and macOS handles it from there. This subsection is a reference card so you know what to do if you ever need to pause it, restart it, or check on it.

| Task | Command |
|---|---|
| Check if Ollama is running | `brew services list \| grep ollama` |
| Watch live API traffic | `tail -f /opt/homebrew/var/log/ollama.log` (Ctrl+C to stop watching, does NOT stop Ollama itself) |
| Watch error log | `tail -f /opt/homebrew/var/log/ollama.err.log` |
| Stop Ollama (free RAM for another job) | `brew services stop ollama` |
| Start Ollama after stopping it | `brew services start ollama` |
| Restart Ollama (e.g. after editing an Ollama config or an environment variable) | `brew services restart ollama` |
| List all Homebrew-managed services | `brew services list` |

> **Why a stop command exists:** while Ollama's idle footprint is small, a loaded model can hold ~10 GB of RAM. If you need that RAM for something else (large video edit, big build, another LLM), `brew services stop ollama` gives it back. Auto-start resumes on next login unless you also run `brew services stop ollama` after a future reboot.

---

## Phase 2 — Pull Required Ollama Models

Ollama must be installed and running at `http://0.0.0.0:11434` before this phase begins (Phase 1 installed and started it; `ollama list` in Step 1.4 confirmed it's reachable). We pull two models: a dedicated embedding model and a multimodal chat/vision model.

**Step 2.1 — Pull the embedding model**

```bash
ollama pull nomic-embed-text
```
> **What it does:** Downloads the `nomic-embed-text` model (~274 MB) into Ollama's local model store.  
> **Why we need it:** This is a *dedicated embedding model* — its only job is to convert text into lists of numbers (called vectors) that represent meaning. ChromaDB stores and searches these vectors. Using a purpose-built embedding model instead of a general chat LLM is the single biggest factor in RAG retrieval accuracy. Think of it like an index at the back of a book — the better the index, the faster you find the right page.

**Step 2.2 — Pull the chat + vision model**

```bash
ollama pull gemma4:e4b
```
> **What it does:** Downloads Google's Gemma 4 e4b model (~10 GB) into Ollama's local model store. The download will take 5–15 minutes on a typical broadband connection.  
> **Why we need it:** This is the model that does almost all the work — it reads images and charts during document ingestion (writes plain-English descriptions of them) AND it generates chat answers when the user asks questions. One model, two jobs, both because Gemma 4 is multimodal.  
> **What "e4b" means:** Gemma 4 ships in several sizes — e2b, e4b, 26b, 31b. The `e` prefix means "effective parameter count" — Gemma 4 uses a MatFormer-style architecture where only a subset of the model's parameters activate for any given input, so the practical RAM and compute footprint is closer to a 4-billion-parameter model than the full weight count would suggest. The end result is a multimodal model that runs on consumer hardware. Capabilities advertised by Google: vision, tools, thinking, audio, cloud-deployable.  
> **Why this size:** e4b strikes the right balance for a 16GB Mac — large enough to handle multimodal RAG with reasonable accuracy, small enough to run alongside the OS and Python without swapping. Going larger (26b) needs 32GB+ as proven by v1 of this project; going smaller (e2b) loses too much answer quality for our field-service use case.

**Step 2.3 — Verify all models are ready**

```bash
ollama list
```
> **What it does:** Lists every model currently downloaded in Ollama.  
> **Why we need it:** Confirms that both required models are present before we build anything that depends on them. If a model is missing, the chatbot will fail silently at runtime — better to catch it now.

You should see at minimum these two in the output:
- `nomic-embed-text`
- `gemma4:e4b`

If either is missing, re-run the corresponding `ollama pull` command above before continuing.

---

## Phase 3 — Install Python 3.12

macOS ships with a system Python, but we must **never** use it for projects. Apple uses the system Python internally for macOS tools and may silently change or break it at any time. We install our own clean, known version via Homebrew — completely separate from Apple's copy.

**Step 3.1 — Check if Python 3.12 is already installed**

```bash
python3.12 --version
```
> **What it does:** Asks specifically for Python version 3.12 and prints its version number if it exists.  
> **Why this specific command:** macOS may have multiple Python versions installed simultaneously. Typing `python3 --version` only tells you what the *default* Python 3 is — which could be 3.9 or 3.11. We ask for `python3.12` explicitly to know whether the exact version we need is present.  
> **What to look for:**
> - You see `Python 3.12.x` — Python 3.12 is already installed. **Skip Step 3.2 and go to Step 3.3.**
> - You see `command not found: python3.12` — Python 3.12 is not installed. Continue to Step 3.2.
> - You see a different version like `Python 3.11.x` — an older version is installed but not 3.12. Continue to Step 3.2 (both can coexist safely — we are not replacing anything).

**Step 3.2 — Install Python 3.12 via Homebrew (only if Step 3.1 showed "command not found")**

```bash
brew install python@3.12
```
> **What it does:** Uses Homebrew to download and install Python 3.12. Homebrew installs it to `/opt/homebrew/bin/python3.12` — this is a completely independent copy from any Python that macOS or a previous installer put on the system. Multiple Python versions can coexist without interfering with each other.  
> **Why version 3.12 specifically:** All of our key packages (Docling, ChromaDB, LangChain, Gradio) are tested and confirmed stable on Python 3.12. Newer versions (3.13+) sometimes have compatibility gaps with machine learning libraries that have not yet updated. Older versions (3.10, 3.11) will mostly work but may produce deprecation warnings. 3.12 is the safe, proven choice as of 2026.  
> **What to expect:** Homebrew downloads Python and its dependencies. Takes 2–5 minutes. You will see a lot of output — this is normal.

**Step 3.3 — Confirm Python 3.12 is accessible**

```bash
python3.12 --version
```
> **What it does:** Verifies the exact Python 3.12 binary is reachable and prints its full version.  
> **What you should see:** `Python 3.12.x` (where x is any patch number, e.g. `Python 3.12.9`).  
> **If you still see "command not found" after installing:** Homebrew may not have linked the binary. Run `brew link python@3.12 --force` then try again.

**Step 3.4 — Verify pip is available for Python 3.12**

```bash
python3.12 -m pip --version
```
> **What it does:** Asks Python 3.12 to run its own built-in pip module and report its version.  
> **Why this form:** Using `python3.12 -m pip` (rather than just `pip3`) guarantees we are using the pip that belongs to *this specific Python version*. If you have multiple Python versions installed, bare `pip3` may point at the wrong one — causing packages to be installed where the project can't find them.  
> **What you should see:** Something like `pip 24.x.x from /opt/homebrew/lib/python3.12/site-packages/pip (python 3.12)`. The `python 3.12` at the end confirms it is the right pip.  
> **Why pip matters:** `pip` is Python's package manager — the tool that downloads and installs Python libraries (ChromaDB, LangChain, Gradio, Docling, etc.). Think of it as the App Store for Python code. Without it, we cannot install any of the software our chatbot depends on.

---

## Phase 4 — Create the Project Structure

Before writing any code, we create the folder skeleton the entire project will live in. Having an organised structure from the start means the IT admin maintaining this later will always know where to find things.

**Step 4.1 — Navigate to your home folder ( by typing cd ~ )and create a new folder called RAGbot**

```bash
cd ~
mkdir RAGbot
```
> **What it does:** Changes your terminal's current working directory to Users/YOUR USERNAME folder. THen we create a new folder called RAGbot where all project files will live All commands from here on run relative to this location unless stated otherwise.  
> **`cd` stands for:** "change directory" — the most fundamental terminal navigation command.

> **`cd ` `** changes directory to the users home directory  — We can also type cd /Users/YOUR_USERNAME
**Step 4.2 — Create all project subdirectories and the package marker file**

```bash
mkdir -p src ingest chroma_db logs
touch src/__init__.py
```
> **What it does:** Creates four folders inside the project directory, then creates an empty `src/__init__.py` file.  
> **`mkdir` stands for:** "make directory". The `-p` flag means "create parent folders too if they don't exist, and don't error if the folder is already there."  
> **`touch` stands for:** Creates an empty file if it doesn't exist (or updates its timestamp if it does). Zero bytes is fine for `__init__.py`.  
> **What each folder is for:**
> - `src/` — Source code. All Python scripts that power the chatbot live here.
> - `ingest/` — The inbox. Drop any PDF, Word, MD, or TXT file here and the ingestion script will process it into ChromaDB. After ingestion, processed files are moved to `ingest/processed/` automatically.
> - `chroma_db/` — ChromaDB's data folder. The vector database writes its files here automatically. You never need to touch this folder manually.
> - `logs/` — Log files. `chatbot.log` is written here automatically during ingestion and while the chatbot runs. Check this file when troubleshooting.
>
> **Why `src/__init__.py` is critical:** Python needs this empty file to recognise `src/` as a *package* — a folder of importable modules. Without it, the line `from src.chat import query` in `app.py` will fail with `ModuleNotFoundError: No module named 'src'` the first time you run the chatbot. This file does not need any content — its mere existence is the signal Python requires.

**Step 4.3 — Confirm the structure looks correct**

```bash
ls -la
ls -la src/
```
> **What it does:** Lists all files and folders in the current directory and inside `src/`.  
> **What you should see:** `src/`, `ingest/`, `chroma_db/`, and `logs/` listed at the project root, and `__init__.py` listed inside `src/`. The `__init__.py` will show a size of 0 — that is correct.

**Verification:**
```bash
ls -la src/
# Must show __init__.py (size 0 is fine).
```

---

## Phase 5 — Create and Activate the Python Virtual Environment

This is one of the most important concepts in Python development.

**Step 5.1 — Create the virtual environment**

```bash
python3.12 -m venv chatbot-16G-venv
```
> **What it does:** Creates a self-contained Python environment in a new folder called `chatbot-16G-venv/` inside your project directory.  
> **Why the name `chatbot-16G-venv`:** The folder name encodes both the project (`chatbot`) and the hardware target (`16G`, for the 16GB Teaching Edition). When you see `chatbot-16G-venv` in your prompt or in your folder list, you know exactly what it is — and if you ever clone the v1 64GB project alongside this one, the two venvs cannot be confused. Using a generic name like `venv` is technically fine but ambiguous; the explicit name is the v2.3 default.  
> **Why this matters:** Imagine Python as a kitchen. Your Mac has a system kitchen (the built-in Python) that macOS uses to cook its own meals. If you start adding exotic ingredients (our libraries like ChromaDB, Docling, etc.) to the system kitchen, you might accidentally ruin macOS's own recipes. A virtual environment is like building a *second, private kitchen* just for this project — completely isolated. You can install, upgrade, or break anything in it without touching the system kitchen. Every professional Python project uses one.  
> **The `-m venv` part:** Tells Python to run its built-in `venv` module (the tool that creates virtual environments).

**Step 5.2 — Activate the virtual environment**

```bash
source chatbot-16G-venv/bin/activate
```
> **What it does:** Activates the virtual environment. Your terminal prompt will change to show `(chatbot-16G-venv)` at the beginning — for example `(chatbot-16G-venv) andrew-m4@Andrew-M4s-Mac-mini Chatbotv2 %`. This is your visual confirmation that you are working *inside* the isolated environment.  
> **Why `source`:** The `source` command runs a script *inside* your current terminal session (rather than opening a new one), which is necessary for the environment switch to take effect.  
> **IMPORTANT:** You must run this command every time you open a new terminal window to work on this project. If you forget, pip installs will go to the wrong Python and nothing will work correctly.

> **Step 5.2 verification — confirm the venv is active**
>
> ```bash
> which python
> python --version
> ```
> The first command should print a path that ends with `chatbot-16G-venv/bin/python` — that proves your `python` invocations are now running inside the venv. The second should print `Python 3.12.x`. If `which python` instead prints `/usr/bin/python3` or `/opt/homebrew/bin/python3.12`, the venv is NOT active — go back and re-run `source chatbot-16G-venv/bin/activate`.

**Step 5.3 — Upgrade pip inside the virtual environment**

```bash
pip install --upgrade pip
```
> **What it does:** Updates pip itself to the latest version inside your virtual environment.  
> **Why before anything else:** pip is used to install everything else. An outdated pip sometimes fails to install modern packages correctly, especially on Apple Silicon. This one-time upgrade prevents hard-to-diagnose errors later.

---

## Phase 6 — Install Python Dependencies

The project ships with `requirements.txt` already in the project root. It lists every Python library the chatbot needs, pinned to the exact version verified working in the v2 starter environment — all 182 packages, direct dependencies plus every transitive sub-dependency. Because every version is pinned, pip's resolver has no choices to make on a fresh install. This is what guarantees that "what worked when this project was built" is exactly what gets installed on your machine.

Pinning every version (e.g. `langchain==0.3.18` rather than just `langchain`) ensures that an IT admin rebuilding this environment in 6 months gets the exact same software stack — not whatever happens to be the latest release of each library, which may have breaking changes.

**Step 6.1 — Confirm `requirements.txt` is present**

The file ships with the project, so you should not need to create it. Confirm it is in place and not truncated:

```bash
ls -la requirements.txt
wc -l requirements.txt
```
> **What it does:** Lists the file (`ls -la`) so you can see it exists, then counts its lines (`wc -l`) so you can confirm it is not truncated.  
> **What you should see:** a file named `requirements.txt` of around 180 lines. If the file is missing or much shorter, do not continue — re-copy it from the v2 starter folder before running the install.

**(Reference — full canonical content of `requirements.txt`)**

You should not need this block on a normal install. It is reproduced below as the source of truth for the file's contents — useful only if `requirements.txt` is ever lost and you need to recreate it from scratch. To rebuild the file from this reference, paste the whole `cat > … EOF` block into your terminal in the project root:

```bash
cat > requirements.txt << 'EOF'
accelerate==1.13.0
aiofiles==23.2.1
aiohappyeyeballs==2.6.1
aiohttp==3.13.5
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
asgiref==3.11.1
attrs==26.1.0
backoff==2.2.1
bcrypt==5.0.0
beautifulsoup4==4.14.3
brotli==1.2.0
build==1.5.0
certifi==2026.4.22
charset-normalizer==3.4.7
chroma-hnswlib==0.7.6
chromadb==0.5.20
click==8.3.3
dataclasses-json==0.6.7
deepsearch-glm==1.0.0
defusedxml==0.7.1
dill==0.4.1
distro==1.9.0
docling==2.14.0
docling-core==2.74.1
docling-ibm-models==3.13.2
docling-parse==3.4.0
durationpy==0.10
easyocr==1.7.2
et_xmlfile==2.0.0
fastapi==0.136.1
ffmpy==1.0.0
filelock==3.29.0
filetype==1.2.0
flatbuffers==25.12.19
frozenlist==1.8.0
fsspec==2026.4.0
googleapis-common-protos==1.74.0
gradio==5.50.0
gradio_client==1.14.0
groovy==0.1.2
grpcio==1.80.0
h11==0.16.0
hf-xet==1.4.3
httpcore==1.0.9
httptools==0.7.1
httpx==0.28.1
httpx-sse==0.4.3
huggingface_hub==0.36.2
idna==3.13
ImageIO==2.37.3
importlib_metadata==8.7.1
importlib_resources==7.1.0
Jinja2==3.1.6
jsonlines==4.0.0
jsonpatch==1.33
jsonpointer==3.1.1
jsonref==1.1.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kubernetes==35.0.0
langchain==0.3.18
langchain-chroma==0.2.0
langchain-community==0.3.17
langchain-core==0.3.84
langchain-ollama==0.2.2
langchain-text-splitters==0.3.11
langsmith==0.3.45
latex2mathml==3.81.0
lazy-loader==0.5
lxml==5.4.0
markdown-it-py==4.0.0
marko==2.2.2
MarkupSafe==2.1.5
marshmallow==3.26.2
mdurl==0.1.2
mmh3==5.2.1
mpire==2.10.2
mpmath==1.3.0
multidict==6.7.1
multiprocess==0.70.19
mypy_extensions==1.1.0
networkx==3.6.1
ninja==1.13.0
numpy==1.26.4
oauthlib==3.3.1
ollama==0.4.7
onnxruntime==1.25.1
opencv-python-headless==4.11.0.86
openpyxl==3.1.5
opentelemetry-api==1.41.1
opentelemetry-exporter-otlp-proto-common==1.41.1
opentelemetry-exporter-otlp-proto-grpc==1.41.1
opentelemetry-instrumentation==0.62b1
opentelemetry-instrumentation-asgi==0.62b1
opentelemetry-instrumentation-fastapi==0.62b1
opentelemetry-proto==1.41.1
opentelemetry-sdk==1.41.1
opentelemetry-semantic-conventions==0.62b1
opentelemetry-util-http==0.62b1
orjson==3.11.8
overrides==7.7.0
packaging==25.0
pandas==2.3.3
pillow==11.1.0
posthog==7.13.2
propcache==0.4.1
protobuf==6.33.6
psutil==7.2.2
pyclipper==1.4.0
pydantic==2.12.3
pydantic-settings==2.14.0
pydantic_core==2.41.4
pydub==0.25.1
Pygments==2.20.0
pypdfium2==4.30.0
PyPika==0.51.1
pyproject_hooks==1.2.0
python-bidi==0.6.7
python-dateutil==2.9.0.post0
python-docx==1.2.0
python-dotenv==1.0.1
python-multipart==0.0.27
python-pptx==1.0.2
pytz==2026.1.post1
PyYAML==6.0.3
referencing==0.37.0
regex==2026.4.4
requests==2.33.1
requests-oauthlib==2.0.0
requests-toolbelt==1.0.0
rich==15.0.0
rpds-py==0.30.0
rtree==1.4.1
ruff==0.15.12
safehttpx==0.1.7
safetensors==0.7.0
scikit-image==0.26.0
scipy==1.17.1
semantic-version==2.10.0
semchunk==3.2.5
setuptools==81.0.0
shapely==2.1.2
shellingham==1.5.4
six==1.17.0
soupsieve==2.8.3
SQLAlchemy==2.0.49
starlette==0.52.1
sympy==1.14.0
tabulate==0.10.0
tenacity==9.1.4
tifffile==2026.3.3
tokenizers==0.22.2
tomlkit==0.13.3
torch==2.11.0
torchvision==0.26.0
tqdm==4.67.1
transformers==4.57.6
tree-sitter==0.25.2
tree-sitter-c==0.24.2
tree-sitter-javascript==0.25.0
tree-sitter-python==0.25.0
tree-sitter-typescript==0.23.2
typer==0.12.5
typing-inspect==0.9.0
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2026.2
urllib3==2.6.3
uuid_utils==0.14.1
uvicorn==0.46.0
uvloop==0.22.1
watchfiles==1.1.1
websocket-client==1.9.0
websockets==14.2
wrapt==2.1.2
xlsxwriter==3.2.9
yarl==1.23.0
zipp==3.23.1
zstandard==0.23.0
EOF
```

**Step 6.2 — Install all packages**

```bash
pip install -r requirements.txt
```
> **What it does:** Reads `requirements.txt` line by line and installs every package at the exact pinned version. The `-r` flag means "read from a file".
>
> **Why this is reliable:** every package in the file is pinned, so pip's resolver makes no decisions. You get byte-for-byte the same environment that was verified working when the project was built. This is the difference between "should work" and "will work".
>
> **What to expect:** 5–15 minutes the first time. You will see a lot of output as pip downloads packages. Docling in particular downloads several supporting ML models on first install — that's normal.
>
> **If you see errors:** The most common cause on Apple Silicon is a package needing compilation when no prebuilt wheel is available for ARM64. If a specific package fails, note its name and look it up in the **Common Pitfalls** section below — most known issues have documented resolutions there. Do NOT re-run the whole `pip install` command if a single package fails — pip is idempotent and will pick up where it left off.

**Step 6.3 — (Optional, advanced) Regenerate `requirements.txt` after a deliberate upgrade**

```bash
# DO NOT run this after a fresh install — the shipped file is the source of truth.
# Only run this when you have INTENTIONALLY upgraded a package (e.g. via `pip install --upgrade gradio`)
# and want to capture the new resolved tree as the new canonical snapshot.
pip freeze > requirements.txt
```
> **What it does:** Writes the *exact* version of every package currently installed in the venv to `requirements.txt`, overwriting whatever was there before.
>
> **When to use it:** Only when you have deliberately upgraded one or more packages and verified the new versions work end-to-end. This is the upgrade path, not the install path. For fresh installs, skip this step entirely — the file already shipped in the project is what you want.

**Step 6.4 — Verify the key packages installed**

```bash
pip list | grep -E "docling|chromadb|langchain|gradio|ollama"
```
> **What it does:** Lists all installed packages and filters (`grep`) the output to show only the ones we care about most.  
> **`|` (pipe):** Sends the output of the left command into the right command as input. This is one of the most powerful concepts in terminal usage — chaining tools together.  
> **What you should see:** Each of the five package names listed with a version number matching what we pinned in `requirements.txt`. If any are missing, install them individually: `pip install <packagename>==<version>`.

---

## Phase 7 — Create the Configuration File

All settings — model names, file paths, chunk sizes — live in one single file. This means the IT admin maintaining the system can change any behaviour (e.g. swap to a different LLM) by editing one file, without touching any code.

**Step 7.1 — Create config.py**

```bash
cat > config.py << 'EOF'
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
EOF
```
> **What it does:** Creates `config.py` with every setting the chatbot needs, heavily commented.  
> **Why a separate config file:** If model names or paths were scattered across `ingest.py`, `chat.py`, and `app.py`, changing one model would require editing three files and risking a mistake. One config file = one place to make changes = far fewer errors in production.  
> **New in v1.2 — `OLLAMA_KEEP_ALIVE`:** Tells Ollama to keep models loaded in RAM for 24 hours. Without this, Ollama unloads any model after 5 minutes of inactivity. During ingestion you switch between the vision model (per image) and the embedding model (per chunk) — Ollama would repeatedly unload and reload both, adding minutes per document with no error message shown.  
> **New in v1.2 — `VISION_CONTEXT_CHARS`:** The describe_image function uses this to control how much surrounding document text is sent alongside an image to the vision model. Previously this was a hardcoded `500` buried inside a function — now it is tunable from config without touching code.  
> **New in v1.3 — `CHUNK_SIZE` / `CHUNK_OVERLAP`:** Raised from 1000/150 to 1800/200. The embedding model (`nomic-embed-text`) can accept roughly 30,000 characters per input — 1000-char chunks were far below its capacity and fragmented long specifications and step-by-step procedures across multiple chunks, degrading retrieval quality. Larger chunks mean fewer, richer results per query.  
> **New in v1.4 — `VISION_MODEL` = `CHAT_MODEL`:** `llama3.2-vision:11b` removed. The multimodal Gemma 4 model handles both image description and chat in one. `VISION_MODEL` is kept as a named alias rather than deleted, so the code stays self-documenting and the config structure is preserved if the models are ever split again. (v2.0 changes the *value* of `CHAT_MODEL` from `gemma4:26b` to `gemma4:e4b` — the alias relationship is unchanged.)  
> **New in v2.5 — `HISTORY_TURNS = 6` (back-ported from v1.5):** Conversation memory was implemented in `src/chat.py` in v1.5 — `format_history()` reads `config.HISTORY_TURNS` to decide how many recent messages to splice into the prompt. The matching `HISTORY_TURNS = 6` line was added to v1.5's changelog narrative but never made it into v1's `config.py` heredoc, so v2 inherited a `chat.py` that referenced a constant that didn't exist. The first user question worked (no history → early return in `format_history()`); the first follow-up question crashed with `AttributeError: module 'config' has no attribute 'HISTORY_TURNS'`. Fixed in v2.5 by adding the constant to the canonical heredoc above. See **U-30** in `UPDATES_NEEDED.md` for the full root-cause write-up. v1's `config.py` likely carries the same latent gap — back-port from a v1 session if needed.

---

## Phase 7.5 — Create the Shared Logging Module

Before writing the main scripts, we create a small shared logging module that all scripts will import. This replaces ad-hoc `print()` calls with structured, timestamped log entries that are written to both the terminal and a rotating log file on disk.

**Why this matters:** When something fails in production six months from now, `print()` output is gone — it only ever appeared in the terminal that ran the command. A rotating log file gives the IT admin a permanent record of every run, every warning, and every error. The `RotatingFileHandler` caps the file at 5MB and keeps 5 backups, so logs never fill the disk.

**Step 7.5.1 — Create src/logging_setup.py**

```bash
cat > src/logging_setup.py << 'PYEOF'
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
PYEOF
```
> **What it does:** Creates `src/logging_setup.py` — a shared logging configuration used by all three main scripts.  
> **`RotatingFileHandler`:** A Python standard library class that automatically manages log file size. When `chatbot.log` reaches 5MB it is renamed to `chatbot.log.1`, a fresh `chatbot.log` is started, and files older than `chatbot.log.5` are deleted. No manual log cleanup needed.  
> **`__name__` explained:** When Python runs `src/ingest.py`, the value of `__name__` inside that file is `'src.ingest'`. Passing it to `setup_logging()` means every log line from ingest.py is tagged `[src.ingest]` — making it easy to filter the log file by script.

**Step 7.5.2 — Verify the logging module is importable**

```bash
python -c "from src.logging_setup import setup_logging; log = setup_logging('test'); log.info('Logging works'); print('OK')"
```
> **What it does:** Imports `logging_setup` and writes one test log line.  
> **What you should see:** A timestamped `[INFO]` line in the terminal, `OK` printed, and a new `logs/chatbot.log` file created.

---

## Phase 8 — Build the Ingestion Pipeline

The ingestion script is the "loader" — it reads documents from the `ingest/` folder, extracts text and images, describes images with the vision model, converts everything to embeddings, and stores it all in ChromaDB.

**Step 8.1 — Create src/ingest.py**

```bash
cat > src/ingest.py << 'PYEOF'
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
PYEOF
```
> **What it does:** Creates the full ingestion pipeline script in `src/ingest.py`.  
> **Key changes from v1.0/v1.2:**
> - **Docling pipeline options** — `generate_picture_images=True` is now explicitly set. Previously this was silently off, meaning all chart/diagram images were skipped with no error.
> - **Correct Docling traversal** — uses `doc.iterate_items()` with `isinstance(element, PictureItem)` which matches Docling's actual API.
> - **Deterministic chunk IDs** — `make_chunk_id()` generates a hash-based ID per chunk. Re-running ingest on an already-ingested document now overwrites existing chunks instead of duplicating them.
> - **Move-to-processed** — after each file is ingested, it is moved to `ingest/processed/`. This prevents accidental double-ingestion and provides a visible audit trail.
> - **`keep_alive`** — passed to the direct `ollama_client` (vision calls) and `ChatOllama` (Phase 9) so models stay resident. Note: `OllamaEmbeddings` does NOT accept `keep_alive` — its Pydantic schema forbids extra fields. The parameter is omitted from `OllamaEmbeddings` in the final script (see Known Fixes below).
> - **Logging** — all `print()` replaced with `log.info/warning/error()`. Output goes to both terminal and `logs/chatbot.log`.
> - **Post-ingest image assertion** — prints a warning if zero image-description chunks were created, making the "vision pipeline is broken" case visible immediately.
> - **New in v1.3 — Two-stage chunking (U-16 + U-17):** Documents are first split on markdown section headings (`MarkdownHeaderTextSplitter`), then each section is cut into 1800-character chunks (`RecursiveCharacterTextSplitter`). The heading path (h1/h2/h3) travels as metadata on every chunk, so a query like "torque spec for input shaft" can match the right section even if the heading itself isn't in the chunk body. Chunk size raised from 1000 → 1800 to keep multi-step procedures and spec tables intact.
>
> **Why the heredoc above looks the way it does — Known Issues already baked into the canonical source:**
>
> The heredoc above is the post-fix canonical state. The two notes below explain the underlying behaviour so a student or future maintainer who wonders "why this line and not the obvious one?" can understand the rationale without having to re-discover it. These are not pending fixes — they are already applied in the heredoc itself.
>
> **Note 1 — why `importlib.metadata.version("docling")` and not `docling.__version__` (resolved in plan v2.4, 2026-05-07).**  
> `docling` 2.14.0 does not expose a `__version__` attribute on its top-level module — unlike `langchain`, `chromadb`, and `gradio`, which do. An earlier draft of this script tried `docling.__version__` and crashed on every run with `AttributeError: module 'docling' has no attribute '__version__'`. The heredoc now uses `importlib.metadata.version("docling")` (a Python standard library call that reads the installed package metadata and works for any installed package regardless of whether it exposes `__version__`) and the bare `import docling` was removed because nothing else in the file needed it. The submodule imports (`from docling.document_converter import …`, etc.) are absolute imports — Python initialises the `docling` package automatically when any submodule of it is imported, so removing the bare top-level import does not affect the rest of the script. **First verified working on Andrew's 16GB dry-run machine on 2026-05-07.**
>
> **Note 2 — why `keep_alive` is NOT passed to `OllamaEmbeddings` (resolved in plan v1.5.1).**  
> `langchain-ollama` 0.2.2's `OllamaEmbeddings` class uses `extra_forbidden` Pydantic validation, meaning it raises a `ValidationError` if any parameter not in its schema is passed. `keep_alive` is not in its schema. The parameter is therefore omitted from the `OllamaEmbeddings` constructor only — it is still correctly passed to `ollama_client.chat()` (vision model calls in this script) and to `ChatOllama` in Phase 9. During an active ingest run, the embedding model is called back-to-back for every chunk so Ollama will not unload it due to inactivity.

---

## INGESTION TEST — Validate the Pipeline Before Proceeding

**Do not skip this step.** Phase 9 (chat.py) and Phase 10 (app.py) both read from ChromaDB. If ingestion is broken, the chatbot will appear to run but will return empty or hallucinated answers with no obvious error. It is far easier to debug one layer at a time than to troubleshoot the entire stack at once.

**Step 8T.1 — Drop a test document into the ingest folder**

Copy any PDF into the `ingest/` folder. Ideally choose one that contains at least one chart, diagram, or embedded image — this exercises the full pipeline including the vision model. A technical manual, product datasheet, or spec sheet all work well.

```bash
ls ingest/
```
> **What it does:** Confirms your file is in the right place before running.  
> **What you should see:** Your PDF filename listed. If the folder is empty, the script will exit with a "No supported files found" message and nothing will be ingested.

**Step 8T.2 — Run the ingestion script**

```bash
# Make sure you are in the PROJECT ROOT before running — not inside src/
# Your prompt should show: (chatbot-16G-venv) .../Chatbotv2 %
# If it shows .../Chatbotv2/src, run: cd ..

python src/ingest.py
```
> **What it does:** Runs the full pipeline — parses the document with Docling, sends any images to gemma4:e4b for plain-English descriptions, converts all text to embeddings via nomic-embed-text, and stores everything in ChromaDB.  
> **IMPORTANT — run from the project root, not from inside `src/`:** The script resolves all paths (ingest/, chroma_db/, logs/) relative to `config.py`, which sits at the project root. Running from inside `src/` will cause path errors.  
> **What to expect:** Several minutes for the first run. Models need to warm up in Ollama, Docling performs layout analysis and OCR, and the embedding model processes every chunk one by one. A 10-page PDF with a few images may take 3–8 minutes. This is normal.  
> **ChromaDB telemetry warnings are harmless:** You may see lines like `Failed to send telemetry event ClientStartEvent`. ChromaDB attempts to send anonymous usage data and fails when there is no internet route. This does not affect any functionality — ignore it.

**Step 8T.3 — Read the output and verify these five things**

```
✅ 1. Version banner appears at the top
      Look for lines like:
        [ENV] Python    : 3.12.x (Darwin arm64)
        [ENV] LangChain : 0.3.18
        [ENV] ChromaDB  : 0.5.20
        [ENV] Docling   : 2.14.0
        [ENV] Ollama    : ollama version is X.X.X
      If Ollama shows a warning instead of a version number,
      Ollama is not running. Fix: open a new terminal and run:
        ollama serve

✅ 2. Non-zero chunk count
      Look for a line like:
        Text: 42 chunks from 38,210 characters (via 2-stage split)
      Any non-zero number is good. Zero means Docling extracted no text —
      check that the file is a real text-based PDF, not a scanned image
      with no OCR output.

✅ 3. Vision model fires (only if your PDF has images/charts)
      Look for:
        Image 1: sending to vision model...
      If your document has diagrams but this line never appears, the image
      extraction pipeline is not working. The post-ingest warning will also
      flag this: "No image descriptions were generated."

✅ 4. File moves to ingest/processed/
      Look for:
        Moved yourfile.pdf → ingest/processed/
      This confirms the idempotency guard is working. The original ingest/
      folder should now be empty.

✅ 5. Success line at the very end
      Look for:
        SUCCESS: XX chunks stored in ChromaDB.
        Collection 'enterprise_kb' in .../chroma_db
      This confirms ChromaDB accepted the data and wrote it to disk.
```

**Step 8T.4 — Confirm ChromaDB has data on disk**

```bash
ls -lh chroma_db/
```
> **What it does:** Checks that ChromaDB actually wrote its files.  
> **What you should see:** A non-empty folder with several files inside. ChromaDB writes binary index files here automatically. An empty folder after a "SUCCESS" run would indicate a silent write failure — in practice this does not happen, but a five-second check removes all doubt.

**Step 8T.5 — Check the persistent log file**

```bash
tail -30 logs/chatbot.log
```
> **What it does:** Shows the last 30 lines of the rotating log file written to disk.  
> **What to look for:** No `[ERROR]` lines. `[WARNING]` lines are acceptable if they relate to images with no caption — that is not a failure. A `[WARNING]` saying the Ollama version could not be queried is a failure and must be resolved before proceeding.  
> **Why bother:** The terminal output and log file should be identical. If they differ, a logging handler is misconfigured. Catching this now means your production log file can be trusted.

**If all five checks above pass:** proceed to Phase 9.  
**If anything is wrong:** paste the full terminal output here and diagnose before writing any more code.

---

## Phase 9 — Build the RAG Query Engine

The chat module handles everything that happens when a user sends a message: embed the question, search ChromaDB, build a prompt with retrieved context, call the LLM, return the answer.

**Step 9.1 — Create src/chat.py**

```bash
cat > src/chat.py << 'PYEOF'
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
PYEOF
```
> **What it does:** Creates `src/chat.py` — the RAG query engine that is called every time a user sends a message.  
> **Key changes from v1.0/v1.2:**
> - **`keep_alive`** — added to both `OllamaEmbeddings` and `ChatOllama`. Previously, the embedding and chat models could be unloaded between queries, causing a slow reload on each user message.
> - **Logging** — initialised via `setup_logging`. All function calls and query details are now written to `logs/chatbot.log` alongside terminal output.
> - **`temperature=0.1`** — unchanged and important. This setting keeps the LLM factual and grounded. For a technical knowledge base you do not want creativity.
> - **New in v1.3 — LCEL chain (U-07):** `RetrievalQA.from_chain_type` replaced with an LCEL pipeline using the `|` pipe operator. LCEL is the current LangChain standard, removes deprecation warnings, and is required for streaming support.
> - **New in v1.3 — `query_stream()` (U-05):** Generator function that yields partial answers token-by-token using `qa_chain.stream()`. Used by `app.py`'s chat handler so responses appear progressively. Also the key prerequisite for any future voice-output (TTS) feature.
> - **New in v1.5 — Conversation memory:** `RAG_PROMPT_TEMPLATE` gains a `{history}` block; new `format_history()` helper trims to `HISTORY_TURNS` and converts Gradio's messages-format list into a plain-text conversation block; `qa_chain` gains a `"history"` branch; `query()` and `query_stream()` both accept an optional `history` argument. `query_stream()` logs the formatted history block at DEBUG level so we can verify in `logs/chatbot.log` that prior turns actually reached the LLM during follow-up-question testing. Memory lives only in the user's browser tab — the server never remembers it between sessions.

---

## Phase 10 — Build the Gradio Chat Interface

Gradio turns our Python functions into a browser-accessible chat interface with no web development required.

**Step 10.1 — Create src/app.py**

```bash
cat > src/app.py << 'PYEOF'
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
PYEOF
```
> **What it does:** Creates `src/app.py` — the Gradio UI. Running this file starts the chatbot web interface.  
> **Key changes from v1.0/v1.2:**
> - **`log_versions()`** — logs Python, Gradio, LangChain, ChromaDB, and Ollama versions at startup for forensics.
> - **Logging** — all `print()` calls replaced with `log.info()`. Startup details and per-message activity are written to `logs/chatbot.log`.
> - **`share=False`** — unchanged and important. Keeps the chatbot local. Setting this to `True` would create a public URL accessible from the internet — never do this with an enterprise knowledge base.
> - **New in v1.3 — `type="messages"` (U-08):** The `gr.Chatbot()` widget now uses the Gradio 5 messages format (`{"role": ..., "content": ...}` dicts). This removes the Gradio 5 deprecation warning that appeared with every chat message using the old `[user, assistant]` tuple format.
> - **New in v1.3 — Streaming `chat_handler` (U-05):** The handler is now a Python generator using `yield` instead of a regular function with `return`. Gradio detects this automatically and streams tokens to the UI as the LLM produces them. The user sees the first words within ~1–2 seconds instead of waiting up to 60 seconds for the full response.
> - **New in v1.5 — Conversation memory pass-through:** `chat_handler` now slices `history[:-2]` (excluding the just-appended user message and empty assistant placeholder) and passes it into `query_stream()` so the LLM can resolve follow-up questions and pronouns. The Gradio `Chatbot` widget already keeps history per browser tab — opening a second tab gives you a completely independent conversation. Nothing is persisted server-side; closing the tab discards the conversation entirely.
> - **v1.5.1 (2026-05-03) — Gradio version pin bumped:** No code edits to `app.py` itself — but the Phase 6 requirements pin moved from `gradio==5.13.0` to `gradio==5.50.0` (with explicit `gradio_client==1.14.0`). The 5.13.0 release shipped `gradio_client` 1.6.0 which crashed at startup with `TypeError: argument of type 'bool' is not iterable` when Pydantic 2.13.3 generated `additionalProperties: True` JSON schemas. After the upgrade `python src/app.py` launches cleanly, the `/info` API endpoint generates correctly, and the misleading downstream `share=True` error message disappears. **Note:** if a future Gradio sub-release deprecates anything used here (`gr.Chatbot(type="messages")`, `Blocks`, the `submit/click` wiring, or the streaming generator pattern), watch for `DeprecationWarning` in `logs/chatbot.log` at startup and update accordingly.

---


## Phase 11 — Running the System after install complete and working

**Step 11.1 — Start a new terminal as root and activate the environment**

```bash
cd /Users/*YOUR USER NAME*/RAGbot
source chatbot-16G-venv/bin/activate
```
> **What it does:** Navigates to the project and activates the virtual environment.  
> **Reminder:** You must do this every time you open a new terminal. The `(chatbot-16G-venv)` prefix in your prompt confirms it is active.

**Step 11.2 — Set Ollama keep-alive before starting**

```bash
export OLLAMA_KEEP_ALIVE=24h
```
> **What it does:** Tells Ollama to keep all loaded models in RAM for 24 hours rather than the default 5-minute timeout.  
> **Why before ingestion:** During ingestion the system alternates between calling the vision model (once per image) and the embedding model (once per chunk). Without this setting, Ollama unloads each model after 5 minutes of inactivity and has to reload it — adding minutes per document with no error message. Setting this once at the start of your work session prevents the issue entirely.  
> **Note:** This only applies to the current terminal session. For a permanent setting on macOS run: `launchctl setenv OLLAMA_KEEP_ALIVE 24h` then restart Ollama.

**Step 11.3 — Drop a test document into the ingest folder**

Copy any PDF, DOCX, MD, or TXT file into:
```
/Users/andrew-m4/Documents/Claude/Projects/Chatbot/ingest/
```
> **Tip for testing:** Use a document you know the content of — so you can verify the chatbot's answers are actually grounded in the document and not guessed.

**Step 11.4 — Run the ingestion script**

```bash
python src/ingest.py
```
> **What it does:** Processes every document in `ingest/`, extracts text and images, describes images using `gemma4:e4b` (multimodal — handles vision and chat), embeds all chunks using `nomic-embed-text`, and stores everything in ChromaDB.  
> **What to watch for in v1.3:**
> - A version banner at startup listing Python, LangChain, ChromaDB, Docling, and Ollama versions
> - `Text: N chunks from M characters (via 2-stage split)` — confirming the two-stage chunker ran. Chunk count will be lower than v1.2 due to the larger 1800-char chunk size.
> - `Image N: sending to vision model...` lines for each image found
> - `Image-description chunks created: N` — if this is 0 for a document that has charts, something is wrong with the Docling pipeline setup
> - Processed files moved to `ingest/processed/` when complete
> - All output also written to `logs/chatbot.log`
>
> **Optional — verify heading metadata made it into ChromaDB** (run after ingest, with venv active):
> ```bash
> python -c "
> import sys; sys.path.insert(0, '.')
> from src.chat import vector_store
> docs = vector_store.similarity_search('any term from your document', k=1)
> print(docs[0].metadata)
> "
> ```
> The output should include `h1`, `h2`, and/or `h3` keys alongside `source` and `chunk_type`. If those keys are absent, the document may not have markdown-style headings — plain-text files and some PDFs export as unstructured prose, which is fine; chunking still works, just without heading metadata.

**Step 11.5 — Start the chatbot**

```bash
python src/app.py
```
> **What it does:** Starts the Gradio web server and launches the chat interface.  
> **What to expect:** The terminal will print a timestamped startup banner and `Running on local URL: http://0.0.0.0:7860`. Leave this terminal window open — closing it stops the chatbot.

**Step 11.6 — Open the chat interface**

Open your browser and go to:
```
http://localhost:7860
```
> **What it does:** Opens the Gradio chat UI in your browser. You should see the chat window and a green status line confirming the number of chunks loaded.

**Step 11.7 — Test with known questions**

Ask the chatbot something you know is in your test document. Good test questions:
- A specific fact from the document
- A request to summarise a section
- A question about a chart or graph (if your test PDF has one)
- A question about something *not* in the document (the system should say it doesn't know)

**Step 11.8 — Ending the RAG retrieval process after testing/use**

When you are done testing or using the chatbot for a session, terminate the running process cleanly with **Ctrl-C** in the terminal where `app.py` is running.

> **What `app.py` actually is:** Despite the simple name, `python src/app.py` is doing two jobs at once in a single Python process. (1) It runs the **Gradio web server** that serves the chat webpage at `http://localhost:7860` so your browser has something to talk to. (2) It runs the **RAG retrieval pipeline** — every time you send a message in the browser, the same Python process embeds your question with `nomic-embed-text`, searches ChromaDB for matching chunks, builds the prompt, and streams tokens out of `gemma4:e4b` via Ollama. So when you stop `app.py`, you are stopping both the webpage AND the retrieval engine. The browser tab will go blank or show "site can't be reached" — that is expected.  
> **Why Ctrl-C and NOT Ctrl-Z:** Ctrl-C sends a SIGINT signal — Python and Gradio listen for it and shut the process down gracefully (web server stops, port 7860 is released, the rotating log file is closed, the Python process exits). Ctrl-Z sends a different signal (SIGTSTP) that *suspends* the process instead of stopping it. The process is still alive, frozen in place, still holding port 7860 and ~10 GB of RAM for the loaded model. If you accidentally Ctrl-Z, see Pitfall 10 for how to recover.  
> **What you will see after Ctrl-C:** the terminal prints one or two interrupt-handler lines (Gradio's "Keyboard interruption in main thread... closing server.") and returns you to your shell prompt. `ps` no longer lists the `python src/app.py` process. Port 7860 is now free for the next start.  
> **Restarting:** When you want the chatbot back, re-run `python src/app.py` from the same terminal (with the venv still active). The first answer after a long idle period may take 5–15 seconds extra because Ollama may need to reload `gemma4:e4b` into RAM.

**Step 11.9 — Check the log file**

```bash
tail -50 logs/chatbot.log
```
> **What it does:** Shows the last 50 lines of the rotating log file.  
> **Why check this:** The log file is your permanent record of every run. It shows version banners, ingestion progress, query activity, and any warnings or errors. If anything seems wrong with the chatbot's behaviour, this is the first place to look.


-----

## Phase 12 — Day-to-Day Usage (Operating Guide)

### Adding new documents to the knowledge base

1. Copy documents into `ingest/`
2. Activate the venv: `source chatbot-16G-venv/bin/activate`
3. Run: `python src/ingest.py`
4. Restart `app.py` if it is running (stop with `Ctrl+C`, then `python src/app.py`)

**Note:** Processed files are automatically moved to `ingest/processed/` after ingestion. Re-running `ingest.py` on an already-ingested document is safe — deterministic chunk IDs mean existing chunks are overwritten, not duplicated.

### Starting the chatbot after a reboot

```bash
cd /Users/* YOUR USER NAME */RAGbot
source chatbot-16G-venv/bin/activate
python src/app.py
```

### Changing the chat model

Edit `config.py`, change the `CHAT_MODEL` value to any model name from `ollama list`, save the file, restart `app.py`. No other files need to change.

### Checking what is in the knowledge base

```bash
source chatbot-16G-venv/bin/activate
python -c "
import sys; sys.path.insert(0,'.')
from src.chat import check_knowledge_base
print(f'Chunks in knowledge base: {check_knowledge_base()}')
"
```

### Wiping and rebuilding the knowledge base

```bash
rm -rf chroma_db/
```
> **What it does:** Deletes the entire ChromaDB data folder. The next `python src/ingest.py` will rebuild from scratch.  
> **When to do this:** If you want to remove documents or if the database gets corrupted.  
> **Before wiping:** Move any documents you want to re-ingest from `ingest/processed/` back to `ingest/` first.

### Reading the logs

```bash
tail -100 logs/chatbot.log          # last 100 lines
grep "ERROR\|WARNING" logs/chatbot.log   # show only problems
grep "Query received" logs/chatbot.log   # show all user questions
```

### Checking Ollama model status during ingestion

```bash
ollama ps
```
> **What it does:** Lists which models are currently loaded in Ollama's memory.  
> **What to look for:** During ingestion, both models (`nomic-embed-text` and `gemma4:e4b`) should appear in the loaded list. If a model disappears between documents, the `OLLAMA_KEEP_ALIVE` setting may not have taken effect — see Phase 11, Step 11.2.

-----


## Appendix A — Project File Map

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

-----

## Appendix B — Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | `src/__init__.py` missing | Run `touch src/__init__.py` from project root |
| `ModuleNotFoundError` (other) | venv not activated | Run `source chatbot-16G-venv/bin/activate` |
| `Connection refused` on Ollama | Ollama not running | Run `ollama serve` in a separate terminal |
| Model not found error | Model not pulled | Run `ollama pull nomic-embed-text` or `ollama pull gemma4:e4b` as needed |
| Empty knowledge base warning | `ingest.py` not run yet | Drop a file in `ingest/` and run `python src/ingest.py` |
| `Image-description chunks created: 0` | Docling image pipeline not firing | Verify `pdf_pipeline_options.generate_picture_images = True` in ingest.py |
| Slow ingestion with model reload delays | `OLLAMA_KEEP_ALIVE` not set | Run `export OLLAMA_KEEP_ALIVE=24h` before ingestion |
| Duplicate chunks after re-running ingest | Ingesting already-processed files | Files in `ingest/processed/` are already done; chunks use deterministic IDs so re-ingest is safe |
| Answers seem wrong | Wrong chunks retrieved | Increase `RETRIEVAL_TOP_K` in `config.py` (try 8) |
| Port 7860 already in use | Another Gradio instance running | Kill it: `lsof -ti:7860 \| xargs kill` |
| No log file created | `logs/` folder missing | Run `mkdir logs` from project root, or re-run Step 3.2 |


-----

## Appendix C  Common Pitfalls (Reference)

This section is a living document of every gotcha encountered during the original buildout. If something looks wrong — read here first before you start debugging. Each entry has the same structure: **Symptom → Cause → Fix.**

### Pitfall 1 — Shell prompt missing the `(chatbot-16G-venv)` indicator

**Symptom:** You run a `pip` or `python` command and see one of:
- "command not found"
- A version of Python that isn't 3.12
- `pip freeze` output containing only a few system packages (`mlx`, `rpm`, `wheel`)
- An import error for a chatbot dependency that you definitely installed

**Cause:** The Python virtual environment is not activated in the current terminal. Virtual-environment activation is **per-shell-session** — it doesn't carry across new terminal windows or tabs. When the venv is not active, `python` and `pip` refer to your system-wide installation, which doesn't see anything we installed for the chatbot.

**Fix:** From the project root, run:
```bash
source chatbot-16G-venv/bin/activate
```
Verify your prompt now shows `(chatbot-16G-venv)` at the start of the line. For example: `(chatbot-16G-venv) andrew-m4@Andrew-M4s-Mac-mini Chatbotv2 %`. If it does, you're good.

**Belt-and-braces verification** — the `which python` check confirms what activated:
```bash
which python
```
This should print a path that ends with `chatbot-16G-venv/bin/python`. If it instead prints `/usr/bin/python3` or `/opt/homebrew/bin/python3.12`, the activate command did not take effect — try again from the project root and confirm the `(chatbot-16G-venv)` prefix appears.

**Make `source chatbot-16G-venv/bin/activate` the first command in every fresh terminal you open for chatbot work.**

---

### Pitfall 2 — `Failed to send telemetry event` warnings on every startup

**Symptom:** Every time you run `python src/ingest.py` or `python src/app.py`, you see one or more lines like:
```
Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
Failed to send telemetry event CollectionQueryEvent: ...
```

**Cause:** ChromaDB tries to send anonymous usage statistics via the `posthog` library on startup. There is a long-standing version-mismatch bug between ChromaDB 0.5.x's telemetry call and certain `posthog` versions. The warning is harmless — ChromaDB itself works fine, the only thing that "fails" is the analytics call we don't care about.

**Fix:** **Ignore the warnings.** They are cosmetic. They do not affect ingestion, retrieval, or chat behaviour. If you want to silence them entirely, set the environment variable `ANONYMIZED_TELEMETRY=False` before running any chatbot command, e.g.:
```bash
ANONYMIZED_TELEMETRY=False python src/app.py
```
This disables ChromaDB's telemetry attempt entirely so the warnings never print. Adding this to a wrapper script or to your shell profile makes it permanent.

---

### Pitfall 3 — `pydantic_core._pydantic_core.ValidationError: 1 validation error for OllamaEmbeddings — keep_alive: Extra inputs are not permitted`

**Symptom:** Running `python src/ingest.py` or `python src/app.py` crashes immediately at startup with the above traceback. Nothing else gets a chance to run. The crash happens at module-load time on the line `embedding_function = OllamaEmbeddings(...)`.

**Cause:** `langchain-ollama` 0.2.2 uses Pydantic with `extra_forbidden`, meaning it rejects any constructor argument outside its defined schema. `keep_alive` is supported by `ChatOllama` and the direct `ollama` client, but **not** by `OllamaEmbeddings`. If your `src/chat.py` or `src/ingest.py` passes `keep_alive=...` into `OllamaEmbeddings(...)`, you'll hit this error.

**Fix:** Open the offending file (`src/chat.py` or `src/ingest.py`) and remove the `keep_alive=config.OLLAMA_KEEP_ALIVE,` line from inside the `OllamaEmbeddings(...)` block. Leave it intact on `ChatOllama(...)` (line further down in `chat.py`) and on the direct `ollama.chat(...)` calls in `describe_image()` (in `ingest.py`) — those both accept it correctly. With v1.6 of this plan, the heredocs already exclude this argument, so a fresh install via the pinned `requirements.txt` should never hit this. If you see it, you've drifted from the canonical setup.

---

### Pitfall 4 — Gradio UI fails to load with `TypeError: argument of type 'bool' is not iterable`, followed by `When localhost is not accessible, a shareable link must be created. Please set share=True...`

**Symptom:** `python src/app.py` prints all the startup banners successfully and reports `Running on local URL: http://0.0.0.0:7860`, but then crashes on the first HTTP request with the bool/iterable TypeError followed by a misleading `share=True` suggestion.

**Cause:** `gradio_client` 1.6.x had a JSON schema parser that didn't handle `additionalProperties: True` schemas (which Pydantic 2.13.x emits in some cases). The fix landed in `gradio_client` 1.7.0+. The `share=True` suggestion is a **red herring** — Gradio's diagnostic incorrectly concludes "localhost unreachable" after the schema parser crashes.

**Fix:** **Do NOT set `share=True`** — that would expose your private knowledge base to the public internet. Instead, upgrade Gradio:
```bash
pip install --upgrade "gradio<6.0"
```
Then restart `app.py`. The pinned version in v1.6 of this plan (`gradio==5.50.0`, `gradio_client==1.14.0`) already includes the fix, so a fresh install via the pinned `requirements.txt` won't hit this. If you see it, your venv has drifted — check `pip show gradio_client` and verify you're on 1.7.0 or higher.

---

### Pitfall 5 — `Error: could not connect to ollama app, is it running?` when running ingest or chat

**Symptom:** When `src/ingest.py` or `src/chat.py` first try to call out to Ollama for embeddings or chat, you get a connection-refused error.

**Cause:** Ollama's background service (the `ollama serve` daemon) is not running. The chatbot makes HTTP requests to `http://0.0.0.0:11434` for every embedding and every LLM call — if there's no server listening on that port, every request fails.

**Fix:** Start Ollama. On macOS, the simplest way is to open the Ollama application from `/Applications` — it runs as a menu-bar app and starts the daemon automatically. To verify it's now running:
```bash
ollama list
```
…should print a table without the connection-refused error. On Linux, run `ollama serve` in a separate terminal (or set up a systemd service to start it at boot).

---

### Pitfall 6 — `pip install` fails partway through with a compilation error on Apple Silicon

**Symptom:** During `pip install -r requirements.txt`, a specific package fails with errors about missing compilers, missing headers, or "wheel could not be built". Common offenders are `chroma-hnswlib`, `torch`, or `opencv-python-headless`.

**Cause:** The package doesn't have a prebuilt wheel for your specific platform/Python combination, so pip is trying to compile it from source — and your machine is missing the C/C++ build tools, Rust toolchain, or system headers it needs.

**Fix:** On macOS, install the Xcode Command Line Tools (one-time, ~3 GB):
```bash
xcode-select --install
```
A GUI installer will pop up; click through it. After it completes, re-run `pip install -r requirements.txt` — pip is idempotent and will skip the packages that already installed, retrying only the failed ones.

If a specific package still fails after Command Line Tools are installed, try installing only that package by itself with verbose output to see the actual compiler error:
```bash
pip install --verbose <package-name>==<version>
```
The error message will then tell you what's actually missing.

---

### Pitfall 7 — Knowledge base appears empty in the UI even though you ran ingest.py

**Symptom:** The Gradio UI status banner says `⚠️ Knowledge base is empty. Drop documents into the ingest/ folder and run: python src/ingest.py` even though you definitely ran ingest.py and saw success messages.

**Cause:** Almost always one of three things:
1. `ingest.py` and `app.py` are using different `CHROMA_DIR` paths (the Chroma database lives somewhere `app.py` isn't looking).
2. The venv used for ingestion was different from the venv `app.py` is now running in (different ChromaDB versions can be incompatible at the storage-format level).
3. ChromaDB telemetry warnings made the actual error invisible — scroll up further in the terminal output from `ingest.py` to see if there's a real error buried among the telemetry warnings.

**Fix:** Run the diagnostic command:
```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.chat import vector_store
print('Chunk count:', vector_store._collection.count())
"
```
- If this prints a non-zero count, ChromaDB has data. The status banner in `app.py` may have a stale cache — restart `app.py` (`Ctrl+C` and re-run `python src/app.py`).
- If this prints 0, ingestion did not actually populate the database. Re-run `python src/ingest.py` from the same venv and watch the output carefully for any error messages between the telemetry noise.

---

### Pitfall 8 — `pip freeze > requirements.txt` produces only 3 lines (`mlx`, `rpm`, `wheel`)

**Symptom:** You ran `pip freeze > requirements.txt` to refresh the pinned dependency file and the resulting file is essentially empty.

**Cause:** Same as Pitfall 1 — the venv was not active when you ran the command, so `pip freeze` reported the system-wide Python's packages (which is a tiny set) instead of the venv's full dependency tree.

**Fix:** Activate the venv first, verify the prompt shows `(chatbot-16G-venv)`, then re-run:
```bash
source chatbot-16G-venv/bin/activate
pip freeze > requirements.txt
wc -l requirements.txt   # Should be 180+ lines for a healthy pinned set
```

### Pitfall 9 — Mac feels frozen / Gradio UI hangs / fans run hard during ingest or chat (16GB-specific)

**Symptom:** During ingestion, or partway through a chat answer, the entire Mac becomes slow or unresponsive. The Gradio UI stops streaming. The cooling fan ramps up. Activity Monitor shows the disk being hit hard. Eventually things recover, but every interaction takes 5–30× longer than expected.

**Cause:** Memory pressure → swap thrashing. On a 16GB machine, the combined working set of macOS + Chrome/Safari + an IDE + Gemma 4 e4b + nomic-embed-text + Python + ChromaDB can exceed 16GB. macOS responds by paging memory to the SSD ("swap"), but inference reads model weights randomly, so every read becomes a disk read. Performance falls off a cliff.

**Diagnosis:**
```bash
# Show memory and swap state
vm_stat | head -20

# Show top memory consumers in real time (q to quit)
top -o mem
```
Look for: **"Pageouts"** in `vm_stat` rising rapidly (anything over a few thousand per minute means active swapping), and the `top` output showing Ollama (or the gemma4 process) using 8–12 GB while Chrome/Slack/IDEs eat the rest.

**Fix (in order of how invasive each is):**

1. **Close other RAM-heavy apps first.** Quit (don't just hide) browsers with many tabs, IDEs, virtual machines, video editors. This is by far the most effective fix and costs nothing. After quitting, run `vm_stat` again — pageouts should slow or stop.
2. **Lower `OLLAMA_KEEP_ALIVE` from `24h` to `5m`** in `config.py`. The `24h` setting is great when you have RAM to spare (v1 does), but on 16GB it pins ~10 GB of the model in RAM even when you're not actively using it. With `5m`, Ollama unloads the model after 5 minutes of inactivity. The trade-off is that the *next* query after a quiet period will be slow (the model has to reload from disk, ~10–20 seconds), but background memory use drops dramatically.
3. **Process documents one at a time during ingest.** If you dropped 50 PDFs into `ingest/` at once, ingestion holds the vision model resident across all of them and competes with macOS for RAM. Do them in batches of 5–10 and let the system breathe between batches.
4. **As a last resort, accept the upgrade.** Genuinely heavy workloads (large multi-document ingest, long chat sessions with several browser tabs open) are happier on 24GB+. A 16GB Mac can do this project — that's the whole point of v2 — but the upper end of the workload is closer to the ceiling than v1's 64GB headroom.

### Pitfall 10 — `zsh: suspended  python src/app.py` after pressing Ctrl-Z

**Symptom:** You meant to stop the chatbot but pressed **Ctrl-Z** instead of **Ctrl-C**. The terminal prints `zsh: suspended  python src/app.py` and gives you a shell prompt back. You assume the process is dead — but `ps` still shows it (e.g. `40761 ttys000  0:13.61 ... python src/app.py`), and the next time you try `python src/app.py` you get `OSError: [Errno 48] Address already in use` because port 7860 is still bound by the suspended copy. On a 16GB machine you may also feel sluggishness because the suspended process still holds ~10 GB of model weights in RAM (this is also a trigger pattern for Pitfall 9).

**Cause:** Ctrl-Z sends the **SIGTSTP** signal, which means "suspend the foreground job" — it pauses the process and gives the shell prompt back, but does NOT terminate the process. The process is frozen in place, still holding all of its memory, open files, and bound network ports. Ctrl-C, by contrast, sends **SIGINT** ("interrupt"), which Gradio handles by shutting down cleanly. The two key combos look almost identical on the keyboard but do completely different things.

**Fix (pick whichever is easiest in the moment):**

1. **`fg` then Ctrl-C.** Type `fg` at the prompt and press Return — this resumes the suspended job in the foreground (Gradio will start serving requests again immediately). Then press **Ctrl-C** so Gradio can shut down gracefully. This is the cleanest path because Gradio runs its normal cleanup routine (port release, log flush, exit).
2. **`kill <PID>`.** Find the PID with `ps` (the number in the second column of the `python src/app.py` row) and run `kill 40761` (substitute your actual PID). This sends SIGTERM. If the process does not exit within a few seconds, escalate to `kill -9 40761` (SIGKILL — uncatchable; use only when SIGTERM is ignored).
3. **`jobs` then `kill %1`.** Type `jobs` to see all suspended/background jobs in this shell, each identified by `[1]`, `[2]`, etc. Then `kill %1` terminates the suspended job by job number rather than PID — easier than reading the full `ps` output.

After cleanup, confirm with `ps | grep app.py` — no rows means the port is free and you can `python src/app.py` again.

**Going forward:** Always use **Ctrl-C** to stop `app.py`. Make it muscle memory; the keys are right next to each other on the keyboard and a slip is easy to make.

---

## Phase 12 — Day-to-Day Usage (Operating Guide)

### Adding new documents to the knowledge base

1. Copy documents into `ingest/`
2. Activate the venv: `source chatbot-16G-venv/bin/activate`
3. Run: `python src/ingest.py`
4. Restart `app.py` if it is running (stop with `Ctrl+C`, then `python src/app.py`)

**Note:** Processed files are automatically moved to `ingest/processed/` after ingestion. Re-running `ingest.py` on an already-ingested document is safe — deterministic chunk IDs mean existing chunks are overwritten, not duplicated.

### Starting the chatbot after a reboot

```bash
cd /Users/* YOUR USER NAME */RAGbot
source chatbot-16G-venv/bin/activate
python src/app.py
```

### Changing the chat model

Edit `config.py`, change the `CHAT_MODEL` value to any model name from `ollama list`, save the file, restart `app.py`. No other files need to change.

### Checking what is in the knowledge base

```bash
source chatbot-16G-venv/bin/activate
python -c "
import sys; sys.path.insert(0,'.')
from src.chat import check_knowledge_base
print(f'Chunks in knowledge base: {check_knowledge_base()}')
"
```

### Wiping and rebuilding the knowledge base

```bash
rm -rf chroma_db/
```
> **What it does:** Deletes the entire ChromaDB data folder. The next `python src/ingest.py` will rebuild from scratch.  
> **When to do this:** If you want to remove documents or if the database gets corrupted.  
> **Before wiping:** Move any documents you want to re-ingest from `ingest/processed/` back to `ingest/` first.

### Reading the logs

```bash
tail -100 logs/chatbot.log          # last 100 lines
grep "ERROR\|WARNING" logs/chatbot.log   # show only problems
grep "Query received" logs/chatbot.log   # show all user questions
```

### Checking Ollama model status during ingestion

```bash
ollama ps
```
> **What it does:** Lists which models are currently loaded in Ollama's memory.  
> **What to look for:** During ingestion, both models (`nomic-embed-text` and `gemma4:e4b`) should appear in the loaded list. If a model disappears between documents, the `OLLAMA_KEEP_ALIVE` setting may not have taken effect — see Phase 11, Step 11.2.

---


*Document version: 2.1 — 2026-05-06*  

