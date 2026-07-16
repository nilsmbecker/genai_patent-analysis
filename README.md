# README.md — Patent Analysis Platform
# GenAI Patent Analysis Platform

AI-powered patent analysis tool. Upload patent PDFs, get summaries of patents, analyse IP (Intellectual Property) risk against your design ideas, get design-around suggestions, and discover innovation gaps across the patent corpus.

---

## What it does

The platform is structured in four phases, each building on the previous:

| Phase | What it does |
|---|---|
| **1 — Ingest** | Upload a patent PDF → extract text + figures → chunk by section → embed → store in database |
| **2 — Risk Analysis** | Describe your product idea → compare against patent claims → get a 0–100 risk score per patent |
| **3 — Design Suggestions** | Take the risk result → generate design-around alternatives that avoid the conflicting claims |
| **4 — Innovation** | Scan the whole patent corpus → cluster technologies → identify unprotected gaps → propose new product directions |

---

## Tech stack

- **Backend:** FastAPI + Uvicorn
- **Database:** Supabase (PostgreSQL 15 + pgvector)
- **Embedding model:** `jinaai/jina-embeddings-v3` (1024-dim, loaded locally via `sentence-transformers`)
- **LLM:** OpenRouter (OpenAI-compatible API)
- **PDF extraction:** PyMuPDF + Tesseract OCR fallback
- **Translation:** `deep-translator` (Google Translate) for non-English patents
- **Frontend:** Jinja2 templates served by FastAPI

---

## Project structure

├── main.py                     # App entry point — FastAPI factory + lifespan
├── requirements.txt
├── .env                        # Your credentials (never commit this)
│
├── app/
│   ├── config.py               # All env vars + constants (manufacturing constraints can be entered)
│   ├── state.py                # Singleton: Supabase client + embedding model
│   ├── models.py               # All Pydantic request/response models
│   │
│   ├── routes/
│   │   ├── ui.py               # HTML page routes (/, /upload, /patent-library, /risk, ...)
│   │   └── api.py              # REST API routes (/health, /api/v1/...)
│   │
│   ├── services/
│   │   ├── ingest.py           # Phase 1: PDF → chunks → embeddings → Supabase
│   │   ├── risk.py             # Phase 2: Hybrid search + per-patent LLM risk assessment
│   │   ├── design.py           # Phase 3: Designer agent + auditor agent
│   │   ├── innovation.py       # Phase 4: Corpus clustering + gap detection + innovation vectors
│   │   ├── llm.py              # Single LLM client (OpenRouter, SHA-256 cache, retry)
│   │   ├── patent_sheet.py     # Renders a per-patent Analysis Sheet as PDF
│   │   ├── management_summary.py # Renders a multi-phase Management Summary as PDF
│   │   └── progress.py         # SSE streaming of pipeline progress to the browser
│   │
│   └── utils/
│       ├── pdf.py              # PyMuPDF text extraction + OCR fallback + section chunking
│       ├── metadata.py         # Regex-based patent metadata extraction (no LLM)
│       └── translation.py      # Language detection + translation to English
│
├── scripts/
│   └── ingest_patents.py       # CLI for batch PDF ingestion
│
├── sql_queries/                # Run manually in Supabase SQL Editor (in order)
│   ├── 001_schema.sql          # Base schema: patent_documents, patent_chunks, hybrid search function
│   ├── 002_migration_v2.sql    # Only needed if you ran an older schema version
│   ├── 003_patent_images.sql   # patent_images table
│   ├── 004_innovation_analyses.sql
│   └── 005_management_summaries.sql
│
├── templates/                  # Jinja2 HTML templates
└── static/                     # Static assets (logo, etc.)

---

## Setup

### 1. Prerequisites

- Python 3.10+
- A [Supabase](https://supabase.com) project with the **pgvector** extension enabled
- An [OpenRouter](https://openrouter.ai) API key
- Tesseract OCR (only needed for scanned PDFs)
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt install tesseract-ocr`

### 2. Clone and install dependencies

```bash
git clone <repo-url>
cd genai_patent-analysis
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key

OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_MODEL=openrouter/auto      # optional — auto picks the best available model

APP_HOST=127.0.0.1                    # optional
APP_PORT=8000                         # optional
DEBUG=false                           # optional
```

### 4. Set up the database

Go to your Supabase project → **SQL Editor** and run the files in `sql_queries/` **in order**:

1. `001_schema.sql` — creates the core tables and the hybrid search function
2. `003_patent_images.sql`
3. `004_innovation_analyses.sql`
4. `005_management_summaries.sql`

> `002_migration_v2.sql` is only needed if you ran an older version of the schema. Skip it on a fresh setup.

Also disable Row Level Security (RLS) on `patent_chunks` for the service role so the app can insert embeddings:

```sql
ALTER TABLE patent_chunks DISABLE ROW LEVEL SECURITY;
```

### 5. Run the app

```bash
python main.py
```

Or with Uvicorn directly:

```bash
uvicorn main:app --reload
```

The app will be available at `http://127.0.0.1:8000`.

> The embedding model (`jinaai/jina-embeddings-v3`) is downloaded from HuggingFace on first use (~500 MB). The app starts and passes health checks before this completes.

---

## Ingesting patents

### Via the web UI

Go to `http://127.0.0.1:8000/upload` and upload a PDF. Metadata is auto-extracted from the document; you can review and correct it before ingesting.

### Via the CLI (batch)

```bash
# Single file
python scripts/ingest_patents.py --pdf path/to/patent.pdf

# Single file with manual metadata override
python scripts/ingest_patents.py --pdf patent.pdf --patent-number EP1234567A1 --jurisdiction EP

# Whole directory
python scripts/ingest_patents.py --dir path/to/folder/

# Whole directory, skip already-ingested patents
python scripts/ingest_patents.py --dir path/to/folder/ --skip-existing
```

The CLI uses the same pipeline as the web endpoint. A failed PDF is logged and skipped — it does not abort the rest of the batch.

---
## How patent ingestion works

1. The PDF is written to a temp file and opened with PyMuPDF.
2. The first 3 pages are read to extract a header text for language detection and metadata.
3. Language is detected from the header (patent-number prefix heuristic first, then `langdetect` fallback). If the document is not in English, the header is translated via Google Translate.
4. Metadata (patent number, title, assignee, jurisdiction, publication date) is extracted with regex — no LLM, no quota used. If critical fields (`patent_number`, `title`) are still missing after regex, the LLM fills only the blanks.
5. A row is upserted into `patent_documents` (on conflict: `patent_number`), so re-ingesting a patent updates its metadata rather than creating a duplicate.
6. All pages are extracted into one full-document string (not page-by-page, to avoid splitting claims that span page boundaries). Pages with very little text are rendered as PNG figures and stored in `patent_images`.
7. The full text is split into labelled chunks by section type: `claim_independent`, `claim_dependent`, `description`.
8. If the document was not in English, all chunks are translated before embedding.
9. Each chunk is embedded with the Jina model (`task="retrieval.passage"`) in batches of 32. Embeddings are stored as `"[x,y,z,...]"` strings in `patent_chunks`.

---

## How the risk analysis works

1. Your proposed design text is embedded using the Jina model (`task="retrieval.query"`).
2. A hybrid search runs against the database: **vector similarity** (pgvector cosine) + **full-text search** (tsvector), fused with Reciprocal Rank Fusion (RRF). Independent claim chunks are weighted higher than description chunks during candidate selection.
3. The top 2 most relevant patents are selected.
4. For each patent, all claim chunks (independent + dependent) and up to 3 description chunks are fetched.
5. The LLM receives the patent claims and your design, and classifies every claim element as:
   - `matched` — clearly present in your design
   - `missing` — clearly absent (you are safe on these)
   - `unclear` — cannot be determined from the information given
6. A deterministic risk score (0–100) is computed in Python:
   ```
   score = (matched / (matched + unclear)) × 100 + (unclear / (matched + unclear)) × 30
   ```
7. The score maps to a label: **HIGH** (≥70) / **MEDIUM** (≥40) / **LOW** (≥10) / **CLEAR** (<10).

The LLM never produces the score — it only classifies elements. Scoring is always deterministic.

---

## How design suggestions work

Design suggestions run Phase 2 first, then build on top of the risk result.

1. The proposed design is embedded and the full risk pipeline runs (same as Phase 2).
2. If the original design has no risk signal at all (`risk_score == 0`, no matched elements on any patent), the pipeline stops — there is nothing to design around.
3. A **designer LLM** receives the matched claim elements from the top 2 risk patents and proposes 2 alternative designs. It is instructed to keep the same fundamental construction type (same material approach and lamination method) and only vary specific dimensions, materials, or methods.
4. Each proposal is **re-scored** against the patent database (top 3 candidates, with a score floor to skip clearly irrelevant patents). Only proposals that score `LOW` or `CLEAR` (risk score < 40) survive.
5. If a proposal fails re-scoring, it gets up to **2 revision attempts**. The designer receives the full cumulative list of claim elements to avoid (original risk + every new conflict found so far) so it cannot fix one collision by reintroducing an earlier one. After 2 failed revisions, the proposal is discarded.
6. Surviving proposals go through a **manufacturing auditor LLM**, which checks them against Fuyao's hard glass constraints (total stack 3.1–6.0 mm, PVB interlayer 0.38–0.76 mm, no HUD conductors, wedge ≤ 0.1 mrad). Proposals that violate a numeric constraint are rewritten; proposals that change the fundamental construction type are rejected outright.
7. Only proposals that passed both the risk filter and the manufacturing audit are returned.

---

## How innovation analysis works

1. If a domain is provided (e.g. "acoustic interlayer"), it is embedded and used to rank the corpus by semantic relevance via hybrid search. Without a domain, the most recently ingested patents are used. Up to 30 patents are selected.
2. If a domain is provided but nothing in the corpus clears the minimum similarity threshold (`0.35`), the pipeline stops immediately rather than running the LLM on unrelated patents.
3. Publication dates are aggregated by year from the database (no LLM) to produce a trend chart.
4. An **analyst LLM** receives a compact summary of all selected patents (up to 2 representative chunks per patent, truncated to 400 chars each). It clusters them into 3–6 technology groups and identifies 3–6 whitespace gaps — areas adjacent to existing patents that are not currently protected.
5. An **innovator LLM** receives the gaps identified by the analyst and generates 3–5 concrete innovation vectors. Each includes a `feasibility` and `novelty` rating (HIGH / MEDIUM / LOW), a rationale for which gap it addresses, and which technology clusters it relates to.
6. Results can be saved to the database for later retrieval.

Total token context per LLM call: ~30 patents × 2 chunks × 400 chars ≈ 6 000 tokens.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/api/v1/extract-metadata` | Extract patent metadata from a PDF (no ingestion) |
| `POST` | `/api/v1/ingest` | Full ingestion pipeline (SSE stream) |
| `GET` | `/api/v1/patents` | List all patents |
| `GET` | `/api/v1/patents/{id}` | Get one patent with all chunks |
| `PATCH` | `/api/v1/patents/{id}` | Update patent metadata |
| `DELETE` | `/api/v1/patents/{id}` | Delete patent + all chunks + images |
| `GET` | `/api/v1/patents/{id}/summary` | LLM-generated patent summary |
| `GET` | `/api/v1/patents/{id}/sheet/pdf` | Download per-patent Analysis Sheet PDF |
| `POST` | `/api/v1/risk-analysis` | Phase 2: IP risk assessment (SSE stream) |
| `POST` | `/api/v1/design-suggestions` | Phase 3: Design-around proposals (SSE stream) |
| `POST` | `/api/v1/innovation` | Phase 4: Gap analysis + innovation vectors (SSE stream) |
| `POST` | `/api/v1/innovation/save` | Save a completed innovation analysis |
| `GET` | `/api/v1/innovation/saved` | List saved analyses |
| `POST` | `/api/v1/management-summaries` | Generate + save a Management Summary PDF |
| `GET` | `/api/v1/management-summaries/{id}/pdf` | Download a Management Summary PDF |

The four pipeline endpoints (`/ingest`, `/risk-analysis`, `/design-suggestions`, `/innovation`) respond with `text/event-stream`. They emit `step` events during processing and a final `result` or `error` event.

---

## Key technical notes

- **Do not change the embedding model** without re-embedding all existing chunks and updating the Supabase column from `vector(1024)` to the new dimension.
- **All Supabase calls are synchronous** (supabase-py). Inside async route handlers they must be wrapped with `asyncio.to_thread()`.
- **Embeddings must be sent to Supabase as a string** in the format `"[0.123, 0.456, ...]"` — PostgREST cannot auto-cast a Python list.
- **All user-supplied text** (design specs, domain filters) is wrapped with `wrap_untrusted()` from `llm.py` before being interpolated into any LLM prompt, to prevent prompt injection.
- **LLM responses are cached** in memory using SHA-256 of the prompt. The cache resets on server restart.
- **All constants** (PVB thickness limits, glass stack dimensions, model name, batch sizes) live in `app/config.py`. Do not use `os.environ.get()` elsewhere.