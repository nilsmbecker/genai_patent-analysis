# CLAUDE.md — Patent Analysis Platform

## Project Context

AI-powered patent analysis tool.
Domain: laminated automotive glass (windshields, HUD zones, PVB interlayers).
Stack: FastAPI + Supabase (pgvector) + OpenRouter (OpenAI-compatible API) + jinaai/jina-embeddings-v3 (1024-dim).

Four phases per assignment:
1. Patent ingestion + structured storage ✅
2. Risk identification (design vs. patent claims) ✅
3. Design-around suggestions (with manufacturing audit) ✅
4. Innovation gap analysis across patent portfolios ✅

---

## Current Architecture

```
app/
  config.py         ← All constants + pydantic-settings BaseSettings
  state.py          ← AppState dataclass (embed_model, supabase)
  models.py         ← All Pydantic request/response models
  routes/
    ui.py           ← Page routes: / /upload /patent-library /risk /design-suggestions /innovation /summaries /downloads
    api.py          ← API routes: /health /api/v1/...
  services/
    llm.py          ← OpenRouter client, retry, in-memory SHA-256 cache
    ingest.py       ← Shared ingestion pipeline (web endpoint + CLI both import this)
    risk.py         ← Phase 2: Hybrid RRF search + risk pipeline + _score_to_label
    design.py       ← Phase 3: Designer + auditor agents (imports from risk.py)
    innovation.py   ← Phase 4: corpus analysis, clustering, gap detection, innovation vectors
    management_summary.py ← Renders multi-patent Management Summary PDF
    patent_sheet.py ← Renders per-patent Analysis Sheet PDF
    progress.py     ← SSE streaming of pipeline progress to the browser
  utils/
    pdf.py          ← PyMuPDF extraction + OCR fallback + chunking
    metadata.py     ← Regex-based patent metadata extraction (no LLM)
    translation.py  ← Language detection + Google Translate via deep-translator
main.py             ← App factory + lifespan only
scripts/
  ingest_patents.py ← CLI arg parsing only; calls app/services/ingest.py
sql_queries/
  schema.sql                  ← Base schema (patent_documents, patent_chunks, match_patent_hybrid)
  migration_v2.sql            ← Fixes fts_tokens: GENERATED → trigger
  003_patent_images.sql
  004_innovation_analyses.sql
  005_management_summaries.sql
templates/          ← Jinja2 templates
  _pipeline_status.html
  base.html
  patent-library.html
  summaries.html
  upload.html
  risk.html
  design-suggestions.html
  innovation.html
  downloads.html
static/
  Fuyao_logo.svg
uploads/            ← Temp uploads, gitignored
requirements.txt
CLAUDE.md
```

---

## Key Design Decisions

- **Embedding model**: `jinaai/jina-embeddings-v3` (1024 dims, `trust_remote_code=True` required). Do not change without re-embedding all chunks and migrating the Supabase column from `vector(1024)`.
- **Hybrid search**: Reciprocal Rank Fusion (RRF) of pgvector cosine similarity + full-text search (`tsvector`). SQL function `match_patent_hybrid` lives in Supabase.
- **pgvector format**: Embeddings must be sent as a string `"[0.123,0.456,...]"` — PostgREST cannot auto-cast Python lists.
- **`fts_tokens` is a trigger column, not GENERATED** — `GENERATED` columns cause silent insert failures via PostgREST. See `sql_queries/migration_v2.sql`.
- **Risk scoring**: LLM identifies claim elements and classifies them into `matched_elements`, `missing_elements`, `unclear_elements`. `risk_score` (0–100) is computed deterministically in Python by `_compute_risk_score()` — never by the LLM. Score formula: `(matched / (matched + unclear)) × 100 + (unclear / (matched + unclear)) × 30`. Missing elements contribute zero — a missing element means the claim cannot be asserted.
- **Claim labels in LLM prompt**: `[INDEPENDENT CLAIM]` / `[DEPENDENT CLAIM]` labels are kept in the context block so the LLM understands legal hierarchy. `[BACKGROUND]` sections are description chunks — LLM uses them for context only, never for scoring.
- **Two-agent pattern**: Phase 2 uses a single risk assessment agent. Phases 3 use designer + auditor agents. Phase 4 uses analyst + innovator agents.
- **Glass domain constants** (PVB thickness, HUD zone, wedge angle) belong in `config.py`, not scattered inline.
- **LLM client**: OpenRouter via OpenAI-compatible API (`openai` SDK, base URL `https://openrouter.ai/api/v1`). In-memory cache keyed by SHA-256 of prompt. Per-minute 429s trigger one retry with backoff. All retry/fallback logic lives exclusively in `app/services/llm.py`.
- **Supabase calls in sync context**: All `state.supabase.*` calls must be wrapped in `asyncio.to_thread()` inside async route handlers.

---

## Phase 1 — Patent Ingestion

### Pipeline (6 steps, streamed via SSE)

**Step 1 — `read`**: Write PDF bytes to a temp file, open with PyMuPDF. Extract header text from the first `META_EXTRACT_PAGES = 3` pages for language detection and metadata extraction.

**Step 2 — `translate`**: Detect source language (patent-number prefix heuristic first, then `langdetect` fallback). If not English, translate the header via Google Translate before metadata extraction. Run regex-based `extract_metadata()` first (no LLM, no quota). Only if `patent_number` or `title` is still blank does `llm_json()` fill the missing fields.

**Step 3 — `save_record`**: Upsert into `patent_documents` on conflict `patent_number` — re-ingesting an existing patent updates its metadata, never duplicates.

**Step 4 — `extract`**: Extract all pages into one full-document string (not page-by-page — claims spanning page boundaries were splitting into incomplete fragments). Run `split_into_chunks()` to label paragraphs as `claim_independent`, `claim_dependent`, or `description`. Also extract figure pages as PNG and store in `patent_images` (one insert per image — batching large blobs exceeds Supabase statement timeout).

**Step 5 — `embed`**: Translate all chunks to English if needed. Encode with `embed_model.encode(task="retrieval.passage", normalize_embeddings=True, batch_size=BATCH_SIZE)`.

**Step 6 — `index`**: Bulk-insert `patent_chunks` rows. Embedding sent as `"[x,y,z,...]"` string — PostgREST cannot auto-cast Python lists.

---

## Phase 2 — Risk Analysis

### Pipeline (5 steps)

**Step 1 — `fetch_independent_claim_chunks`**: Embed user design → hybrid RRF search → weight results by section type (`claim_independent ×2.0`, `claim_dependent ×1.2`, `description ×0.8`). Weighting only affects candidate patent selection, not what the LLM sees.

**Step 2 — `select_candidate_patents`**: Aggregate chunks by patent (sum of weighted scores, match count, max score). Select top `TOP_CANDIDATE_PATENTS = 2`.

**Step 3 — `fetch_claim_family`**: Fetch all claim chunks (independent + dependent) + up to 3 description chunks for each candidate patent.

**Step 4 — `build_patent_context_block`**: Format into labelled block: `[INDEPENDENT CLAIM]`, `[DEPENDENT CLAIM]`, `[BACKGROUND]`. Description chunks are background context only.

**Step 5 — `call_agent_risk_patent`**: LLM returns `matched_elements`, `missing_elements`, `unclear_elements` (each element tagged with `claim_type: independent|dependent`). `_compute_risk_score()` calculates the final score in Python. Elements are stripped back to plain strings before returning to the frontend.

### Risk score labels
- ≥ 70 → HIGH
- ≥ 40 → MEDIUM
- ≥ 10 → LOW
- < 10 → CLEAR

---

## Phase 3 — Design Suggestions

### Pipeline (3 steps + refinement loop)

**Step 1 — Risk re-run**: Runs Phase 2 in full on the original design. If `risk_score == 0` and no matched elements exist on any patent, the pipeline stops — nothing to design around.

**Step 2 — `call_agent_designer`**: Builds a concise risk summary (top 2 patents, matched elements only) and prompts the designer LLM to propose 2 alternative designs. The designer is constrained to keep the same fundamental construction type (same material approach and lamination method) and only vary specific dimensions, materials, or methods.

**Step 3 — Re-score + refinement loop**: Each proposal is embedded and run through `run_patent_risk_pipeline` against `DESIGNER_RESCORE_TOP_N = 3` candidate patents (wider than Phase 2's default of 2), gated by `DESIGNER_RESCORE_SCORE_FLOOR = 0.2`.
- If a proposal scores `LOW` or `CLEAR` (risk_score < 40) → it survives.
- If it fails → `_revise_proposal()` is called with the **cumulative** avoid list (original risk elements + all new conflicts found so far across all rounds). Up to `MAX_REFINEMENT_ROUNDS = 2` revisions per proposal. After 2 failed revisions the proposal is discarded.

**Step 4 — `call_agent_auditor`**: Surviving proposals are validated against Fuyao's hard glass constraints (`GLASS_TOTAL_MIN`–`GLASS_TOTAL_MAX` mm stack, `PVB_MIN_MM`–`PVB_MAX_MM` mm interlayer, no HUD conductors, wedge ≤ 0.1 mrad). Auditor sees truncated text (300/150 chars) to keep its prompt small; if it echoes a field back unchanged, the full original text is restored in Python — no extra LLM call. Proposals that violate a numeric constraint are rewritten; proposals that change the fundamental construction type are rejected outright.

Only proposals that pass both the risk filter and the manufacturing audit are returned.

---

## Phase 4 — Innovation Opportunities

### Pipeline (4 steps)

**Step 1 — `fetch_corpus_overview`**: Embed domain → hybrid-search to rank up to `MAX_CORPUS_PATENTS = 30` patents. Fetches `MAX_CHUNKS_PER_PATENT = 2` representative chunks per patent (`MAX_CLAIM_CHARS = 400`). Scope: `"full"` / `"claims"` / `"description"`. Rejects domains with no real corpus match (`INNOVATION_MIN_VECTOR_SIMILARITY = 0.35`).

**Step 2 — `extract_trend_data`**: Pure DB aggregation, no LLM. Groups `publication_date` by year into `[{year, count}]`.

**Step 3 — `call_agent_analyst`** (LLM call 1): Clusters patents into 3–6 technology groups, identifies 3–6 whitespace gaps (`opportunity_level: HIGH|MEDIUM|LOW`).

**Step 4 — `call_agent_innovator`** (LLM call 2): Generates 3–5 innovation vectors grounded in the gaps. Each carries `feasibility`, `novelty` (HIGH/MEDIUM/LOW), `gap_rationale`, `addresses_clusters`.

### Token budget
30 patents × 2 chunks × 400 chars ≈ 24 000 chars ≈ 6 000 tokens of context per LLM call.

---

## Known Issues

1. **No `.env.example`**: Missing — should be created for onboarding.
2. **SQL files are not auto-applied**: `sql_queries/` files must be run manually in the Supabase SQL editor. They are reference files only.

---

## Environment Variables Required

```
SUPABASE_URL=
SUPABASE_ANON_KEY=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/auto     # optional
APP_HOST=127.0.0.1                   # optional
APP_PORT=8000                        # optional
DEBUG=false                          # optional
```

---

## Running the App

```bash
pip install -r requirements.txt
python main.py
# or:
uvicorn main:app --reload
```

CLI ingestion:
```bash
# Single file
python scripts/ingest_patents.py --pdf path/to/patent.pdf

# Directory batch
python scripts/ingest_patents.py --dir path/to/folder/

# Skip already-ingested patents
python scripts/ingest_patents.py --dir path/to/folder/ --skip-existing
```

---

## Code Rules

- Include meaningful comments at the top of each file: what it does, global variables, functions, complex logic. Short and on point.
- No backwards-compat shims. Delete dead code.
- No premature abstraction. Solve the actual problem.
- All LLM prompts must request `ONLY minified JSON` — the parser depends on it.
- Never commit `.env`. Use `.env.example` for documentation.
- Supabase RLS must be **disabled** on `patent_chunks` for the service role to insert embeddings.
- Prompt injection defense: all user-submitted strings (design specs, domain, focus fields) must be wrapped with `wrap_untrusted()` from `llm.py` before interpolation into any prompt.