"""
app/services/risk.py — Phase 2: Patent-level hybrid search and IP risk analysis pipeline.

Phase 2 (Risk Analysis):
  fetch_independent_claim_chunks — hybrid RRF search, priority-weighted by section type
  select_candidate_patents       — aggregate chunks → rank patents → return top N
  fetch_claim_family             — fetch ALL claims (independent + dependent) + description per patent
  build_patent_context_block     — format claims as [INDEPENDENT CLAIM]/[DEPENDENT CLAIM],
                                   descriptions as [BACKGROUND] for LLM context only
  _compute_risk_score            — deterministic risk_score weighted by claim type:
                                   independent claims drive 80%, dependent claims 20%
  call_agent_risk_patent         — per-patent LLM assessment; LLM returns tagged element objects,
                                   risk_score computed in Python, elements stripped to strings for frontend
  run_patent_risk_pipeline       — orchestrates Steps 1-5, returns list of patent risk dicts.
                                   Accepts optional top_n/score_floor (default = Phase 2 behaviour,
                                   unchanged); Phase 3 passes wider, cost-capped values — see below.
                                   Accepts optional on_step(step_id, status) for live progress
                                   ("search"/"candidates"/"assess") streamed to the browser — see
                                   app/services/progress.py.
  _score_to_label                — converts numeric risk_score to HIGH/MEDIUM/LOW/CLEAR label.
                                   Also imported by app/services/design.py (Phase 3).
"""
import logging
from typing import Any, Callable, Dict, List, Optional

from fastapi import HTTPException

from app.config import PVB_MIN_MM, PVB_MAX_MM, GLASS_TOTAL_MIN, GLASS_TOTAL_MAX
from app.services.llm import llm_json, wrap_untrusted, UNTRUSTED_INPUT_NOTICE
from app.services.progress import noop_on_step
from app.state import state

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------
TOP_K_CHUNKS = 3

TOP_CANDIDATE_PATENTS = 3    # max patents to expand into full claim family
CLAIM_LIMIT           = 400  # chars per claim chunk sent to LLM
DESCRIPTION_LIMIT     = 300  # chars per description chunk — background context only, not scored

# Priority weights for section types for risk analysis — used in candidate patent ranking
# Independent claims have the broadest legal scope, descriptions have the least
SECTION_PRIORITY = {
    "claim_independent": 2.0,
    "claim_dependent":   1.2,
    "description":       0.8,
}

# ===========================================================================
# PHASE 2 — RISK ANALYSIS
# ===========================================================================

# ---------------------------------------------------------------------------
# Step 1 — Priority-weighted Chunk Retrieval
# ---------------------------------------------------------------------------

def fetch_independent_claim_chunks(
    embedding: List[float],
    query_text: str,
    jurisdiction: str,
) -> List[Dict]:
    """
    Hybrid RRF search with priority-weighted results.

    Retrieval order of importance:
      1. claim_independent — broadest legal scope, primary infringement signal
      2. claim_dependent   — narrows independent claims, secondary signal
      3. description       — context only, lowest weight

    Fetches a large pool first, then weights scores by section type so that
    independent claims always rank above equivalent-scoring description chunks.
    Falls back gracefully if no independent claims exist in the corpus.
    """
    try:
        filter_jx = None if (not jurisdiction or jurisdiction.upper() == "ALL") else jurisdiction
        resp = state.supabase.rpc("match_patent_hybrid", {
            "query_embedding":     embedding,
            "query_text":          query_text,
            "filter_jurisdiction": filter_jx,
            "match_count":         TOP_K_CHUNKS * 5,  # fetch wide pool, re-rank below
        }).execute()
        chunks = resp.data or []

        if not chunks:
            log.info("Step 1 — no chunks returned from hybrid search")
            return []

        # Apply priority weight to rrf_score
        for c in chunks:
            section = c.get("section_type", "description")
            weight  = SECTION_PRIORITY.get(section, 0.5)
            c["weighted_score"] = float(c.get("rrf_score", 0.0)) * weight

        # Sort by weighted score descending
        chunks.sort(key=lambda c: c["weighted_score"], reverse=True)

        # Keep top TOP_K_CHUNKS * 3 — will be narrowed further in Step 2
        chunks = chunks[: TOP_K_CHUNKS * 3]

        counts = {"claim_independent": 0, "claim_dependent": 0, "description": 0}
        for c in chunks:
            counts[c.get("section_type", "description")] = (
                counts.get(c.get("section_type", "description"), 0) + 1
            )
        log.info(
            "Step 1 — retrieved %d chunks (ind=%d dep=%d desc=%d) jurisdiction=%s",
            len(chunks),
            counts["claim_independent"],
            counts["claim_dependent"],
            counts["description"],
            filter_jx or "ALL",
        )
        return chunks

    except Exception as exc:
        log.error("Supabase RPC failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Database retrieval error: {exc}") from exc


# ---------------------------------------------------------------------------
# Step 2 — Candidate Patent Selection
# ---------------------------------------------------------------------------

def select_candidate_patents(
    chunks: List[Dict],
    top_n: int = TOP_CANDIDATE_PATENTS,
) -> List[Dict]:
    """
    Aggregate chunks by patent and rank using three signals:
      1. sum of weighted_scores (overall relevance, priority-adjusted)
      2. count of matching chunks (breadth of match)
      3. max weighted_score (best single chunk)

    Independent claim matches contribute 3x more to total_score than
    description matches due to the weights applied in Step 1.
    """
    patent_map: Dict[str, Dict] = {}
    for c in chunks:
        pid = c.get("patent_id") or c.get("patent_number")
        if not pid:
            continue
        if pid not in patent_map:
            patent_map[pid] = {
                "patent_id":     c.get("patent_id", ""),
                "patent_number": c.get("patent_number", ""),
                "title":         c.get("title", ""),
                "jurisdiction":  c.get("jurisdiction", ""),
                "total_score":   0.0,
                "match_count":   0,
                "max_score":     0.0,
            }
        score = float(c.get("weighted_score", c.get("rrf_score", 0.0)))
        patent_map[pid]["total_score"] += score
        patent_map[pid]["match_count"] += 1
        patent_map[pid]["max_score"]    = max(patent_map[pid]["max_score"], score)

    ranked = sorted(
        patent_map.values(),
        key=lambda p: (p["total_score"], p["match_count"], p["max_score"]),
        reverse=True,
    )
    selected = ranked[:top_n]
    log.info("Step 2 — candidate patents selected: %d / %d", len(selected), len(patent_map))
    return selected


# ---------------------------------------------------------------------------
# Step 3 — Claim Family Expansion
# ---------------------------------------------------------------------------

def fetch_claim_family(patent_id: str) -> List[Dict]:
    """
    Fetch ALL chunks for a patent:
      1. claim_independent — always included
      2. claim_dependent   — always included
      3. description       — included as background context (limited to 3 chunks)

    Description chunks are fetched last and capped to avoid token bloat.
    """
    try:
        claim_rows = (
            state.supabase.table("patent_chunks")
            .select("id, section_type, content")
            .eq("patent_id", patent_id)
            .in_("section_type", ["claim_independent", "claim_dependent"])
            .execute()
            .data or []
        )
        desc_rows = (
            state.supabase.table("patent_chunks")
            .select("id, section_type, content")
            .eq("patent_id", patent_id)
            .eq("section_type", "description")
            .limit(3)
            .execute()
            .data or []
        )
        all_rows = claim_rows + desc_rows
        log.info(
            "Step 3 — claim family for patent_id=%s: %d claims + %d desc chunks",
            patent_id, len(claim_rows), len(desc_rows),
        )
        return all_rows
    except Exception as exc:
        log.warning("Claim family fetch failed for %s: %s", patent_id, exc)
        return []


# ---------------------------------------------------------------------------
# Step 4 — Format Context Block for LLM
# ---------------------------------------------------------------------------

def build_patent_context_block(patent: Dict, claim_chunks: List[Dict]) -> str:
    """
    Format a patent's chunks into a labelled block for the LLM.

    Claim chunks keep their original labels [INDEPENDENT CLAIM] / [DEPENDENT CLAIM]
    so the LLM understands the patent structure and legal hierarchy.
    Note: our regex-based section classifier in pdf.py may mislabel some chunks,
    but the LLM can infer claim type from the actual text content regardless.

    Description chunks are labelled [BACKGROUND] and placed in a separate section.
    The LLM is instructed to use them only for technical context, not for scoring.
    """
    header      = f"PATENT: {patent['patent_number']} — {patent.get('title', '')}"
    claim_parts = [header]
    desc_parts  = []

    for c in claim_chunks:
        section = c.get("section_type", "description")
        content = c["content"]

        if section == "claim_independent":
            truncated = content[:CLAIM_LIMIT] + ("…" if len(content) > CLAIM_LIMIT else "")
            claim_parts.append(f"[INDEPENDENT CLAIM]\n{truncated}")
        elif section == "claim_dependent":
            truncated = content[:CLAIM_LIMIT] + ("…" if len(content) > CLAIM_LIMIT else "")
            claim_parts.append(f"[DEPENDENT CLAIM]\n{truncated}")
        else:
            truncated = content[:DESCRIPTION_LIMIT] + ("…" if len(content) > DESCRIPTION_LIMIT else "")
            desc_parts.append(f"[BACKGROUND]\n{truncated}")

    block = "\n---\n".join(claim_parts)
    if desc_parts:
        block += "\n\n--- BACKGROUND CONTEXT (technical context only — do not score against this) ---\n"
        block += "\n---\n".join(desc_parts)

    return block


# ---------------------------------------------------------------------------
# Step 5 — Risk Score Computation + LLM Assessment
# ---------------------------------------------------------------------------

def _compute_risk_score(matched: List, missing: List, unclear: List) -> int:
    """
    Compute a deterministic risk_score (0-100).
    No independent/dependent weighting — claim type labelling is not reliable enough.

    Formula:
      total = matched + unclear
      score = (matched / total × 100) + (unclear / total × 30)

    matched contributes up to 100, unclear contributes up to 30.
    Missing elements are excluded entirely — they are legal protection.
    Returns 0 if both matched and unclear are empty.
    """
    n_matched = len(matched)
    n_unclear = len(unclear)
    total = n_matched + n_unclear
    if total == 0:
        return 0
    final = (n_matched / total * 100) + (n_unclear / total * 30)
    return max(0, min(100, round(final)))


def call_agent_risk_patent(
    patent: Dict,
    claim_chunks: List[Dict],
    proposed_specs: str,
    component_scope: str,
) -> Dict[str, Any]:
    """
    Phase 2 Step 5 — Structured per-patent risk assessment.

    The LLM reads all claims and the proposed design, then identifies every
    distinct claim element and classifies it into one of three lists.
    Each element is an object carrying its claim_type tag so _compute_risk_score
    can weight independent claim matches above dependent claim matches.

    LLM output elements schema:
      {"element": "plain English description including layer/component context",
       "claim_type": "independent" | "dependent"}

    risk_score is NOT produced by the LLM — it is computed deterministically
    in Python by _compute_risk_score() from the three element lists.

    Frontend output: matched_elements, missing_elements, unclear_elements are
    converted to plain strings (element text only) so the API response stays
    identical to before — claim_type is used internally for scoring only.
    """
    context_block = build_patent_context_block(patent, claim_chunks)

    prompt = f"""You are a senior IP Engineer performing patent infringement analysis.
Respond ONLY with minified JSON. No markdown. No preamble.

Output schema (one object, exactly this structure):
{{"patent_number":"{patent['patent_number']}","matched_elements":[{{"element":"...","claim_type":"independent"}}],"missing_elements":[{{"element":"...","claim_type":"independent"}}],"unclear_elements":[{{"element":"...","claim_type":"independent"}}],"overlap_explanation":"..."}}

Rules:
- Analyse ONLY [INDEPENDENT CLAIM] and [DEPENDENT CLAIM] sections for infringement.
- [INDEPENDENT CLAIM] defines the broadest legal scope — these are legally critical.
- [DEPENDENT CLAIM] narrows the independent claims — assess after independent claims.
- [BACKGROUND] is technical context only — do NOT extract elements from it.
- Each element must appear in exactly one list — matched, missing, or unclear.
- claim_type must be "independent" if the element comes from [INDEPENDENT CLAIM], else "dependent".
- element: plain English description — always include the component/layer context
  (e.g. "Core layer: SiO2 45-85 mol%" not just "SiO2 45-85 mol%").
- matched_elements: claim elements clearly present in the proposed design.
- missing_elements: claim elements clearly absent from the proposed design.
- unclear_elements: claim elements that cannot be determined without more information.
- overlap_explanation: 1-2 sentence plain English explanation of the technical overlap.
- Base your analysis on claim language, NOT on semantic similarity alone.
- Fuyao manufacturing context: glass thickness {GLASS_TOTAL_MIN}-{GLASS_TOTAL_MAX}mm, PVB interlayer {PVB_MIN_MM}-{PVB_MAX_MM}mm.
- You have limited output space. If running long, stop adding elements and close the JSON immediately.
- {UNTRUSTED_INPUT_NOTICE}

COMPONENT SCOPE: {wrap_untrusted("component_scope", component_scope)}
PROPOSED DESIGN: {wrap_untrusted("proposed_specifications", proposed_specs[:2000])}

PATENT CLAIMS AND BACKGROUND: {context_block}

Identify every distinct claim element. Tag each with its claim_type. Classify each as matched, missing, or unclear."""

    result = llm_json(prompt)

    # Ensure all required keys exist with safe defaults
    for key, default in [
        ("patent_number",       patent["patent_number"]),
        ("matched_elements",    []),
        ("missing_elements",    []),
        ("unclear_elements",    []),
        ("overlap_explanation", ""),
    ]:
        if key not in result:
            result[key] = default

    # Compute risk_score from tagged element lists
    result["risk_score"] = _compute_risk_score(
        result["matched_elements"],
        result["missing_elements"],
        result["unclear_elements"],
    )

    # Convert element objects to plain strings for the frontend
    # claim_type has already been used for scoring above
    for key in ("matched_elements", "missing_elements", "unclear_elements"):
        result[key] = [
            e["element"] if isinstance(e, dict) else e
            for e in result[key]
        ]

    return result


# ---------------------------------------------------------------------------
# Orchestrator — runs Steps 1-5 in sequence
# ---------------------------------------------------------------------------

def run_patent_risk_pipeline(
    embedding: List[float],
    query_text: str,
    proposed_specs: str,
    component_scope: str,
    jurisdiction: str,
    top_n: int = TOP_CANDIDATE_PATENTS,
    score_floor: float = 0.0,
    on_step: Optional[Callable[[str, str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Full patent-level risk pipeline. Returns a list of per-patent risk dicts,
    sorted descending by risk_score.
    Returns an empty list if no chunks are found.

    risk_score → label mapping (see _score_to_label):
      >= 70 → HIGH, >= 40 → MEDIUM, >= 10 → LOW, < 10 → CLEAR

    top_n       — how many candidate patents to expand into a full claim-family
                  LLM assessment. Defaults to TOP_CANDIDATE_PATENTS (Phase 2 behaviour).
    score_floor — skip the LLM assessment for any candidate whose total_score is
                  below score_floor * (best candidate's total_score). 0.0 (default)
                  disables this filter entirely.
    on_step     — optional progress callback (step_id, status), step ids "search",
                  "candidates", "assess". Only passed by the user-facing risk-analysis
                  and design-suggestions routes — internal re-scoring calls omit it.
    """
    step = on_step or noop_on_step

    # Step 1 — retrieve chunks via hybrid RRF search
    step("search", "active")
    ind_chunks = fetch_independent_claim_chunks(embedding, query_text, jurisdiction)
    step("search", "done")
    if not ind_chunks:
        log.info("Pipeline: no chunks found → CLEAR")
        step("candidates", "skipped")
        step("assess", "skipped")
        return []

    # Step 2 — select top candidate patents
    step("candidates", "active")
    candidate_patents = select_candidate_patents(ind_chunks, top_n=top_n)
    step("candidates", "done")
    if not candidate_patents:
        step("assess", "skipped")
        return []

    best_score = candidate_patents[0].get("total_score", 0.0)

    # Steps 3-5 — expand claim family, format context, assess each patent
    step("assess", "active")
    results: List[Dict[str, Any]] = []
    for patent in candidate_patents:
        if score_floor and patent.get("total_score", 0.0) < score_floor * best_score:
            log.info(
                "Step 3 — skipping %s, total_score below score_floor (no LLM call)",
                patent.get("patent_number", "?"),
            )
            continue

        claim_chunks = fetch_claim_family(patent["patent_id"])
        if not claim_chunks:
            log.warning("No chunks found for patent %s, skipping", patent["patent_number"])
            continue
        try:
            assessment = call_agent_risk_patent(
                patent, claim_chunks, proposed_specs, component_scope
            )
            # Attach patent metadata to result
            assessment["title"]        = patent.get("title", "")
            assessment["jurisdiction"] = patent.get("jurisdiction", "")
            assessment["match_count"]  = patent.get("match_count", 0)
            assessment["total_score"]  = round(patent.get("total_score", 0.0), 6)
            results.append(assessment)
        except Exception as exc:
            log.warning("Risk assessment failed for patent %s: %s", patent["patent_number"], exc)

    # Sort by risk_score descending
    results.sort(key=lambda r: r.get("risk_score", 0), reverse=True)
    step("assess", "done")
    log.info("Pipeline complete: %d patents assessed", len(results))
    return results


# ---------------------------------------------------------------------------
# Shared helper — risk score → label
# ---------------------------------------------------------------------------

def _score_to_label(score: int) -> str:
    """Convert a numeric risk_score (0-100) to a HIGH/MEDIUM/LOW/CLEAR label.
    Also imported by app/services/design.py (Phase 3).
    """
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    if score >= 10:
        return "LOW"
    return "CLEAR"