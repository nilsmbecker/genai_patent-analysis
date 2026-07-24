"""
app/services/design.py — Phase 3: Design-around suggestion agents.

Imports run_patent_risk_pipeline and _score_to_label from app.services.risk (Phase 2)
to re-score proposals without duplicating any retrieval logic.

Phase 3 (Design Suggestions):
  call_agent_designer   — proposes 2 alternative designs based on risk output,
                          re-scores each via run_patent_risk_pipeline (DESIGNER_RESCORE_TOP_N
                          candidates, gated by DESIGNER_RESCORE_SCORE_FLOOR). A failing proposal
                          gets up to MAX_REFINEMENT_ROUNDS revisions (_revise_proposal) before
                          being discarded. Keeps only proposals that end up LOW/CLEAR.
  _revise_proposal      — asks the designer to fix one failing proposal, given the cumulative
                          list of claim elements to avoid across all rounds so far
  call_agent_auditor    — validates surviving proposals against Fuyao manufacturing constraints
                          and against the original spec (catches invented "original" baselines)
"""
import json
import logging
from typing import Any, Dict, List

from app.config import PVB_MIN_MM, PVB_MAX_MM, GLASS_TOTAL_MIN, GLASS_TOTAL_MAX
from app.models import DesignAroundProposal
from app.services.llm import llm_json, wrap_untrusted, UNTRUSTED_INPUT_NOTICE
from app.services.risk import run_patent_risk_pipeline, _score_to_label

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Re-score constants — deliberately fixed, independent of corpus size. A larger
# top_n widens the safety net (checks more candidate patents per proposal); the
# score_floor keeps that affordable by skipping the LLM call for candidates that
# clearly aren't close matches (uses scores already computed during retrieval,
# no extra embedding or LLM call). Lowered from 5 to keep the design-suggestions
# wait time reasonable without meaningfully narrowing the safety net (the
# top-ranked candidates carry almost all the risk).
DESIGNER_RESCORE_TOP_N       = 3
DESIGNER_RESCORE_SCORE_FLOOR = 0.2

# Refinement-loop constant — a proposal that fails re-scoring gets up to this many
# revision attempts (carrying forward every claim element found so far, not just
# the latest one) before it is discarded. Worst-case cost per proposal:
# 1 + MAX_REFINEMENT_ROUNDS re-score calls plus MAX_REFINEMENT_ROUNDS revision
# calls — a fixed multiplier, not DB-size-dependent.
MAX_REFINEMENT_ROUNDS = 2


# ===========================================================================
# PHASE 3 — DESIGN SUGGESTIONS
# ===========================================================================

def call_agent_designer(
    proposed_specs: str,
    component_scope: str,
    patent_assessments: List[Dict[str, Any]],
    embed_model,
    jurisdiction: str,
) -> List[Dict[str, Any]]:
    """
    Phase 3 — Design suggestion agent.

    Step 1: Build a concise risk summary from the patent_assessments produced by
            run_patent_risk_pipeline (new schema: matched_elements, risk_score, etc.)
            and prompt the LLM to propose 2 alternative designs. The prompt also
            tells the designer to preserve the original construction type (same
            rule call_agent_auditor enforces) so it doesn't waste a round proposing
            something the audit step will reject anyway.
    Step 2: Re-score each proposal using run_patent_risk_pipeline against
            DESIGNER_RESCORE_TOP_N candidate patents (currently equal to Phase 2's
            own TOP_CANDIDATE_PATENTS), gated by DESIGNER_RESCORE_SCORE_FLOOR so the
            check doesn't cost extra LLM calls for clearly-irrelevant candidates.
    Step 3: If a proposal fails (HIGH/MEDIUM), don't discard it immediately — ask
            the designer to revise it (_revise_proposal) against the cumulative set
            of claim elements to avoid so far, and re-score again. Up to
            MAX_REFINEMENT_ROUNDS revisions per proposal before giving up.
    Step 4: Return only proposals that end up LOW or CLEAR (i.e. risk_score < 40).
    """
    # Summarise the highest-risk patents for the designer prompt (top 2 is sufficient)
    risk_summary = [
        {
            "patent_number":    a.get("patent_number", ""),
            "risk_score":       a.get("risk_score", 0),
            "matched_elements": a.get("matched_elements", []),
        }
        for a in patent_assessments[:2]
    ]
    risk_block    = json.dumps(risk_summary)
    original_risk = _score_to_label(patent_assessments[0].get("risk_score", 0) if patent_assessments else 0)

    prompt = f"""IP Design Engineer. Respond ONLY with minified JSON. No markdown.

Schema: {{"design_arounds":[{{"id":"DA1","description":"2-3 sentence engineering description","rationale":"which matched claim elements are avoided and how","addresses_claims":["PATENT_NUMBER Claim N"]}},{{"id":"DA2","description":"...","rationale":"...","addresses_claims":[]}}]}}

Original risk: {original_risk}
Matched claim elements to avoid: {risk_block}
Scope: {wrap_untrusted("component_scope", component_scope)}
Original spec: {wrap_untrusted("proposed_specifications", proposed_specs[:400])}

{UNTRUSTED_INPUT_NOTICE}

Propose 2 short alternative designs that structurally avoid the matched claim elements. Keep the
same fundamental construction type as the original (same material approach and bonding/lamination
method) — only vary specific dimensions, materials, or methods. Do not abandon lamination, switch
glass to polymer, or remove the bonding interlayer; such proposals will be rejected later anyway."""

    try:
        designer_output = llm_json(prompt)
    except Exception as exc:
        log.error("call_agent_designer: proposal step failed: %s", exc)
        raise

    proposals = designer_output.get("design_arounds", [])
    log.info(
        "Designer proposed %d alternatives — re-scoring each via run_patent_risk_pipeline",
        len(proposals),
    )

    surviving: List[Dict[str, Any]] = []
    for proposal in proposals:
        proposal_spec = proposal.get("description", "")
        if not proposal_spec:
            continue

        # Cumulative list of claim elements this proposal must avoid — starts with
        # the original risk patents and grows with every new conflict found during
        # re-scoring, so a revision never "fixes" one collision by reintroducing
        # an earlier one.
        avoid_elements = list(risk_summary)
        current        = proposal
        passed         = False

        for round_num in range(MAX_REFINEMENT_ROUNDS + 1):
            try:
                proposal_embedding = embed_model.encode(
                    [proposal_spec], task="retrieval.query",
                    normalize_embeddings=True, show_progress_bar=False
                )[0].tolist()
                proposal_results  = run_patent_risk_pipeline(
                    proposal_embedding, proposal_spec, proposal_spec, component_scope, jurisdiction,
                    top_n=DESIGNER_RESCORE_TOP_N, score_floor=DESIGNER_RESCORE_SCORE_FLOOR,
                )
                proposal_top_score  = proposal_results[0].get("risk_score", 0) if proposal_results else 0
                proposal_risk_label = _score_to_label(proposal_top_score)

                current["risk_score"] = proposal_risk_label
                log.info(
                    "Proposal %s round %d re-scored: %s (score=%d)",
                    current.get("id"), round_num, proposal_risk_label, proposal_top_score,
                )

                if proposal_risk_label in ("LOW", "MEDIUM", "CLEAR"):
                    passed = True
                    break

                if round_num == MAX_REFINEMENT_ROUNDS:
                    log.info(
                        "Proposal %s filtered out after %d refinement round(s) (risk=%s)",
                        current.get("id"), round_num, proposal_risk_label,
                    )
                    break

                # New conflict found — add it to the cumulative avoid list and ask
                # the designer to revise, carrying forward everything avoided so far.
                new_conflict = proposal_results[0]
                avoid_elements.append({
                    "patent_number":    new_conflict.get("patent_number", ""),
                    "risk_score":       new_conflict.get("risk_score", 0),
                    "matched_elements": new_conflict.get("matched_elements", []),
                })
                log.info(
                    "Proposal %s conflicts with %s (round %d) — requesting revision",
                    current.get("id"), new_conflict.get("patent_number", "?"), round_num,
                )
                revised = _revise_proposal(current, avoid_elements, component_scope)
                if not revised.get("description"):
                    break
                current       = revised
                proposal_spec = revised["description"]

            except Exception as exc:
                log.warning("Re-scoring proposal %s failed, skipping: %s", current.get("id"), exc)
                break

        if passed:
            surviving.append(current)

    log.info("%d / %d proposals passed risk filter", len(surviving), len(proposals))
    return surviving


def _revise_proposal(
    proposal: Dict[str, Any],
    avoid_elements: List[Dict[str, Any]],
    component_scope: str,
) -> Dict[str, Any]:
    """
    Phase 3 — Refinement step. Called when a proposal fails re-scoring.
    Asks the designer to revise the single failing proposal, carrying forward the
    full cumulative list of claim elements to avoid (original risk + every new
    conflict found so far across all rounds) — not just the latest conflict — so
    the revision doesn't trade one collision for an earlier one.
    """
    avoid_block = json.dumps(avoid_elements)
    prompt = f"""IP Design Engineer. Respond ONLY with minified JSON. No markdown.

Schema: {{"id":"{proposal.get('id', '')}","description":"2-3 sentence engineering description","rationale":"which matched claim elements are avoided and how"}}

Your previous proposal still conflicts with a patent. Revise it so it avoids ALL of the
following claim elements (from every patent encountered so far, not just the latest one):
{avoid_block}

Scope: {wrap_untrusted("component_scope", component_scope)}
Previous proposal: {proposal.get('description', '')[:400]}

{UNTRUSTED_INPUT_NOTICE}

Propose one revised design that structurally avoids every listed element. Keep the same
fundamental construction type as the original (same material approach and bonding/lamination
method) — only vary specific dimensions, materials, or methods. Do not abandon lamination, switch
glass to polymer, or remove the bonding interlayer to escape a conflict; such proposals will be
rejected later anyway."""

    try:
        revised = llm_json(prompt)
    except Exception as exc:
        log.warning("Revision call failed for proposal %s: %s", proposal.get("id"), exc)
        return {}

    revised.setdefault("id", proposal.get("id", ""))
    return revised


def call_agent_auditor(
    design_arounds: List[Dict],
    component_scope: str,
    proposed_specs: str,
) -> List[DesignAroundProposal]:
    """
    Manufacturing audit agent.

    Validates each design proposal against Fuyao's hard glass constraints.
    Rewrites any proposal that violates a numeric constraint. The original spec is
    included so the auditor can verify "rewritten" descriptions actually reference
    what the user submitted, instead of inventing a baseline. Proposals that change
    the fundamental construction type (not just a dimension/method within it) are
    rejected outright via passed_audit=false rather than rewritten — dropped from
    the returned list, since no rewrite can fix "this is a different kind of product."

    The auditor LLM only sees a 300/150-char trimmed description/rationale (to keep
    its prompt small). If it echoes a field back unchanged, the full original text
    is restored for the response instead of the truncated echo — no extra LLM call.
    """
    if not design_arounds:
        return []

    # Trim each proposal to avoid token overflow in the auditor prompt
    trimmed = [
        {
            "id":          da.get("id", ""),
            "description": da.get("description", "")[:300],
            "rationale":   da.get("rationale", "")[:150],
        }
        for da in design_arounds
    ]
    da_block = json.dumps(trimmed)

    prompt = f"""Fuyao Glass Auditor. Respond ONLY with minified JSON. No markdown.

Constraints: thickness {GLASS_TOTAL_MIN}-{GLASS_TOTAL_MAX}mm, PVB {PVB_MIN_MM}-{PVB_MAX_MM}mm, no HUD conductors, wedge ≤0.1mrad.

A design-around must keep the same fundamental construction type as the original — same general
material approach and bonding/lamination method — only varying specific dimensions, materials, or
methods to avoid the claimed elements. If a proposal changes the basic construction approach itself
(e.g. monolithic instead of laminated, polymer instead of glass, removes the bonding interlayer
entirely), set passed_audit to false and explain why in audit_notes. Do not rewrite such a proposal
into something that merely looks numerically compliant — reject it instead.

Schema: {{"audited_design_arounds":[{{"id":"...","description":"...","rationale":"...","passed_audit":true,"audit_notes":"..."}}]}}

{UNTRUSTED_INPUT_NOTICE}

ORIGINAL SPEC (what the proposals must actually be compared against): {wrap_untrusted("proposed_specifications", proposed_specs[:300])}
SCOPE: {wrap_untrusted("component_scope", component_scope)}
DESIGNS: {da_block}

Check each against constraints. If a description or rationale misstates the original spec above, correct it. Rewrite if violated (but not a fundamental construction change — reject those per above)."""

    audit_result = llm_json(prompt)
    audited_map  = {a["id"]: a for a in audit_result.get("audited_design_arounds", [])}
    trimmed_map  = {t["id"]: t for t in trimmed}

    merged: List[DesignAroundProposal] = []
    for da in design_arounds:
        da_id = da.get("id", "")
        if da_id in audited_map:
            a = audited_map[da_id]
            if not a.get("passed_audit", True):
                log.info(
                    "Proposal %s failed manufacturing audit, dropping: %s",
                    da_id, a.get("audit_notes", ""),
                )
                continue

            # The auditor only ever sees the trimmed (300/150 char) text. If it didn't
            # actually rewrite a field — i.e. it echoed the trimmed input back unchanged —
            # use the full original text instead of the truncated echo. No extra LLM call,
            # no extra tokens: this is a pure Python comparison against text we already have.
            t = trimmed_map.get(da_id, {})
            audited_desc = a.get("description", "")
            audited_rat  = a.get("rationale", "")
            final_desc = da.get("description", "") if audited_desc == t.get("description") else audited_desc
            final_rat  = da.get("rationale", "")   if audited_rat  == t.get("rationale")   else audited_rat

            merged.append(DesignAroundProposal(
                id=da_id,
                description=final_desc or da.get("description", ""),
                rationale=final_rat or da.get("rationale", ""),
                audited=True,
                audit_notes=a.get("audit_notes"),
                risk_score=da.get("risk_score"),
            ))
        else:
            merged.append(DesignAroundProposal(
                id=da_id,
                description=da.get("description", ""),
                rationale=da.get("rationale", ""),
                audited=False,
                audit_notes="Audit result not returned.",
                risk_score=da.get("risk_score"),
            ))
    return merged