# ADR-001: Consensus Gating and Thematic Validation for Thematic MAA Mapping

**Date:** 2026-11-07  
**Status:** Accepted  
**Author:** Technical Writer (documentation of backend_engineer Phases 2–7)  
**Jira Ticket:** CC-2026-000  

## Context

The CC-Mappers project improves thematic Men-at-Arms (MAA) mapping accuracy in the Crusader Conflicts mod. During Phases 2–7, several root causes of thematic MAA inaccuracy were addressed:

1. **Faction context stripped from prompts** — LLM prompts for thematic MAA assignments lacked cultural grounding (subculture, heritage, prefix).
2. **Majority-vote force-apply clobbering JSON overrides** — Consensus/tournament paths could overwrite JSON-mandated units.
3. **Total validation exemption for thematic units** — Thematic MAA bypassed pool membership, prefix, class, and cultural checks.
4. **Ungated global fuzzy matching** — Low-confidence fallback matches outside any cultural or class gate.
5. **Hardcoded keyword prompt gate** — `thematic_keywords` tuple diverged from `CK3_MAA_TYPES_THEMATIC` classification set.

## Decision

### Phase 2: Cultural Context Injection

Thematic MAA prompts now include faction cultural context:
- `faction_to_subculture_map` — faction subculture
- `faction_to_heritage_map` — faction heritage  
- `faction_to_prefix_map` — dominant cultural unit-key prefixes
- These are threaded through `run_llm_unit_assignment_pass` → `_build_batch_request_object` → `_build_unit_replacement_prompt`

**Files changed:** `mapper_tools/llm_helper.py`, `mapper_tools/llm_orchestrator.py`

### Phase 3: Consensus and Override Gating

Majority-vote and tournament corrections are now gated against:
- `json_excluded_units_maa` — per-MAA-type JSON exclusions
- `json_excluded_units_core` — core tag JSON exclusions
- `known_bad_units` — mechanically flagged bad units

If a suggested unit fails any gate, the correction is rejected and logged.

**Files changed:** `mapper_tools/llm_orchestrator.py` (`_apply_roster_corrections`)

### Phase 4: Bounded Thematic Validation

Thematic units remain exempt from culture-only/prefix restrictions but are subject to:
- Class/category two-tier gating (same broad category accepted, cross-category rejected)
- Exclusion-set checks

Expected Attila classes per MAA type (`maa_type_classes`) are rendered into review prompts for first-attempt compliance.

**Files changed:** `mapper_tools/llm_orchestrator.py`, `mapper_tools/llm_helper.py`

### Phase 5: Gated Global Fuzzy Match

Global fuzzy matching fallback for thematic MAA is gated at:
- Threshold ≥ 0.85 (Levenshtein ratio)
- Same broad category check (`get_class_category`)

Non-thematic global fallbacks remain flagged for fallback review (existing behavior).

**Files changed:** `mapper_tools/unit_management.py`, `mapper_tools/processing_passes.py`

### Phase 6: Classification-Driven Thematic Gate

Removed hardcoded `thematic_keywords` tuple from `_build_roster_review_prompt`. Thematic identification is now driven by:
- `CK3_MAA_TYPES_THEMATIC` set
- `classification_map` lookup

**Files changed:** `mapper_tools/llm_helper.py`  
`PROMPT_VERSION` bumped from `"6"` to `"7"` to invalidate stale cache entries.

### Phase 7: Fallback Review and Vector DB

- Thematic global-pool picks flagged with `fallback_tracking.mark_fallback(element, 'thematic_global_pool')`
- Fallback review round audits flagged thematic units
- Vector DB semantic search added as prioritized candidate tier for thematic MAA before fuzzy fallback

**Files changed:** `mapper_tools/processing_passes.py`, `mapper_tools/unit_management.py`, `mapper_tools/llm_orchestrator.py`

## Consequences

### Positive

- Thematic MAA mappings remain LLM-driven but are culturally grounded, class-gated, and JSON-override-safe
- JSON overrides are preserved from majority-vote consensus overwrites
- Thematic units can still be cross-faction reused (by design) but are now class/category bounded
- Stale cache entries are invalidated via `PROMPT_VERSION` bump

### Negative / Trade-offs

- Increased prompt token usage due to cultural context injection (mitigated by truncating candidate lists)
- Classification map must be populated by `run_llm_maa_classification_pass` before thematic unit assignment
- Thematic global-pool picks require fallback review round to run

### Risks

- Over-gating could break intended cross-faction reuse for thematic units (mitigated: two-tier gating allows same broad category)
- Stale classification map could lead to incorrect thematic/non-thematic identification (mitigated: fallback to `CK3_MAA_TYPES_THEMATIC` set)

## Related Documentation

- `.cecli.plans.md` — Section 16 (Documentation) checklist
- `mapper_tools/README.md` — Faction Culture Map JSON documentation
- `PROMPT_VERSION` comment block in `mapper_tools/llm_helper.py`
- Test cases TC-01 through TC-08 in `.cecli.plans.md` Section 11

## When to Revisit

- If `CK3_MAA_TYPES_THEMATIC` changes, update the classification-driven gate accordingly
- If new thematic MAA types are added, ensure they are classified via `run_llm_maa_classification_pass`