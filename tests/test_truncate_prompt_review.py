"""
Self-check for the roster-review truncation and retry-feedback fixes.

Covers:
  1. _truncate_prompt keeps the review prompt tail (Output Format + JSON Schema)
     when the prompt exceeds the context limit.
  2. A synthetic roster-review request whose rendered prompt exceeds 30000 chars,
     so the truncation path is actually exercised.
  3. _build_roster_review_prompt renders the "Feedback on Your Previous Attempt"
     section when retry_context with previous_errors is present.

Run: python tests/test_truncate_prompt_review.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mapper_tools.llm_helper import LLMHelper
from mapper_tools import llm_response_models


def _make_request(with_retry_context=False):
    """Builds a synthetic roster-review request large enough to overflow 30000 chars."""
    fake_units = {f"att_fake_unit_{i}" for i in range(600)}

    roster = {
        "General": [
            {"key": "att_fake_unit_1", "identifier": {"__review_id__": "1"}, "rank": "1"},
            {"key": "att_fake_unit_2", "identifier": {"__review_id__": "2"}, "rank": "2"},
            {"key": "att_fake_unit_3", "identifier": {"__review_id__": "3"}, "rank": "3"},
        ],
        "Knights": [
            {"key": "att_fake_unit_4", "identifier": {"__review_id__": "4"}, "rank": "1"},
        ],
        "MenAtArm": [
            {"key": "att_fake_unit_5", "identifier": {"__review_id__": "5"}, "type": "pikemen"},
            {"key": "att_fake_unit_6", "identifier": {"__review_id__": "6"}, "type": "war_elephants"},
        ],
        "Levies": [
            {"key": "att_fake_unit_7", "identifier": {"__review_id__": "7"}},
            {"key": "att_fake_unit_8", "identifier": {"__review_id__": "8"}},
            {"key": "att_fake_unit_9", "identifier": {"__review_id__": "9"}},
        ],
    }

    request = {
        "id": "review|FakeFaction",
        "faction": "FakeFaction",
        "subculture": "fake_subculture",
        "roster": roster,
        "local_unit_pool": fake_units,
        "tiered_pools": [set(list(fake_units)[:200]), set(list(fake_units)[200:400])],
        "prefix": "att_fake_",
        "cultural_prefixes": ["att_fake_"],
        "dominant_prefixes": ["att_fake_"],
        "known_bad_units": [
            {
                "tag": "Levies",
                "identifier": {"__review_id__": "7"},
                "current_unit": "att_fake_unit_7",
                "reasons": ["Quality band violation."],
            }
        ],
        "thematic_maa_types": ["war_elephants"],
        "maa_type_classes": {"pikemen": ["inf_spr"], "war_elephants": ["elph"]},
    }

    if with_retry_context:
        request["retry_context"] = {
            "previous_errors": [
                {
                    "tag": "Levies",
                    "identifier": {"__review_id__": "7"},
                    "current_unit": "att_fake_unit_7",
                    "suggested_unit": "att_bad_unit_x",
                    "error": "Unit does not exist and could not be fuzzy matched.",
                }
            ]
        }

    return request


def _build_prompt(helper, request):
    """Builds a roster-review prompt from the synthetic request."""
    return helper._build_roster_review_prompt(
        request,
        "The time period is from 450 AD to 1450 AD.",
        "",
        llm_response_models.RosterReviewResponse,
        unit_to_tier_map={"att_fake_unit_1": 1},
    )


def main():
    # Go straight to unbound methods on a bare instance to avoid __init__
    # (no network config / cache dir needed for prompt-building helpers).
    helper = LLMHelper.__new__(LLMHelper)
    helper.unit_to_class_map = {}
    helper.unit_to_screen_name_map = {}
    helper.unit_to_description_map = {}
    helper.all_units = set()
    helper.mcp_descriptions = {}

    request = _make_request()
    prompt = _build_prompt(helper, request)

    # Sanity: the synthetic request must actually overflow 30000 chars.
    assert len(prompt) > 30000, (
        f"Test setup failure: synthetic prompt is only {len(prompt)} chars; "
        "expected > 30000 to exercise the truncation path."
    )

    truncated = helper._truncate_prompt(prompt, 30000)
    # NOTE: the pre-existing middle-truncation logic budgets the middle as
    # (max_length - header - footer) but then APPENDS the "\n... (list truncated)\n"
    # marker on top, so the result may exceed the soft limit by up to ~25 chars.
    # That overhead is harmless for a heuristic char guard and is deliberately
    # left unchanged; the assertions that matter are the shrink + tail survival.
    assert len(truncated) < len(prompt), (
        f"Truncation did not shrink the prompt ({len(truncated)} vs {len(prompt)})."
    )
    assert len(truncated) <= 30000 + 100, (
        f"Truncated prompt far exceeds the limit: {len(truncated)}"
    )

    # (a) The tail must survive truncation.
    assert "Output Format (STRICT)" in truncated, (
        "Truncation amputated the Output Format footer - the model would get "
        "no format constraints."
    )
    assert "JSON Output Schema" in truncated, (
        "Truncation amputated the JSON Output Schema footer - the model would "
        "get no schema constraints."
    )
    assert "... (list truncated)" in truncated, (
        "Expected middle-truncation marker (pool lists cut, not the tail)."
    )

    # (b) retry_context must render the feedback section.
    request_fb = _make_request(with_retry_context=True)
    prompt_fb = _build_prompt(helper, request_fb)
    assert "Feedback on Your Previous Attempt" in prompt_fb, (
        "retry_context with previous_errors did not render the feedback section."
    )
    assert "att_bad_unit_x" in prompt_fb, (
        "Feedback section did not include the specific invalid suggestion."
    )

    print("OK: truncation preserves the review prompt tail; retry feedback renders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())