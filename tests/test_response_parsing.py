"""Runnable self-check for llm_helper response-text handling.

Covers _get_response_text (.content -> .reasoning_content fallback) and
_extract_json_block (prose + fenced ```json stripping) — the reasoning-model
failure where the answer arrives only in .reasoning_content wrapped in prose.

Run: python tests/test_response_parsing.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mapper_tools.llm_helper import _extract_json_block, _get_response_text


class _FakeMessage:
    """Minimal stand-in for a litellm chat completion message."""

    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


def main():
    # 1. .content wins when populated.
    assert _get_response_text(_FakeMessage(content='{"a": 1}', reasoning_content='thinking')) == '{"a": 1}'

    # 2. Falls back to .reasoning_content when .content is None or empty.
    assert _get_response_text(_FakeMessage(content=None, reasoning_content='answer')) == 'answer'
    assert _get_response_text(_FakeMessage(content='', reasoning_content='answer')) == 'answer'
    assert _get_response_text(_FakeMessage()) is None

    # 3. Prose + fenced ```json block (the reported .reasoning_content shape)
    #    is stripped down to the bare JSON payload.
    prose = (
        'The faction "Langobard Separatists" refers to the Lombards (Langobards), '
        'who were a Germanic tribe.\n\n'
        '```json\n{\n  "chosen_group_key": "East Germanic"\n}\n```\n'
    )
    assert _extract_json_block(prose) == '{\n  "chosen_group_key": "East Germanic"\n}'

    # 4. Bare JSON (the normal case) passes through untouched.
    assert _extract_json_block('{"a": 1}') == '{"a": 1}'
    # 5. Fences without the 'json' tag are handled too.
    assert _extract_json_block('blah\n```\n{"a": 1}\n```') == '{"a": 1}'
    # 6. Unterminated fence (truncated response) yields the JSON tail.
    assert _extract_json_block('```json\n{"a": 1}') == '{"a": 1}'
    # 7. Non-string payloads (provider content-block lists) pass through.
    assert _extract_json_block(None) is None
    assert _extract_json_block(["block"]) == ["block"]

    print("All response-parsing checks passed.")


if __name__ == "__main__":
    main()
