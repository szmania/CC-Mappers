"""Regression guards for the Windows event-loop policy.

Root cause reference: mapper_tools/langgraph_agent.py used to install
asyncio.WindowsSelectorEventLoopPolicy at import time. The Selector loop caps
Windows at select()'s FD_SETSIZE (512 sockets), so long LLM review runs died
with "ValueError: too many file descriptors in select()". Entry points now own
the policy (faction_fixer.main forces WindowsProactorEventLoopPolicy); this
module must never mutate it again.
"""

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LANGGRAPH_AGENT_PATH = REPO_ROOT / "mapper_tools" / "langgraph_agent.py"


def test_langgraph_agent_source_never_sets_event_loop_policy():
    """AST guard: langgraph_agent.py must never call asyncio.set_event_loop_policy()."""
    tree = ast.parse(
        LANGGRAPH_AGENT_PATH.read_text(encoding="utf-8"),
        filename=str(LANGGRAPH_AGENT_PATH),
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "set_event_loop_policy", (
                f"langgraph_agent.py line {node.lineno} calls set_event_loop_policy(); "
                "entry points own the event-loop policy (see faction_fixer.main)."
            )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only event-loop policy guard")
def test_importing_langgraph_agent_does_not_force_selector_policy():
    """Behavior guard: importing the module must not install a Selector policy on Windows."""
    import asyncio

    pytest.importorskip("mapper_tools.langgraph_agent")
    policy = asyncio.get_event_loop_policy()
    assert not isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy), (
        "Importing mapper_tools.langgraph_agent forced a Selector event-loop policy; "
        "long runs will crash with 'ValueError: too many file descriptors in select()'."
    )