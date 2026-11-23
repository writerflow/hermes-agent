"""AST guard: no sync ``with owner_scope(...)`` directly in an ``async def`` body under ``gateway/``.

On a multiplexed gateway the sync ``owner_scope`` reads the profile ``.env``, ``config.yaml`` and
terminal policy. In a coroutine that read runs on the event-loop thread. Coroutines use
``async with async_owner_scope(...)`` (read in a worker thread) or ``prepare_owner_scope`` followed by
``bind_prepared_owner_scope`` where a check-to-use boundary must not gain a suspension.

What this guard cannot see (not covered, by design):
- ``owner_scope`` reached through a sync helper that a coroutine calls (the many sync ``def`` callers,
  e.g. ``session_local.py`` and ``session_policy.py``, are outside it);
- a scope object entered through ``ExitStack`` or stored and entered later;
- reads that happen inside an otherwise correct scope (jobs file, gateway config, session DB).
It also does not look inside nested sync ``def``, ``lambda`` or ``class`` bodies: those are not on the
loop by construction of the AST walk.

``ALLOW_LIST`` stays empty. Any future entry needs a reason written next to it.
"""

import ast
from pathlib import Path

GATEWAY = Path(__file__).resolve().parents[2] / "gateway"

# file:line -> reason. Empty on purpose.
ALLOW_LIST: dict = {}


def sync_owner_scope_in_async(source: str) -> list:
    """Line numbers of every sync ``with owner_scope(...)`` directly in an ``async def`` body."""
    found = []

    def walk(node, in_async):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.AsyncFunctionDef):
                walk(child, True)
            elif isinstance(child, (ast.FunctionDef, ast.Lambda, ast.ClassDef)):
                walk(child, False)
            else:
                if in_async and isinstance(child, ast.With):
                    for item in child.items:
                        call = item.context_expr
                        if isinstance(call, ast.Call):
                            func = call.func
                            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                            if name == "owner_scope":
                                found.append(child.lineno)
                walk(child, in_async)

    walk(ast.parse(source), False)
    return found


def test_guard_detects_a_sync_scope_in_a_coroutine():
    src = "async def f(a):\n    with owner_scope(a):\n        pass\n"
    assert sync_owner_scope_in_async(src) == [2]
    src = "async def f(a):\n    with sessions.owner_scope(a):\n        pass\n"
    assert sync_owner_scope_in_async(src) == [2]


def test_guard_ignores_the_async_form_and_plain_functions():
    assert sync_owner_scope_in_async("async def f(a):\n    async with async_owner_scope(a):\n        pass\n") == []
    assert sync_owner_scope_in_async("def f(a):\n    with owner_scope(a):\n        pass\n") == []
    nested = "async def f(a):\n    def g():\n        with owner_scope(a):\n            pass\n"
    assert sync_owner_scope_in_async(nested) == []
    prepared = "async def f(a, p):\n    with bind_prepared_owner_scope(a, p):\n        pass\n"
    assert sync_owner_scope_in_async(prepared) == []


def test_no_sync_owner_scope_in_a_gateway_coroutine():
    found = []
    for path in sorted(GATEWAY.rglob("*.py")):
        rel = path.relative_to(GATEWAY.parent).as_posix()
        for line in sync_owner_scope_in_async(path.read_text(encoding="utf-8")):
            if f"{rel}:{line}" not in ALLOW_LIST:
                found.append(f"{rel}:{line}")
    assert found == [], "sync owner_scope in an async def (use async_owner_scope or prepare/bind): " + ", ".join(found)
