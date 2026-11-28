"""Shared fixtures and helpers for the WRI-2639 timed-wake tests (not collected as tests)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_task_state as task_state
from hermes_cli import kanban_db_wakes as wakes

NOW = 1_795_000_000
DUE = NOW - 100
FUTURE = NOW + 3600


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


def wake_rows(conn, task_id=None):
    if task_id:
        rows = conn.execute("SELECT * FROM task_wakes WHERE task_id = ? ORDER BY wake_id", (task_id,))
    else:
        rows = conn.execute("SELECT * FROM task_wakes ORDER BY wake_id")
    return [dict(r) for r in rows]


def events(conn, task_id, kinds=None):
    return [(e.id, e.kind, e.payload) for e in kb.list_events(conn, task_id)
            if kinds is None or e.kind in kinds]


def park(conn, title="parked", *, wake_at=DUE, op=None, assignee="worker"):
    """Create a card and park it with an armed typed wake; returns the task id."""
    task_id = kb.create_task(conn, title=title, assignee=assignee)
    outcome = task_state.schedule_task_transition(
        conn, task_id, actor_scope="operator", operation_id=op or f"park:{task_id}",
        reason="wait", wake_at=wake_at, wake_input="test" if wake_at is not None else None,
    )
    assert outcome.committed, outcome
    return task_id


def tick(conn, *, now=NOW, **kw):
    """One reconcile with a fresh result object."""
    result = kbd.DispatchResult()
    wakes.reconcile_timed_wakes(conn, result, now=now, **kw)
    return result


def task_columns(conn, *, exclude=("action_generation",)):
    out = {}
    for row in conn.execute("SELECT * FROM tasks ORDER BY id"):
        data = dict(row)
        out[data["id"]] = {k: v for k, v in data.items() if k not in exclude}
    return out


def action_generations(conn):
    return {r["id"]: r["action_generation"]
            for r in conn.execute("SELECT id, action_generation FROM tasks")}
