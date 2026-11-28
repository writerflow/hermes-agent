"""WRI-2639 Part A T3: who may arm a wake (own card; operator context for any parked card)."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone, UTC
from pathlib import Path

import pytest

from agent import delegation_context
from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_wakes as wakes


def _future(seconds=7200):
    return datetime.fromtimestamp(int(time.time()) + seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return h


@pytest.fixture
def worker(home, monkeypatch):
    with kbc.connect_closing() as conn:
        mine = kb.create_task(conn, title="mine", assignee="test-worker")
        kb.claim_task(conn, mine)
        run_id = kb._current_run_id(conn, mine)
        sibling = kb.create_task(conn, title="sibling", assignee="peer")
        kb.schedule_task(conn, sibling, reason="parked")
    monkeypatch.setenv("HERMES_KANBAN_TASK", mine)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return mine, sibling


def _rows():
    with kbc.connect_closing() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM task_wakes ORDER BY wake_id")]


def _parked(title="parked"):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title=title, assignee="peer")
        kb.schedule_task(conn, task_id, reason="parked")
    return task_id


def _wake(task_id, at=None, expect=None):
    return kc._cmd_wake(argparse.Namespace(task_id=task_id, at=at or _future(), expect_park=expect))


# ----------------------------------------------------------------- tool entry point

def test_schedule_wake_at_arms_own_card_with_run_binding(worker):
    from tools import kanban_tools as kt

    mine, _ = worker
    out = json.loads(kt._handle_schedule({"reason": "wait for CI", "wake_at": _future()}))

    assert out["ok"] is True and out["status"] == "scheduled" and out["wake_at"]
    (row,) = _rows()
    assert (row["task_id"], row["state"]) == (mine, "armed")


def test_worker_cannot_arm_wake_on_sibling_card(worker):
    from tools import kanban_tools as kt

    _, sibling = worker
    out = json.loads(kt._handle_schedule({"task_id": sibling, "reason": "x", "wake_at": _future()}))

    assert out.get("error")
    assert _rows() == []


def test_schedule_wake_at_refused_from_sticky_blocked_source(worker):
    from tools import kanban_tools as kt

    mine, _ = worker
    with kbc.connect_closing() as conn:
        assert kb.block_task(conn, mine, kind="needs_input", reason="human needed")

    out = json.loads(kt._handle_schedule({"reason": "x", "wake_at": _future()}))

    assert out.get("error") and _rows() == []
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, mine).status == "blocked"


def test_schedule_with_bad_wake_at_parks_nothing(worker):
    from tools import kanban_tools as kt

    mine, _ = worker
    out = json.loads(kt._handle_schedule({"reason": "x", "wake_at": "2030-01-01T00:00:00"}))

    assert "wake_at_ambiguous_timezone" in out["error"]
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, mine).status == "running"


def test_tool_refuses_wake_on_another_card_when_variable_is_inherited_by_cron(worker, monkeypatch):
    # RA3 (tool entry point): a set HERMES_KANBAN_TASK is never an operator context,
    # even when no dispatcher-owned worker is running (a cron job that inherited it).
    from tools import kanban_tools as kt

    _, sibling = worker
    monkeypatch.setattr(delegation_context, "is_dispatcher_owned_worker_context", lambda: False)

    out = json.loads(kt._handle_schedule({"task_id": sibling, "reason": "x", "wake_at": _future()}))

    assert wakes.operator_context() is False
    assert out.get("error") and _rows() == []


def _ready_sibling():
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title="schedulable sibling", assignee="peer")


def _status_and_rows(task_id):
    with kbc.connect_closing() as conn:
        return kb.get_task(conn, task_id).status, len(_rows())


def test_tool_inherited_variable_cannot_arm_schedulable_sibling(worker, monkeypatch):
    # RA3: the sibling is ready, so only the arm predicate can refuse it (a Scheduled one is
    # refused by schedule_task regardless).
    from tools import kanban_tools as kt

    monkeypatch.setattr(delegation_context, "is_dispatcher_owned_worker_context", lambda: False)
    sibling = _ready_sibling()

    out = json.loads(kt._handle_schedule({"task_id": sibling, "reason": "x", "wake_at": _future()}))

    assert out.get("error")
    assert _status_and_rows(sibling) == ("ready", 0)


def test_cli_schedule_inherited_variable_cannot_arm_schedulable_sibling(worker, monkeypatch):
    monkeypatch.setattr(delegation_context, "is_dispatcher_owned_worker_context", lambda: False)
    sibling = _ready_sibling()

    rc = kc._cmd_schedule(argparse.Namespace(task_id=sibling, reason=["x"], ids=None, wake_at=_future()))

    assert rc != 0
    assert _status_and_rows(sibling) == ("ready", 0)


# --------------------------------------------------------------------- CLI

def test_cli_schedule_wake_at_flag(home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="cli", assignee="w")

    rc = kc._cmd_schedule(argparse.Namespace(task_id=task_id, reason=["wait"], ids=None, wake_at=_future()))

    assert rc == 0 and [r["task_id"] for r in _rows()] == [task_id]


def test_cli_schedule_bad_wake_at_changes_nothing(home, capsys):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="cli", assignee="w")

    rc = kc._cmd_schedule(argparse.Namespace(task_id=task_id, reason=["wait"], ids=None, wake_at="tomorrow"))

    assert rc != 0 and "wake_at refused" in capsys.readouterr().err and _rows() == []
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "ready"


def test_cli_wake_arms_existing_park_in_operator_context(home):
    task_id = _parked()

    assert _wake(task_id) == 0

    (row,) = _rows()
    assert (row["task_id"], row["state"]) == (task_id, "armed")
    with kbc.connect_closing() as conn:
        assert [e.kind for e in kb.list_events(conn, task_id)].count("wake_armed") == 1


def test_cli_wake_refused_when_hermes_kanban_task_set(home, monkeypatch, capsys):
    task_id = _parked()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_other")

    assert _wake(task_id) != 0
    assert "operator-only" in capsys.readouterr().err and _rows() == []


def test_cli_wake_refused_in_cron_that_inherited_a_worker_variable(home, monkeypatch):
    # RA3: the variable is set but no dispatcher-owned worker is running (a cron job).
    task_id = _parked()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_inherited")
    monkeypatch.setattr(delegation_context, "is_dispatcher_owned_worker_context", lambda: False)

    assert _wake(task_id) != 0 and _rows() == []


def test_cli_wake_refused_in_delegated_child(home):
    task_id = _parked()
    with delegation_context.delegated_child_context("s1"):
        assert wakes.operator_context() is False
        assert _wake(task_id) != 0
    assert _rows() == []


def test_cli_wake_refuses_not_scheduled_and_park_changed_and_sticky(home, capsys):
    with kbc.connect_closing() as conn:
        todo = kb.create_task(conn, title="todo", assignee="w")
        held = kb.create_task(conn, title="held", assignee="w")
        kb.block_task(conn, held, kind="needs_input", reason="human")
    parked = _parked()
    with kbc.connect_closing() as conn:
        park_event = max(e.id for e in kb.list_events(conn, parked) if e.kind == "scheduled")

    assert _wake(todo) != 0
    assert _wake(held) != 0
    assert _wake(parked, expect=park_event + 99) != 0
    assert _rows() == []
    assert _wake(parked, expect=park_event) == 0


def test_cli_wake_rearm_supersedes_previous_row(home):
    task_id = _parked()
    assert _wake(task_id, _future(3600)) == 0
    assert _wake(task_id, _future(7200)) == 0

    states = [r["state"] for r in _rows()]
    assert states == ["superseded", "armed"]


def test_cli_wake_migrate_is_an_unknown_command(home):
    assert "wake-migrate" not in kc._HANDLERS
    assert {"wake", "wakes"} <= set(kc._HANDLERS)
