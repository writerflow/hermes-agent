"""WRI-2639 Part A T4: the reconcile seam and the fail-closed due action."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_wakes as wakes
from hermes_cli import kanban_control
from tests.hermes_cli._timed_wake_support import (
    DUE, FUTURE, NOW, action_generations, board, events, park, task_columns, tick, wake_rows,
)


def test_due_wake_becomes_due_held_with_one_event_and_card_stays_scheduled(board):
    task_id = park(board)

    result = tick(board)

    (row,) = wake_rows(board, task_id)
    assert row["state"] == "due_held" and row["held_reason"] == "release_guard_unavailable"
    held = events(board, task_id, {"wake_due_held"})
    assert len(held) == 1
    payload = held[0][2]
    assert payload["wake_id"] == row["wake_id"] and payload["late_by_seconds"] == 100
    assert row["due_event_id"] == held[0][0]
    assert kb.get_task(board, task_id).status == "scheduled"
    assert result.wakes["due_held"] == [row["wake_id"]]


def test_future_wake_is_untouched(board):
    task_id = park(board, wake_at=FUTURE)
    before = (wake_rows(board), events(board, task_id))

    tick(board)

    assert (wake_rows(board), events(board, task_id)) == before


def test_part_a_changes_only_action_generation_by_wake_event_count(board):
    # RA4: run on reconcile_timed_wakes directly, on a board holding only wake cards.
    ids = [park(board, f"c{i}", wake_at=DUE - i) for i in range(3)] + [park(board, "later", wake_at=FUTURE)]
    cols_before = task_columns(board)
    gen_before = action_generations(board)
    before_counts = {t: len(events(board, t, {"wake_due_held", "wake_superseded", "wake_invalid"})) for t in ids}

    tick(board)

    assert task_columns(board) == cols_before
    for task_id in ids:
        added = len(events(board, task_id, {"wake_due_held", "wake_superseded", "wake_invalid"})) - before_counts[task_id]
        assert action_generations(board)[task_id] - gen_before[task_id] == added
    assert all(c["park_generation"] == cols_before[t]["park_generation"] for t, c in task_columns(board).items())


def test_dry_run_tick_writes_nothing_and_reports(board):
    task_id = park(board)
    before = (wake_rows(board), events(board, task_id), action_generations(board))

    result = tick(board, dry_run=True)

    assert (wake_rows(board), events(board, task_id), action_generations(board)) == before
    assert result.wakes["would_due_held"] == [wake_rows(board)[0]["wake_id"]]


def test_paused_board_leaves_wake_armed_and_resume_catches_up(board):
    task_id = park(board)
    kanban_control.set_control_scope("paused", changed_by="test", reason="hold")

    held = tick(board)
    assert wake_rows(board)[0]["state"] == "armed" and held.wakes["control_held"] == 1

    kanban_control.set_control_scope("running", changed_by="test", reason="go")
    tick(board)
    assert wake_rows(board)[0]["state"] == "due_held"
    assert len(events(board, task_id, {"wake_due_held"})) == 1


def test_invalid_stored_row_reported_once_and_card_untouched(board):
    task_id = park(board, wake_at=FUTURE)
    board.execute("UPDATE task_wakes SET wake_at = 'garbage'")
    board.commit()
    cols = task_columns(board)

    first = tick(board)
    tick(board)

    assert len(events(board, task_id, {"wake_invalid"})) == 1
    assert first.wakes["invalid"] == [wake_rows(board)[0]["wake_id"]]
    assert task_columns(board) == cols
    assert wake_rows(board)[0]["state"] == "armed"


def test_dispatch_once_runs_reconcile_after_recompute_ready_before_lanes(board, monkeypatch, tmp_path):
    _spawnable(monkeypatch, tmp_path)
    task_id = park(board)
    ready_id = kb.create_task(board, title="ready", assignee="worker")

    result = kbd.dispatch_once(board, spawn_fn=lambda task, workspace: 1234)

    assert result.wakes.get("due_held")
    assert kb.get_task(board, task_id).status == "scheduled"
    assert [s[0] for s in result.spawned] == [ready_id]


def _spawnable(monkeypatch, tmp_path):
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda profile: True)
    monkeypatch.setattr(kbd._kbw, "resolve_workspace", lambda task, board=None: str(tmp_path))


def test_one_reconciler_failure_does_not_abort_the_tick(board, monkeypatch, tmp_path):
    _spawnable(monkeypatch, tmp_path)
    park(board)
    ready_id = kb.create_task(board, title="ready", assignee="worker")
    monkeypatch.setattr(wakes, "reconcile_timed_wakes",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    result = kbd.dispatch_once(board, spawn_fn=lambda task, workspace: 1234)

    assert result.wait_reconcile_errors == 1
    assert [s[0] for s in result.spawned] == [ready_id]


def test_describe_suppression_mentions_wake_due_held(board):
    park(board)
    result = kbd.DispatchResult()
    wakes.reconcile_timed_wakes(board, result, now=NOW)

    assert "wake_due_held=1" in kbd.describe_suppression([result])


def test_due_held_backlog_does_not_starve_new_armed_rows(board):
    old = [park(board, f"old{i}", wake_at=DUE - 1000 - i) for i in range(3)]
    tick(board)
    assert {r["state"] for r in wake_rows(board)} == {"due_held"}
    fresh = park(board, "fresh", wake_at=DUE)

    tick(board, sweep_limit=1)

    assert [r["state"] for r in wake_rows(board, fresh)] == ["due_held"]
    assert old


def test_reconcile_on_in_memory_board_skips_and_counts():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    wakes.ensure_schema(conn)
    conn.execute(
        "INSERT INTO task_wakes (task_id, park_event_id, park_generation, wake_at, wake_input, "
        "state, armed_by, armed_at, arm_operation_id) VALUES ('t', 1, 0, 1, 'x', 'armed', 'a', 1, 'op')"
    )

    result = kbd.DispatchResult()
    wakes.reconcile_timed_wakes(conn, result, now=NOW)

    assert result.wakes == {"skipped_non_file_board": 1}
    assert conn.execute("SELECT state FROM task_wakes").fetchone()[0] == "armed"
