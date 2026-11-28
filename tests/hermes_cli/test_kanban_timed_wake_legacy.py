"""WRI-2639 Part A T7 (RA2 deviation): legacy ``SCHEDULED_UNTIL=`` reasons are inert and only reported.

No dispatcher-written ``wake_legacy_marker_seen`` event exists in this PR.
"""

from __future__ import annotations

import argparse
import json

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_wakes as wakes
from tests.hermes_cli._timed_wake_support import (
    FUTURE, NOW, action_generations, events, park, task_columns, tick, wake_rows,
)
from tests.hermes_cli import _timed_wake_support as _support

board = _support.board  # re-exported pytest fixture


def _legacy(conn, title="legacy", marker="SCHEDULED_UNTIL=2026-01-01T00:00:00Z"):
    task_id = kb.create_task(conn, title=title, assignee="w")
    assert kb.schedule_task(conn, task_id, reason=f"{marker} waiting")
    return task_id


def test_past_marker_in_reason_stays_scheduled_after_many_ticks(board):
    task_id = _legacy(board)

    for _ in range(5):
        tick(board)

    assert kb.get_task(board, task_id).status == "scheduled"
    assert wake_rows(board) == []


def test_wakes_legacy_report_lists_not_armed_with_display_parse_and_writes_nothing(board, capsys):
    task_id = _legacy(board)
    bad = _legacy(board, "bad", "SCHEDULED_UNTIL=tomorrow")
    before = (task_columns(board), action_generations(board), events(board, task_id))

    rc = kc._cmd_wakes(argparse.Namespace(state=None, legacy=True, json=True))

    out = {item["task_id"]: item for item in json.loads(capsys.readouterr().out)}
    assert rc == 0
    assert out[task_id]["status"] == "NOT ARMED" and out[task_id]["display_parse"].startswith("valid 2026-01-01")
    assert out[bad]["display_parse"].startswith("malformed")
    assert (task_columns(board), action_generations(board), events(board, task_id)) == before


def test_reconcile_tick_over_legacy_marker_writes_no_event_and_changes_no_column(board):
    # RA2: replaces the three once-per-park / no-reason-text / columns event tests.
    task_id = _legacy(board)
    cols, gen, ev = task_columns(board), action_generations(board), events(board, task_id)

    tick(board)
    tick(board)

    assert task_columns(board) == cols
    assert action_generations(board) == gen  # action_generation included
    assert events(board, task_id) == ev


def test_legacy_report_ignores_older_park_marker(board):
    task_id = _legacy(board)
    kb.unblock_task(board, task_id)
    assert kb.schedule_task(board, task_id, reason="no marker this time")

    assert wakes.legacy_marker_report(board) == []


def test_legacy_report_ignores_card_with_live_wake_row(board):
    task_id = kb.create_task(board, title="both", assignee="w")
    from hermes_cli import kanban_db_task_state as ts

    out = ts.schedule_task_transition(
        board, task_id, actor_scope="operator", operation_id="p",
        reason="SCHEDULED_UNTIL=2026-01-01T00:00:00Z", wake_at=FUTURE, wake_input="x",
    )
    assert out.committed

    assert wakes.legacy_marker_report(board) == []
