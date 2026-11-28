"""WRI-2639 Part A: arming a typed wake time with a park (T1) and refusing bad input (T2)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_task_state as task_state


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


def _wakes(conn, task_id=None):
    sql = "SELECT * FROM task_wakes"
    args = ()
    if task_id:
        sql += " WHERE task_id = ?"
        args = (task_id,)
    return [dict(r) for r in conn.execute(sql + " ORDER BY wake_id", args)]


def _events(conn, task_id):
    return [
        (e.id, e.kind, e.payload)
        for e in kb.list_events(conn, task_id)
    ]


def _schedule(conn, task_id, *, op="park:1", wake_at=None, wake_input="in test", **kw):
    return task_state.schedule_task_transition(
        conn, task_id, actor_scope="operator", operation_id=op, reason="wait",
        wake_at=wake_at, wake_input=wake_input if wake_at is not None else None, **kw,
    )


def _snapshot(conn, task_id):
    return (
        tuple(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()),
        _events(conn, task_id),
        _wakes(conn, task_id),
        [tuple(r) for r in conn.execute("SELECT * FROM task_runs WHERE task_id = ?", (task_id,))],
    )


# ---------------------------------------------------------------- T1

def test_schedule_with_wake_at_persists_row_and_event_atomically(board):
    task_id = kb.create_task(board, title="parked", assignee="worker")
    wake_at = int(time.time()) + 3600

    outcome = _schedule(board, task_id, wake_at=wake_at)

    assert outcome.committed is True
    with kbc.connect_closing() as other:
        task = other.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        rows = _wakes(other, task_id)
        scheduled = [e for e in _events(other, task_id) if e[1] == "scheduled"]
    assert task["status"] == "scheduled"
    assert len(rows) == 1
    row = rows[0]
    assert row["state"] == "armed"
    assert row["wake_at"] == wake_at
    assert len(scheduled) == 1
    assert scheduled[0][2] == {"reason": "wait", "wake_at": wake_at}
    assert row["park_event_id"] == scheduled[0][0]
    assert row["park_generation"] == task["park_generation"]
    assert row["armed_by"] == "operator"
    assert row["arm_operation_id"] == "park:1"


def test_plain_schedule_creates_no_row_and_payload_unchanged(board):
    task_id = kb.create_task(board, title="plain", assignee="worker")

    assert _schedule(board, task_id).committed is True

    assert _wakes(board) == []
    scheduled = [e for e in _events(board, task_id) if e[1] == "scheduled"]
    assert [e[2] for e in scheduled] == [{"reason": "wait"}]


def test_arm_rolls_back_with_park_when_row_insert_fails(board, monkeypatch):
    from hermes_cli import kanban_db_wakes

    task_id = kb.create_task(board, title="rollback", assignee="worker")
    before = _snapshot(board, task_id)

    def boom(*args, **kwargs):
        raise RuntimeError("injected insert failure")

    monkeypatch.setattr(kanban_db_wakes, "arm_in_transaction", boom)
    with pytest.raises(RuntimeError, match="injected insert failure"):
        _schedule(board, task_id, wake_at=int(time.time()) + 3600)

    with kbc.connect_closing() as other:
        assert _snapshot(other, task_id) == before
        assert other.execute(
            "SELECT COUNT(*) FROM task_transition_receipts WHERE operation_id = 'park:1'"
        ).fetchone()[0] == 0


def test_arm_replays_same_operation_id_once(board):
    task_id = kb.create_task(board, title="replay", assignee="worker")
    wake_at = int(time.time()) + 3600

    first = _schedule(board, task_id, wake_at=wake_at)
    replay = _schedule(board, task_id, wake_at=wake_at)

    assert replay == first
    assert len(_wakes(board, task_id)) == 1
    assert [e[1] for e in _events(board, task_id)].count("scheduled") == 1
    changed = _schedule(board, task_id, wake_at=wake_at + 60)
    assert changed.committed is False
    assert changed.refusal == "operation_id_collision"


def test_park_generation_stored_matches_after_end_run(board):
    task_id = kb.create_task(board, title="running", assignee="worker")
    assert kb.claim_task(board, task_id, ttl_seconds=None) is not None
    run_id = board.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()[0]
    assert run_id is not None

    outcome = _schedule(board, task_id, wake_at=int(time.time()) + 3600, expected_run_id=run_id)

    assert outcome.committed is True
    with kbc.connect_closing() as other:
        generation = other.execute(
            "SELECT park_generation FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()[0]
        (row,) = _wakes(other, task_id)
    assert row["park_generation"] == generation


def test_rearming_supersedes_the_previous_live_row(board):
    task_id = kb.create_task(board, title="rearm", assignee="worker")
    first_at = int(time.time()) + 3600
    assert _schedule(board, task_id, op="park:1", wake_at=first_at).committed
    # Leave Scheduled without closing the row (the reconciler closes it lazily).
    assert kb.unblock_task(board, task_id)
    assert _schedule(board, task_id, op="park:2", wake_at=first_at + 600).committed

    rows = _wakes(board, task_id)
    assert [(r["state"], r["resolution"]) for r in rows] == [
        ("superseded", "rearmed"),
        ("armed", None),
    ]



# ---------------------------------------------------------------- T2

from datetime import datetime, timezone, UTC

from hermes_cli import kanban_db_wakes as wakes

NOW = int(datetime(2026, 11, 27, 12, 0, 0, tzinfo=UTC).timestamp())


@pytest.mark.parametrize(
    "text, epoch",
    [
        ("2026-11-30T15:00:00Z", int(datetime(2026, 11, 30, 15, tzinfo=UTC).timestamp())),
        ("2026-11-30T09:00:00-06:00", int(datetime(2026, 11, 30, 15, tzinfo=UTC).timestamp())),
        ("2026-11-30T20:30:00+05:30", int(datetime(2026, 11, 30, 15, tzinfo=UTC).timestamp())),
        ("2026-11-30T15:00Z", int(datetime(2026, 11, 30, 15, tzinfo=UTC).timestamp())),
    ],
)
def test_validation_accepts(text, epoch):
    parsed = wakes.parse_wake_at(text, now=NOW)
    assert (parsed.epoch, parsed.text) == (epoch, text)


@pytest.mark.parametrize(
    "value, code",
    [
        ("2026-11-30T15:00:00", "wake_at_ambiguous_timezone"),
        ("2026-11-30T15:00", "wake_at_ambiguous_timezone"),
        ("2026-11-30", "wake_at_format"),
        ("2026-11-30T15:00:00 CST", "wake_at_format"),
        ("2026-11-30T15:00:00 America/Chicago", "wake_at_format"),
        ("2026-11-30T15:00:00.500Z", "wake_at_format"),
        ("2026-11-30T15:00:60Z", "wake_at_invalid_datetime"),
        ("2026-13-30T15:00:00Z", "wake_at_invalid_datetime"),
        ("2026-11-30T15:00:00+24:00", "wake_at_bad_offset"),
        ("2026-11-30T15:00:00+14:30", "wake_at_bad_offset"),
        ("2026-11-30t15:00:00Z", "wake_at_format"),
        ("2026-11-30T15:00:00z", "wake_at_format"),
        (1795000000, "wake_at_not_string"),
        (1795000000.5, "wake_at_not_string"),
        (True, "wake_at_not_string"),
        (None, "wake_at_not_string"),
        ("", "wake_at_empty"),
        ("   ", "wake_at_empty"),
        ("2026-11-27T12:00:00Z", "wake_at_in_past"),
        ("2026-11-26T12:00:00Z", "wake_at_in_past"),
        ("2027-11-29T12:00:01Z", "wake_at_too_far"),
    ],
)
def test_validation_refuses(value, code):
    with pytest.raises(wakes.WakeInputError) as err:
        wakes.parse_wake_at(value, now=NOW)
    assert err.value.code == code


@pytest.mark.parametrize(
    "value",
    ["2026-11-30T15:00:00", "2026-11-30", "garbage", "2020-01-01T00:00:00Z", 5, True],
)
def test_validation_refusal_leaves_the_board_byte_identical(board, value):
    task_id = kb.create_task(board, title="refuse", assignee="worker")
    before = _snapshot(board, task_id)

    with pytest.raises(wakes.WakeInputError):
        kb.schedule_task(board, task_id, reason="wait", wake_at=value)

    assert _snapshot(board, task_id) == before
    assert board.execute("SELECT COUNT(*) FROM task_transition_receipts").fetchone()[0] == (
        board.execute(
            "SELECT COUNT(*) FROM task_transition_receipts WHERE operation_kind = 'create'"
        ).fetchone()[0]
    )


def test_legacy_schedule_accepts_a_typed_wake(board):
    task_id = kb.create_task(board, title="legacy api", assignee="worker")
    text = datetime.fromtimestamp(int(time.time()) + 7200, UTC).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    assert kb.schedule_task(board, task_id, reason="wait", wake_at=text) is True

    (row,) = _wakes(board, task_id)
    assert row["wake_input"] == text and row["state"] == "armed"


def test_legacy_marker_in_reason_is_not_interpreted(board):
    task_id = kb.create_task(board, title="prose", assignee="worker")

    assert kb.schedule_task(
        board, task_id, reason="SCHEDULED_UNTIL=2099-01-01T00:00:00Z waiting"
    )

    assert _wakes(board) == []


def test_sticky_blocked_source_refuses_a_wake_and_changes_nothing(board):
    task_id = kb.create_task(board, title="held", assignee="worker")
    assert kb.block_task(board, task_id, kind="needs_input", reason="need a decision")
    before = _snapshot(board, task_id)

    outcome = _schedule(board, task_id, op="park:held", wake_at=int(time.time()) + 3600)

    assert (outcome.committed, outcome.refusal) == (False, "wake_hold_outstanding")
    assert _snapshot(board, task_id) == before
