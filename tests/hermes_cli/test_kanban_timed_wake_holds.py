"""WRI-2639 Part A T6: hold precedence and supersession of a due wake."""

from __future__ import annotations

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_task_state as task_state
from hermes_cli import kanban_db_wakes as wakes
from tests.hermes_cli._timed_wake_support import (
    DUE, FUTURE, NOW, action_generations, events, park, task_columns, tick, wake_rows,
)
from tests.hermes_cli import _timed_wake_support as _support

board = _support.board  # re-exported pytest fixture

WAKE_KINDS = {"wake_due_held", "wake_superseded", "wake_invalid"}


def _only_wake_events_changed(conn, task_ids, before_cols, before_gen, before_events):
    assert task_columns(conn) == before_cols
    gens = action_generations(conn)
    for task_id in task_ids:
        added = len(events(conn, task_id, WAKE_KINDS)) - before_events[task_id]
        assert gens[task_id] - before_gen[task_id] == added


def _run(conn, task_ids):
    cols, gen = task_columns(conn), action_generations(conn)
    ev = {t: len(events(conn, t, WAKE_KINDS)) for t in task_ids}
    result = tick(conn)
    _only_wake_events_changed(conn, task_ids, cols, gen, ev)
    return result


def _row(conn, task_id):
    return wake_rows(conn, task_id)[-1]


def test_repark_without_wake_supersedes_old_wake_and_never_fires(board):
    task_id = park(board, wake_at=FUTURE)
    kb.unblock_task(board, task_id)
    assert kb.schedule_task(board, task_id, reason="again")
    board.execute("UPDATE task_wakes SET wake_at = ?", (DUE,))
    board.commit()

    _run(board, [task_id])

    assert (_row(board, task_id)["state"], _row(board, task_id)["resolution"]) == ("superseded", "repark")
    assert events(board, task_id, {"wake_due_held"}) == []


def test_repark_with_new_wake_supersedes_old_and_unique_index_holds(board):
    task_id = park(board, wake_at=FUTURE)
    kb.unblock_task(board, task_id)
    out = task_state.schedule_task_transition(
        board, task_id, actor_scope="operator", operation_id="repark", reason="again",
        wake_at=FUTURE + 10, wake_input="x",
    )
    assert out.committed
    rows = wake_rows(board, task_id)
    assert [r["state"] for r in rows] == ["superseded", "armed"] and rows[0]["resolution"] == "rearmed"
    live = [r for r in rows if r["state"] in wakes.LIVE_STATES]
    assert len(live) == 1


def test_manual_unblock_then_running_card_untouched_row_closed_park_exited(board):
    task_id = park(board)
    board.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    board.commit()
    before = task_columns(board)

    tick(board)

    assert task_columns(board) == before
    assert (_row(board, task_id)["state"], _row(board, task_id)["resolution"]) == ("superseded", "park_exited")


def test_newer_block_never_released(board):
    task_id = park(board)
    kb.unblock_task(board, task_id)
    assert kb.block_task(board, task_id, kind="needs_input", reason="need a human")

    _run(board, [task_id])

    assert _row(board, task_id)["state"] == "superseded"
    assert kb.get_task(board, task_id).status == "blocked"


def test_creator_hold_source_refused_at_arm(board):
    task_id = kb.create_task(board, title="held", assignee="worker", initial_status="blocked")
    out = task_state.schedule_task_transition(
        board, task_id, actor_scope="operator", operation_id="p", reason="r", wake_at=FUTURE, wake_input="x",
    )
    assert (out.committed, out.refusal) == (False, "wake_hold_outstanding")
    assert wake_rows(board) == []


def test_explicit_blocked_event_after_arm_holds(board):
    task_id = park(board)
    kb._append_event(board, task_id, "blocked", {"kind": "needs_input", "reason": "late hold"})
    board.commit()

    _run(board, [task_id])

    assert (_row(board, task_id)["state"], _row(board, task_id)["held_reason"]) == ("due_held", "sticky_block")
    assert kb.get_task(board, task_id).status == "scheduled"


def test_live_claim_holds(board):
    task_id = park(board)
    board.execute("UPDATE tasks SET claim_lock = 'lock', claim_expires = ? WHERE id = ?", (NOW + 600, task_id))
    board.commit()

    tick(board)

    assert _row(board, task_id)["held_reason"] == "live_claim"
    assert kb.get_task(board, task_id).status == "scheduled"


def test_review_ownership_supersedes(board):
    task_id = park(board)
    board.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
    board.commit()
    tick(board)
    assert _row(board, task_id)["resolution"] == "park_exited"


def test_archived_supersedes(board):
    task_id = park(board)
    board.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (task_id,))
    board.commit()
    tick(board)
    assert _row(board, task_id)["resolution"] == "park_exited"


def test_deleted_task_row_superseded_without_card_event(board):
    task_id = park(board)
    board.execute("PRAGMA foreign_keys = OFF")
    board.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    board.commit()

    tick(board)

    row = wake_rows(board)[0]
    assert (row["state"], row["resolution"]) == ("superseded", "task_missing")
    assert board.execute("SELECT COUNT(*) FROM task_events WHERE kind = 'wake_superseded'").fetchone()[0] == 0


def test_reassign_while_parked_holds_park_generation_changed(board):
    task_id = park(board)
    board.execute("UPDATE tasks SET park_generation = park_generation + 1 WHERE id = ?", (task_id,))
    board.commit()

    tick(board)

    assert _row(board, task_id)["held_reason"] == "park_generation_changed"
    assert kb.get_task(board, task_id).status == "scheduled"


def test_unknown_block_identity_holds(board):
    task_id = park(board)
    board.execute("UPDATE tasks SET block_identity_state = 'unknown' WHERE id = ?", (task_id,))
    board.commit()

    tick(board)

    assert _row(board, task_id)["held_reason"] == "block_identity_unknown"


def test_classification_rerun_inside_transaction(board, monkeypatch):
    task_id = park(board)
    original = wakes._due_pass

    def hold_after_read(conn, report, **kw):
        # The tick has read its candidate rows; a hold lands before the kernel op runs.
        rows = conn.execute("SELECT * FROM task_wakes WHERE state = 'armed'").fetchall()
        assert rows
        conn.execute("UPDATE tasks SET claim_lock = 'late', claim_expires = ? WHERE id = ?", (NOW + 600, task_id))
        conn.commit()
        return original(conn, report, **kw)

    monkeypatch.setattr(wakes, "_due_pass", hold_after_read)

    tick(board)

    assert _row(board, task_id)["held_reason"] == "live_claim"


def test_sweep_closes_superseded_due_held_rows_and_is_bounded(board):
    ids = [park(board, f"c{i}", wake_at=DUE - i) for i in range(3)]
    tick(board)
    for task_id in ids:
        board.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    board.commit()
    fresh = park(board, "fresh")

    bounded = tick(board, sweep_limit=1)
    assert bounded.wakes["sweep_truncated"] == 3  # 3 swept candidates + the fresh row, 1 examined
    assert _row(board, fresh)["state"] == "due_held"
    tick(board, sweep_limit=10)

    assert {_row(board, t)["state"] for t in ids} == {"superseded"}


def test_held_reason_written_once(board):
    task_id = park(board)
    board.execute("UPDATE tasks SET claim_lock = 'x', claim_expires = ? WHERE id = ?", (NOW + 600, task_id))
    board.commit()
    tick(board)
    board.execute("UPDATE tasks SET claim_lock = NULL, claim_expires = NULL WHERE id = ?", (task_id,))
    board.commit()

    tick(board)
    tick(board)

    assert _row(board, task_id)["held_reason"] == "live_claim"
    assert len(events(board, task_id, {"wake_due_held"})) == 1
