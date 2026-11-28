"""WRI-2639 Part A T5: exactly-once catch-up (several due, restart, two connections, repeats)."""

from __future__ import annotations

import threading

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_task_state as task_state
from hermes_cli import kanban_db_wakes as wakes
from tests.hermes_cli._timed_wake_support import (
    DUE, NOW, events, park, tick, wake_rows,
)
from tests.hermes_cli import _timed_wake_support as _support

board = _support.board  # re-exported pytest fixture

# The terminal action of Part A is ``due_held``; Part B ("released") is held on the WRI-2631 contract.
TERMINAL = "due_held"


def _due_events(conn):
    return conn.execute("SELECT COUNT(*) FROM task_events WHERE kind = 'wake_due_held'").fetchone()[0]


def test_sixty_due_wakes_drain_fifty_then_ten_then_none(board):
    for i in range(60):
        park(board, f"c{i}", wake_at=DUE - i)

    first = tick(board)
    assert (len(first.wakes["due_held"]), first.wakes["deferred"]) == (50, 10)
    second = tick(board)
    assert (len(second.wakes["due_held"]), second.wakes.get("deferred")) == (10, 0)
    third = tick(board)
    assert not third.wakes.get("due_held")
    assert _due_events(board) == 60
    assert {r["state"] for r in wake_rows(board)} == {TERMINAL}


def test_new_connection_after_downtime_catches_up_exactly_once(board):
    task_id = park(board, wake_at=NOW - 5000)
    board.close()
    kb._INITIALIZED_PATHS.clear()

    with kbc.connect_closing() as fresh:
        tick(fresh)
        tick(fresh)
        (event,) = events(fresh, task_id, {"wake_due_held"})
    assert event[2]["late_by_seconds"] == 5000


def test_commit_failure_mid_transition_rolls_back_then_one_event(board):
    task_id = park(board)
    board.execute(
        "CREATE TRIGGER refuse_receipt BEFORE INSERT ON task_transition_receipts "
        "WHEN NEW.operation_kind = 'wake_due' BEGIN SELECT RAISE(ABORT, 'forced'); END"
    )
    board.commit()

    failed = tick(board)

    assert failed.wait_reconcile_errors == 1
    assert wake_rows(board)[0]["state"] == "armed"
    assert events(board, task_id, {"wake_due_held"}) == []

    board.execute("DROP TRIGGER refuse_receipt")
    board.commit()
    tick(board)
    assert len(events(board, task_id, {"wake_due_held"})) == 1


def test_lost_ack_replay_returns_stored_outcome_without_second_event(board):
    task_id = park(board)
    wake_id = wake_rows(board)[0]["wake_id"]
    request = wakes.WakeDueRequest(task_id, "dispatcher", f"wake-due:{wake_id}", wake_id, NOW)
    first = task_state._run_operation(board, request).outcome
    # The caller never saw the result; a later tick replays the same operation id.
    again = task_state._run_operation(
        board, wakes.WakeDueRequest(task_id, "dispatcher", f"wake-due:{wake_id}", wake_id, NOW + 60)
    ).outcome

    assert first.committed and again.committed and again.event_id == first.event_id
    assert len(events(board, task_id, {"wake_due_held"})) == 1


def test_tick_replays_the_same_operation_id_after_a_lost_ack(board):
    # The receipt, not only the CAS, makes a repeated tick a replay: put the row back to
    # 'armed' as if the first commit's result was never observed, then tick again.
    task_id = park(board)
    tick(board)
    board.execute("UPDATE task_wakes SET state = 'armed'")
    board.commit()

    tick(board, now=NOW + 60)

    assert len(events(board, task_id, {"wake_due_held"})) == 1


def test_second_request_for_same_wake_under_a_different_id_is_refused_by_the_cas(board):
    # RA1: the compare-and-set alone must stop a duplicate that carries a fresh operation id.
    task_id = park(board)
    wake_id = wake_rows(board)[0]["wake_id"]
    mk = lambda op: wakes.WakeDueRequest(task_id, "dispatcher", op, wake_id, NOW)

    assert task_state._run_operation(board, mk("wake-due:a")).outcome.committed
    second = task_state._run_operation(board, mk("wake-due:b")).outcome

    assert (second.committed, second.refusal) == (False, "compare_and_set")
    assert len(events(board, task_id, {"wake_due_held"})) == 1


def test_two_connections_race_produce_one_transition(board):
    task_id = park(board)
    barrier = threading.Barrier(2)
    errors = []

    def run():
        try:
            with kbc.connect_closing() as conn:
                barrier.wait(timeout=10)
                tick(conn)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors
    assert len(events(board, task_id, {"wake_due_held"})) == 1
    assert [r["state"] for r in wake_rows(board)] == [TERMINAL]


def test_repeated_ticks_are_idempotent(board):
    task_id = park(board)
    tick(board)
    (first_row,) = wake_rows(board)
    first_events = events(board, task_id)

    tick(board)
    tick(board)
    tick(board, now=NOW + 86400)

    (row,) = wake_rows(board)
    assert (row["held_reason"], row["due_event_id"]) == (first_row["held_reason"], first_row["due_event_id"])
    assert events(board, task_id) == first_events


def test_two_due_wakes_on_different_tasks_each_get_one_event(board):
    a, b = park(board, "a"), park(board, "b")

    tick(board)
    tick(board)

    assert len(events(board, a, {"wake_due_held"})) == 1
    assert len(events(board, b, {"wake_due_held"})) == 1


@pytest.mark.skip(reason="Part B held: WRI-2631 contract")
def test_terminal_released_variant_is_held():
    raise AssertionError("un-skipped by B1")
