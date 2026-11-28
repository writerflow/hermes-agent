"""Typed timed wakes for Scheduled Kanban cards (WRI-2639, Part A).

A card parked with a typed ``wake_at`` gets one ``task_wakes`` row in the same
transaction as the park. Each dispatcher tick runs :func:`reconcile_timed_wakes`
(through ``reconcile_typed_waits``): a due wake is detected and reported
exactly once, and the card **stays Scheduled**. Part A releases no card; the
guarded release is Part B and needs the WRI-2631 contract. Legacy
``SCHEDULED_UNTIL=`` text in a reason is never interpreted, only reported.

Kernel requests and handlers live here and are registered into
``kanban_db_task_state``'s request tables, like the other typed operations.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timedelta, timezone, UTC
from typing import Any, Optional

_log = logging.getLogger(__name__)

WAKE_INPUT_MAX = 64
WAKE_HORIZON_SECONDS = 366 * 86400
RECONCILER_ACTOR = "timed-wake-reconciler"
DISPATCHER_SCOPE = "dispatcher"
LEGACY_MARKER = "SCHEDULED_UNTIL="
LIVE_STATES = ("armed", "due_held", "release_unknown")

_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS task_wakes (
        wake_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        park_event_id INTEGER NOT NULL,
        park_generation INTEGER NOT NULL,
        wake_at INTEGER NOT NULL,
        wake_input TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN
            ('armed','due_held','release_unknown','released','superseded')),
        held_reason TEXT, resolution TEXT,
        armed_by TEXT NOT NULL, armed_at INTEGER NOT NULL,
        arm_operation_id TEXT NOT NULL UNIQUE,
        due_event_id INTEGER, release_operation_id TEXT UNIQUE,
        release_event_id INTEGER, resolved_at INTEGER)""",
    "CREATE UNIQUE INDEX IF NOT EXISTS one_live_wake_per_task ON task_wakes(task_id) "
    "WHERE state IN ('armed','due_held','release_unknown')",
    "CREATE INDEX IF NOT EXISTS task_wakes_scan ON task_wakes(state, wake_at)",
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Additive DDL, safe to repeat; older binaries ignore the table."""
    for statement in _STATEMENTS:
        conn.execute(statement)


# --------------------------------------------------------------- parsing

class WakeInputError(ValueError):
    """A refused ``wake_at`` value; ``code`` is stable and the board is untouched."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class WakeTime:
    epoch: int
    text: str


_ISO = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?(Z|[+-]\d{2}:\d{2})?$"
)


def parse_wake_at(text: object, *, now: int, check_range: bool = True) -> WakeTime:
    """The one parser: ``YYYY-MM-DDTHH:MM[:SS]`` plus ``Z`` or ``±HH:MM``, nothing else."""
    if not isinstance(text, str):
        raise WakeInputError("wake_at_not_string")
    if not text.strip():
        raise WakeInputError("wake_at_empty")
    match = _ISO.match(text)
    if match is None or len(text) > WAKE_INPUT_MAX:
        raise WakeInputError("wake_at_format")
    zone = match.group(7)
    if zone is None:
        raise WakeInputError("wake_at_ambiguous_timezone")
    tzinfo = UTC
    if zone != "Z":
        hours, minutes = int(zone[1:3]), int(zone[4:6])
        if minutes > 59 or hours > 14 or (hours == 14 and minutes):
            raise WakeInputError("wake_at_bad_offset")
        tzinfo = timezone(timedelta(hours=hours, minutes=minutes) * (-1 if zone[0] == "-" else 1))
    year, month, day, hour, minute = (int(match.group(i)) for i in range(1, 6))
    try:
        moment = datetime(year, month, day, hour, minute, int(match.group(6) or 0), tzinfo=tzinfo)
    except ValueError:
        raise WakeInputError("wake_at_invalid_datetime") from None
    epoch = int(moment.timestamp())
    if check_range and epoch <= now:
        raise WakeInputError("wake_at_in_past")
    if check_range and epoch > now + WAKE_HORIZON_SECONDS:
        raise WakeInputError("wake_at_too_far")
    return WakeTime(epoch, text)


def operator_context() -> bool:
    """One predicate for arming at every entry point.

    ``HERMES_KANBAN_TASK`` unset (a value a cron job inherited from a worker
    counts as set) and not a ``delegate_task`` child or its descendant.
    """
    if os.environ.get("HERMES_KANBAN_TASK"):
        return False
    try:
        from agent import delegation_context
        return not delegation_context.is_delegated_child_process_context()
    except (ImportError, AttributeError):  # fail closed: an unreadable context is not an operator
        return False


def may_arm(task_id: str) -> bool:
    """A dispatcher-owned worker may arm its own card; any other card needs an operator context."""
    try:
        from agent.delegation_context import owned_kanban_task
        if task_id and owned_kanban_task() == task_id:
            return True
    except (ImportError, AttributeError):  # fail closed: an unreadable context may not arm
        return False
    return operator_context()


# ------------------------------------------------------- kernel requests

@dataclass(frozen=True)
class ArmWakeRequest:
    task_id: str
    actor_scope: str
    operation_id: str
    wake_at: int
    wake_input: str
    expected_park_event_id: Optional[int] = None
    operation_kind: str = "arm_wake"
    authority_kind: str = "wake"


@dataclass(frozen=True)
class WakeDueRequest:
    task_id: str
    actor_scope: str
    operation_id: str
    wake_id: int
    now: int
    operation_kind: str = "wake_due"
    authority_kind: str = "wake"


@dataclass(frozen=True)
class WakeSupersedeRequest:
    task_id: str
    actor_scope: str
    operation_id: str
    wake_id: int
    operation_kind: str = "wake_supersede"
    authority_kind: str = "wake"


@dataclass(frozen=True)
class WakeInvalidRequest:
    task_id: str
    actor_scope: str
    operation_id: str
    wake_id: int
    code: str
    operation_kind: str = "wake_invalid"
    authority_kind: str = "wake"


def _request_hash(request) -> str:
    from hermes_cli.kanban_db_task_state import _canonical_hash

    # ``now`` is a clock reading, not request identity: a replay on a later tick is the same request.
    return _canonical_hash({
        item.name: getattr(request, item.name)
        for item in fields(request)
        if item.name not in {"authority_kind", "now"}
    })


REQUEST_HASHERS = {
    ArmWakeRequest: _request_hash,
    WakeDueRequest: _request_hash,
    WakeSupersedeRequest: _request_hash,
    WakeInvalidRequest: _request_hash,
}


def scheduled_payload(request) -> dict:
    """Payload of the ``scheduled`` event: unchanged unless a wake is armed."""
    payload: dict[str, Any] = {"reason": request.reason}
    if request.wake_at is not None:
        payload["wake_at"] = request.wake_at
    return payload


def schedule_hash_fields(request) -> dict:
    if request.wake_at is None:
        return {}
    return {"wake_at": request.wake_at, "wake_input": request.wake_input}


# ----------------------------------------------------------- the arming

def _refuse(request, code: str):
    from hermes_cli.kanban_db_task_state import TransitionOutcome, _AppliedTransition

    return _AppliedTransition(TransitionOutcome(False, request.operation_id, refusal=code))


def _last_row_id(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def arm_in_transaction(
    private: sqlite3.Connection,
    task_id: str,
    wake: WakeTime,
    *,
    actor_scope: str,
    operation_id: str,
    park_event_id: int,
) -> int:
    """Insert the live row for this park, closing any older live row for the task.

    ``park_generation`` is read after the park's last ``tasks`` update, so it is
    the value a later reader sees.
    """
    now = int(time.time())
    private.execute(
        "UPDATE task_wakes SET state = 'superseded', resolution = 'rearmed', resolved_at = ? "
        "WHERE task_id = ? AND state IN ('armed', 'due_held')",
        (now, task_id),
    )
    generation = private.execute(
        "SELECT park_generation FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()[0]
    private.execute(
        "INSERT INTO task_wakes (task_id, park_event_id, park_generation, wake_at, wake_input, "
        "state, armed_by, armed_at, arm_operation_id) VALUES (?, ?, ?, ?, ?, 'armed', ?, ?, ?)",
        (task_id, park_event_id, generation, wake.epoch, wake.text, actor_scope, now, operation_id),
    )
    return _last_row_id(private)


def _check_wake_value(request) -> Optional[str]:
    value, text = request.wake_at, request.wake_input
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return "wake_at_invalid"
    if not isinstance(text, str) or not text or len(text) > WAKE_INPUT_MAX:
        return "wake_input_invalid"
    return None


def apply_schedule(private: sqlite3.Connection, request):
    """``_apply_schedule`` plus the typed row, in the one park transaction."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_task_state import _apply_schedule

    if request.wake_at is None:
        return _apply_schedule(private, request)
    code = _check_wake_value(request)
    status = private.execute(
        "SELECT status FROM tasks WHERE id = ?", (request.task_id,)
    ).fetchone()
    if code is None and status is not None and status["status"] == "blocked":
        # A timer must not stand in for an outstanding needs_input/capability/creator hold.
        if kb._has_sticky_block(private, request.task_id):
            code = "wake_hold_outstanding"
    if code:
        return _refuse(request, code)
    applied = _apply_schedule(private, request)
    if applied.outcome.committed:
        arm_in_transaction(
            private, request.task_id, WakeTime(request.wake_at, request.wake_input),
            actor_scope=request.actor_scope, operation_id=request.operation_id,
            park_event_id=applied.outcome.event_id,
        )
    return applied


def apply_arm_wake(private: sqlite3.Connection, request: ArmWakeRequest):
    """Arm an already-parked card (operator command)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_task_state import TransitionOutcome, _AppliedTransition

    code = _check_wake_value(request)
    row = private.execute(
        "SELECT status, block_identity_state FROM tasks WHERE id = ?", (request.task_id,)
    ).fetchone()
    park = None
    if code is None and row is None:
        code = "task_not_found"
    elif code is None and row["block_identity_state"] == "unknown":
        code = "block_identity_unknown"
    elif code is None:
        park = private.execute(
            "SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind = 'scheduled'",
            (request.task_id,),
        ).fetchone()[0]
        if row["status"] != "scheduled" or park is None:
            code = "not_parked"
        elif request.expected_park_event_id not in (None, park):
            code = "park_changed"
        elif kb._has_sticky_block(private, request.task_id):
            code = "wake_hold_outstanding"
    if code:
        return _refuse(request, code)
    kb._append_event(private, request.task_id, "wake_armed", {
        "wake_at": request.wake_at, "park_event_id": park, "actor": request.actor_scope,
    })
    event_id = _last_row_id(private)
    arm_in_transaction(
        private, request.task_id, WakeTime(request.wake_at, request.wake_input),
        actor_scope=request.actor_scope, operation_id=request.operation_id, park_event_id=park,
    )
    return _AppliedTransition(TransitionOutcome(
        True, request.operation_id, event_id, row["block_identity_state"]
    ))


# --------------------------------------------------- hold classification

def classify(private: sqlite3.Connection, wake: sqlite3.Row, *, superseded_only: bool = False):
    """``("superseded"|"held", code)`` for a due row, ``None`` when nothing holds it.

    Runs on the connection that is about to write, so a hold that landed after
    the tick read is honoured. Order is the precedence table in the plan.
    """
    from hermes_cli import kanban_db as kb

    task = private.execute(
        "SELECT status, claim_lock, current_run_id, block_identity_state, park_generation "
        "FROM tasks WHERE id = ?", (wake["task_id"],)
    ).fetchone()
    if task is None:
        return "superseded", "task_missing"
    if task["status"] != "scheduled":
        return "superseded", "park_exited"
    newest = private.execute(
        "SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind = 'scheduled'",
        (wake["task_id"],),
    ).fetchone()[0]
    if newest != wake["park_event_id"]:
        return "superseded", "repark"
    if superseded_only:
        return None
    if kb._has_sticky_block(private, wake["task_id"]):
        return "held", "sticky_block"
    if task["claim_lock"] is not None or task["current_run_id"] is not None:
        return "held", "live_claim"
    if task["block_identity_state"] == "unknown":
        return "held", "block_identity_unknown"
    if task["park_generation"] != wake["park_generation"]:
        return "held", "park_generation_changed"
    return None


def _load_wake(private: sqlite3.Connection, request) -> Optional[sqlite3.Row]:
    wake = private.execute(
        "SELECT * FROM task_wakes WHERE wake_id = ?", (request.wake_id,)
    ).fetchone()
    return wake if wake is not None and wake["task_id"] == request.task_id else None


def _close_superseded(private: sqlite3.Connection, request, wake, code: str):
    """CAS ``armed|due_held -> superseded``; one event unless the card is gone."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_task_state import TransitionOutcome, _AppliedTransition

    updated = private.execute(
        "UPDATE task_wakes SET state = 'superseded', resolution = ?, resolved_at = ? "
        "WHERE wake_id = ? AND state IN ('armed', 'due_held')",
        (code, int(time.time()), request.wake_id),
    )
    if updated.rowcount != 1:
        return _refuse(request, "compare_and_set")
    event_id = None
    identity = "none"
    if code != "task_missing":
        kb._append_event(private, request.task_id, "wake_superseded", {
            "wake_id": request.wake_id, "reason_code": code, "actor": RECONCILER_ACTOR,
        })
        event_id = _last_row_id(private)
        identity = private.execute(
            "SELECT block_identity_state FROM tasks WHERE id = ?", (request.task_id,)
        ).fetchone()[0]
    return _AppliedTransition(TransitionOutcome(True, request.operation_id, event_id, identity))


def apply_wake_due(private: sqlite3.Connection, request: WakeDueRequest):
    """Part A terminal action: ``armed -> due_held`` and one event; no card column but
    ``action_generation`` changes (the event trigger bumps it)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_task_state import TransitionOutcome, _AppliedTransition

    wake = _load_wake(private, request)
    if wake is None:
        return _refuse(request, "wake_not_found")
    verdict = classify(private, wake)
    if verdict is not None and verdict[0] == "superseded":
        return _close_superseded(private, request, wake, verdict[1])
    held_reason = verdict[1] if verdict else "release_guard_unavailable"
    updated = private.execute(
        "UPDATE task_wakes SET state = 'due_held', held_reason = ? "
        "WHERE wake_id = ? AND state = 'armed'",
        (held_reason, request.wake_id),
    )
    if updated.rowcount != 1:
        return _refuse(request, "compare_and_set")
    kb._append_event(private, request.task_id, "wake_due_held", {
        "wake_id": request.wake_id, "wake_at": wake["wake_at"],
        "late_by_seconds": max(0, request.now - wake["wake_at"]),
        "held_reason": held_reason, "actor": RECONCILER_ACTOR,
    })
    event_id = _last_row_id(private)
    private.execute(
        "UPDATE task_wakes SET due_event_id = ? WHERE wake_id = ?", (event_id, request.wake_id)
    )
    identity = private.execute(
        "SELECT block_identity_state FROM tasks WHERE id = ?", (request.task_id,)
    ).fetchone()[0]
    return _AppliedTransition(TransitionOutcome(True, request.operation_id, event_id, identity))


def apply_wake_supersede(private: sqlite3.Connection, request: WakeSupersedeRequest):
    """Sweep of an already-held row: only the superseded half of the classification."""
    wake = _load_wake(private, request)
    if wake is None:
        return _refuse(request, "wake_not_found")
    verdict = classify(private, wake, superseded_only=True)
    if verdict is None:
        return _refuse(request, "not_superseded")
    return _close_superseded(private, request, wake, verdict[1])


def apply_wake_invalid(private: sqlite3.Connection, request: WakeInvalidRequest):
    """Report a stored row that is not a valid wake; the card and the row stay as they are."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_task_state import TransitionOutcome, _AppliedTransition

    task = private.execute(
        "SELECT block_identity_state FROM tasks WHERE id = ?", (request.task_id,)
    ).fetchone()
    if task is None:
        return _refuse(request, "task_not_found")
    kb._append_event(private, request.task_id, "wake_invalid", {
        "wake_id": request.wake_id, "code": request.code, "actor": RECONCILER_ACTOR,
    })
    return _AppliedTransition(TransitionOutcome(
        True, request.operation_id, _last_row_id(private), task["block_identity_state"]
    ))


HANDLERS = {
    ArmWakeRequest: apply_arm_wake,
    WakeDueRequest: apply_wake_due,
    WakeSupersedeRequest: apply_wake_supersede,
    WakeInvalidRequest: apply_wake_invalid,
}


def arm_wake_transition(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    wake_at: int,
    wake_input: str,
    actor_scope: str,
    operation_id: str,
    expected_park_event_id: Optional[int] = None,
):
    """Arm a typed wake on an already-Scheduled card through the receipt kernel."""
    from hermes_cli.kanban_db_task_state import _run_operation

    request = ArmWakeRequest(
        task_id, actor_scope, operation_id, wake_at, wake_input, expected_park_event_id
    )
    return _run_operation(conn, request).outcome


# ------------------------------------------------------------- reconcile

def _table_exists(conn: sqlite3.Connection) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'task_wakes'"
    ).fetchone() is not None


def _bucket(report: dict, key: str) -> list:
    return report.setdefault(key, [])


def _run(conn: sqlite3.Connection, request) -> Any:
    from hermes_cli.kanban_db_task_state import _run_operation

    return _run_operation(conn, request).outcome


def _due_pass(conn, report, *, now, dry_run, due_limit, result) -> None:
    where = "state = 'armed' AND typeof(wake_at) = 'integer' AND wake_at <= ?"
    total = conn.execute(f"SELECT COUNT(*) FROM task_wakes WHERE {where}", (now,)).fetchone()[0]
    rows = conn.execute(
        f"SELECT wake_id, task_id FROM task_wakes WHERE {where} "
        "ORDER BY wake_at, wake_id LIMIT ?", (now, due_limit),
    ).fetchall()
    report["deferred"] = max(0, total - len(rows))
    if dry_run:
        report["would_due_held"] = [row["wake_id"] for row in rows]
        return
    for row in rows:
        wake_id = row["wake_id"]
        try:
            outcome = _run(conn, WakeDueRequest(
                row["task_id"], DISPATCHER_SCOPE, f"wake-due:{wake_id}", wake_id, now
            ))
            if outcome.committed:
                state = conn.execute(
                    "SELECT state FROM task_wakes WHERE wake_id = ?", (wake_id,)
                ).fetchone()
                _bucket(report, "due_held" if state and state[0] == "due_held" else "superseded"
                        ).append(wake_id)
        except Exception:  # health: allow BLE001 -- one bad wake row never blocks the others; logged with traceback
            result.wait_reconcile_errors += 1
            _log.warning("timed wake %s: due action failed; it stays armed", wake_id, exc_info=True)


def _invalid_pass(conn, report, *, dry_run, due_limit, result) -> None:
    rows = conn.execute(
        "SELECT wake_id, task_id FROM task_wakes WHERE state IN ('armed', 'due_held') "
        "AND (typeof(wake_at) != 'integer' OR wake_at <= 0) ORDER BY wake_id LIMIT ?",
        (due_limit,),
    ).fetchall()
    for row in rows:
        _bucket(report, "invalid").append(row["wake_id"])
        if dry_run:
            continue
        try:
            _run(conn, WakeInvalidRequest(
                row["task_id"], DISPATCHER_SCOPE, f"wake-invalid:{row['wake_id']}:bad_wake_at",
                row["wake_id"], "bad_wake_at",
            ))
        except Exception:  # health: allow BLE001 -- one bad wake row never blocks the others; logged with traceback
            result.wait_reconcile_errors += 1
            _log.warning("timed wake %s: invalid-row report failed", row["wake_id"], exc_info=True)


def _sweep(conn, report, *, sweep_limit, result) -> None:
    held = "state IN ('due_held', 'release_unknown')"
    total = conn.execute(f"SELECT COUNT(*) FROM task_wakes WHERE {held}").fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM task_wakes WHERE {held} ORDER BY wake_id LIMIT ?", (sweep_limit,)
    ).fetchall()
    report["sweep_truncated"] = max(0, total - len(rows))
    for row in rows:
        if classify(conn, row, superseded_only=True) is None:
            continue
        try:
            wake_id = row["wake_id"]
            outcome = _run(conn, WakeSupersedeRequest(
                row["task_id"], DISPATCHER_SCOPE, f"wake-supersede:{wake_id}", wake_id
            ))
            if outcome.committed:
                _bucket(report, "superseded").append(wake_id)
        except Exception:  # health: allow BLE001 -- one bad wake row never blocks the others; logged with traceback
            result.wait_reconcile_errors += 1
            _log.warning("timed wake %s: supersede failed", row["wake_id"], exc_info=True)


def reconcile_timed_wakes(
    conn: sqlite3.Connection,
    result,
    *,
    now: Optional[int] = None,
    dry_run: bool = False,
    sweep_limit: int = 500,
    due_limit: int = 50,
) -> dict:
    """Detect due wakes and fail closed; the report is also stored on ``result.wakes``.

    ``now`` is injected by tests. Exactly-once rests on the database: the
    ``armed`` compare-and-set and the kernel's unique operation id.
    """
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_control import read_control

    report: dict[str, Any] = result.wakes
    if not _table_exists(conn):
        return report
    now = int(time.time()) if now is None else int(now)
    live = conn.execute(
        "SELECT COUNT(*) FROM task_wakes WHERE state IN ('armed', 'due_held', 'release_unknown')"
    ).fetchone()[0]
    if not kbc._main_db_file(conn):
        # The protected kernel needs a file-backed board: nothing is claimed here.
        report["skipped_non_file_board"] = live
        return report
    if read_control(conn).state != "running":
        report["control_held"] = live
        return report
    if not live:
        return report
    if conn.in_transaction:
        conn.commit()
    _due_pass(conn, report, now=now, dry_run=dry_run, due_limit=due_limit, result=result)
    _invalid_pass(conn, report, dry_run=dry_run, due_limit=due_limit, result=result)
    if not dry_run:
        _sweep(conn, report, sweep_limit=sweep_limit, result=result)
    return report


# --------------------------------------------------------------- reports

@dataclass
class WakeResultFields:
    """Timed-wake report carried on ``DispatchResult``; ``due_held`` ids mean "due, release held"."""

    wakes: dict = field(default_factory=dict)
    wait_reconcile_errors: int = 0


def wake_suppression_counts(result) -> dict:
    """``{"wake_due_held": N}`` for ``describe_suppression`` when a tick held due wakes."""
    held = result.wakes.get("due_held")
    return {"wake_due_held": len(held)} if held else {}


def reconcile_typed_waits(conn: sqlite3.Connection, result, *, dry_run: bool = False) -> None:
    """Dispatcher-tick entry point for typed waits; a failure never stops the tick."""
    try:
        reconcile_timed_wakes(conn, result, dry_run=dry_run)
    except Exception:  # health: allow BLE001 -- typed waits never stop the dispatcher tick; logged with traceback
        result.wait_reconcile_errors += 1
        _log.warning("typed wait reconcile failed; the tick continues", exc_info=True)


def list_wakes(conn: sqlite3.Connection, states: Optional[list[str]] = None) -> list[dict]:
    if not _table_exists(conn):
        return []
    if not states:
        return [dict(row) for row in conn.execute("SELECT * FROM task_wakes ORDER BY wake_id")]
    rows = conn.execute("SELECT * FROM task_wakes ORDER BY wake_id").fetchall()
    return [dict(row) for row in rows if row["state"] in states]


def _display_parse(reason: str) -> str:
    tokens = re.findall(re.escape(LEGACY_MARKER) + r"(\S+)", reason)
    if len(tokens) != 1:
        return f"ambiguous: {len(tokens)} markers"
    try:
        wake = parse_wake_at(tokens[0], now=0, check_range=False)
    except WakeInputError as exc:
        return f"malformed: {exc.code}"
    return "valid " + datetime.fromtimestamp(wake.epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def legacy_marker_report(conn: sqlite3.Connection) -> list[dict]:
    """Read-only: Scheduled cards whose *current* park reason has a legacy marker and no live row."""
    out = []
    for task in conn.execute("SELECT id FROM tasks WHERE status = 'scheduled' ORDER BY id"):
        task_id = task["id"]
        park = conn.execute(
            "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = 'scheduled' "
            "ORDER BY id DESC LIMIT 1", (task_id,),
        ).fetchone()
        if park is None:
            continue
        try:
            reason = (json.loads(park["payload"] or "{}") or {}).get("reason") or ""
        except (TypeError, ValueError):
            continue
        if LEGACY_MARKER not in str(reason):
            continue
        if _table_exists(conn) and conn.execute(
            "SELECT 1 FROM task_wakes WHERE task_id = ? AND state IN ('armed', 'due_held', "
            "'release_unknown')", (task_id,),
        ).fetchone():
            continue
        out.append({
            "task_id": task_id, "park_event_id": park["id"], "status": "NOT ARMED",
            "display_parse": _display_parse(str(reason)),
        })
    return out
