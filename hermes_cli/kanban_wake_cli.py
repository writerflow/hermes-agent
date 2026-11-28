"""Operator CLI for typed timed wakes: ``kanban wake`` (arm) and ``kanban wakes`` (report).

Part A only detects a due wake and holds; nothing here releases a card.
"""

from __future__ import annotations

import argparse
import secrets
import time
from datetime import datetime, timezone, UTC

from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_wakes as wakes
from hermes_cli.kanban_output import _err, _json_out


def _actor() -> str:
    from hermes_cli.profiles import current_profile_name
    return current_profile_name("user") or "user"


def _cmd_wake(args: argparse.Namespace) -> int:
    """Arm a typed wake on an already-Scheduled card (operator context only)."""
    if not wakes.operator_context():
        return _err(
            "kanban wake is operator-only: it is refused when HERMES_KANBAN_TASK is set "
            "(including an inherited value) or in a delegate_task child"
        )
    try:
        wake = wakes.parse_wake_at(args.at, now=int(time.time()))
    except wakes.WakeInputError as exc:
        return _err(f"wake_at refused: {exc.code}")
    with kbc.connect_closing() as conn:
        outcome = wakes.arm_wake_transition(
            conn, args.task_id, wake_at=wake.epoch, wake_input=wake.text,
            actor_scope=_actor(), operation_id=f"cli-wake:{args.task_id}:{secrets.token_hex(12)}",
            expected_park_event_id=args.expect_park,
        )
    if not outcome.committed:
        return _err(f"cannot arm a wake on {args.task_id}: {outcome.refusal}")
    print(f"Armed wake on {args.task_id} for {wake.text} (a due wake is held, not released)")
    return 0


def _format_row(row: dict) -> str:
    at = datetime.fromtimestamp(row["wake_at"], UTC).strftime("%Y-%m-%dT%H:%M:%SZ") \
        if isinstance(row["wake_at"], int) else f"invalid({row['wake_at']!r})"
    note = f" [{row['held_reason']}]" if row["held_reason"] else ""
    if row["state"] == "due_held":
        note = f" due, release held{note}"
    return f"  #{row['wake_id']} {row['task_id']} {row['state']} at {at}{note}"


def _cmd_wakes(args: argparse.Namespace) -> int:
    """Read-only report of wake rows, or of legacy SCHEDULED_UNTIL= reasons (``--legacy``)."""
    with kbc.connect_closing() as conn:
        if args.legacy:
            legacy = wakes.legacy_marker_report(conn)
            if _json_out(args, legacy):
                return 0
            for item in legacy:
                print(f"  {item['task_id']} NOT ARMED (park event {item['park_event_id']}): "
                      f"{item['display_parse']}")
            if not legacy:
                print("No Scheduled card carries a legacy SCHEDULED_UNTIL= marker without a wake.")
            return 0
        rows = wakes.list_wakes(conn, [args.state] if args.state else None)
    if _json_out(args, rows):
        return 0
    for row in rows:
        print(_format_row(row))
    if not rows:
        print("No wakes.")
    return 0
