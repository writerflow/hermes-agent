"""WRI-2639 Part A T8: the storage boundary and board transfer for ``task_wakes``."""

from __future__ import annotations

import sqlite3
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_transfer as kt
from tests.hermes_cli._timed_wake_support import FUTURE, park, wake_rows
from tests.hermes_cli import _timed_wake_support as _support

board = _support.board  # re-exported pytest fixture


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO task_wakes (task_id, park_event_id, park_generation, wake_at, wake_input, state, "
        "armed_by, armed_at, arm_operation_id) VALUES ('t', 1, 0, 1, 'x', 'armed', 'a', 1, 'op')",
        "UPDATE task_wakes SET state = 'due_held'",
        "DELETE FROM task_wakes",
    ],
)
def test_task_wakes_written_only_by_protected_operations(board, statement):
    park(board, wake_at=FUTURE)
    path = kb.kanban_db_path()

    with kbc._ordinary_connection(path) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            conn.execute(statement)
    assert len(wake_rows(board)) == 1 and wake_rows(board)[0]["state"] == "armed"


def test_board_transfer_round_trip_keeps_armed_wakes(tmp_path, monkeypatch):
    # Transfer snapshots the whole database file, so wake rows travel with their cards. The
    # restored copy re-evaluates its own due rows; nothing is dropped silently.
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT",
                "HERMES_KANBAN_ATTACHMENTS_ROOT", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)

    def use(name):
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))
        kb._INITIALIZED_PATHS.clear()

    use("source")
    kb.create_board("alpha", name="Alpha")
    with kbc.connect_closing(board="alpha") as conn:
        task_id = park(conn, wake_at=int(time.time()) + 3600)
    archive = kt.export_board("alpha", str(tmp_path / "alpha"))["archive"]

    use("target")
    result = kt.import_board(archive)

    with kbc.connect_closing(board=result["board"]) as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM task_wakes")]
    assert [(r["task_id"], r["state"]) for r in rows] == [(task_id, "armed")]
