"""Write-ahead records for held work-scope placement."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from hermes_constants import get_routing_process_hermes_home
from utils.files import atomic_write_text


def _live_child_birth(pid: int) -> int:
    proc = Path(f"/proc/{pid}")
    stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    # The private gate's peer must still be our child, not a reused PID.
    if int(stat[1]) != os.getpid() or stat[0] in {"Z", "X"}:
        raise OSError("scope launcher is not a live child")
    return int(stat[19])


def verify_background_birth(pid: int, birth: int) -> None:
    """Check the held child again after the launcher reports placement."""
    if _live_child_birth(pid) != birth:
        raise OSError("scope launcher birth changed before release")


def record_background_intent(pid: int, unit: str, session_id: str) -> int:
    """Persist the held child's origin and return its recorded birth."""
    proc = Path(f"/proc/{pid}")
    birth = _live_child_birth(pid)
    origin, = (line[3:] for line in (proc / "cgroup").read_text().splitlines() if line.startswith("0::/"))
    attempt = uuid.uuid4().hex
    data = {
        "version": 1,
        "attempt_id": attempt,
        "state": "unresolved",
        "role": "background_tool",
        "session_id": session_id,
        "unit": unit,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "pid_namespace": os.readlink(proc / "ns/pid"),
        "uid": proc.stat().st_uid,
        "pid": pid,
        "start_time": birth,
        "origin_cgroup": origin,
    }
    home = get_routing_process_hermes_home()
    directory = home / "work-scope-placements"
    atomic_write_text(directory / f"{attempt}.json", json.dumps(data), mode=0o600)
    # The normal utility's directory fsync is best effort. Here a failed
    # fsync must keep the request gate closed, including on the first mkdir.
    for parent in (directory, home):
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return birth
