"""WRI-2592 native acceptance; never run this workload on a serving host.

The VM operator opts in with a root-owned /run/hermes-foreground-scope-test
containing '<boot-id> <test-uid>' for this disposable boot and unprivileged
account. The marker deliberately does not survive a reboot or the VM's teardown.
"""

from contextlib import contextmanager
import json
import os
import select
import shlex
import signal
import stat
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from tests.waiting import wait_until
from tools import foreground_scope, process_registry
from tools.environments import base
from tools.terminal_tool import terminal_tool
from tools.terminal_tool_lifecycle import cleanup_vm

pytestmark = pytest.mark.platforms("linux")


@pytest.fixture
def disposable_vm(tmp_path, monkeypatch):
    marker = Path("/run/hermes-foreground-scope-test")
    try:
        details = marker.lstat()
    except FileNotFoundError:
        raise pytest.skip.Exception("native scope acceptance requires a disposable VM authorization marker")
    assert stat.S_ISREG(details.st_mode) and details.st_uid == 0 and not details.st_mode & 0o022
    uid = os.getuid()
    assert uid != 0 and os.geteuid() == uid, "native acceptance must use an unprivileged account"
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    assert marker.read_text().split() == [boot_id, str(uid)], "marker is not for this boot and user"
    assert Path("/sys/fs/cgroup/cgroup.controllers").is_file(), "unified cgroup v2 required"
    assert process_registry._systemd_run_user_scope_available(), "test account needs a working user manager"

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv("HERMES_SUPERVISED_CHILD", "1")
    monkeypatch.setenv("INVOCATION_ID", "test-supervisor")
    monkeypatch.setattr("agent.runtime_session_store._worker_process", True)
    # Enable only the test process's settings reader, not any persisted config.
    monkeypatch.setattr(foreground_scope, "read_raw_config_readonly", lambda: {
        "terminal": {"foreground_scope": {"enabled": True}},
    })
    task = f"scope-vm-{uuid.uuid4().hex}"
    try:
        yield task
    finally:
        cleanup_vm(task)


def _process_birth(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
    return int(fields[19])


@pytest.fixture
def sibling(disposable_vm):
    # A new process outside the scope that can answer a fresh challenge.
    script = "import sys\nfor line in sys.stdin: print(line.strip(), flush=True)"
    with subprocess.Popen([sys.executable, "-u", "-c", script], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, start_new_session=True) as proc:
        assert proc.stdin is not None and proc.stdout is not None
        try:
            yield proc
        finally:
            proc.stdin.close()
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=5)


def _responsive(proc, challenge):
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(challenge + b"\n")
    proc.stdin.flush()
    assert select.select([proc.stdout], [], [], 5)[0], "sibling did not answer"
    assert proc.stdout.readline() == challenge + b"\n"


def _original_empty(directory, events):
    try:
        if b"populated 0" in os.pread(events, 4096, 0).splitlines():
            return True
    except OSError:
        # An invalidated events descriptor is not success by itself.
        pass
    try:
        directory.stat()
    except FileNotFoundError:
        return True
    return False


@contextmanager
def _owned_pidfd(pid, birth):
    fd = os.pidfd_open(pid)
    try:
        assert _process_birth(pid) == birth
        try:
            yield fd
        finally:
            if not select.select([fd], [], [], 0)[0]:
                try:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                assert select.select([fd], [], [], 5)[0], "fixture descendant did not exit"
    finally:
        os.close(fd)


def test_yielded_stop_drains_the_original_tree(disposable_vm, sibling, tmp_path, monkeypatch):
    ready = tmp_path / "ready.json"
    # The fixture child has a separate session and a bounded lifetime even if
    # the terminal itself cannot clean up. Publish readiness atomically.
    payload = tmp_path / "owned_tree.py"
    payload.write_text(
        "import json, subprocess, sys\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], "
        "start_new_session=True)\n"
        "raw = Path(f'/proc/{child.pid}/stat').read_text().rpartition(')')[2].split()\n"
        "group = Path(f'/proc/{child.pid}/cgroup').read_text().strip().removeprefix('0::')\n"
        "ready = Path(sys.argv[1])\n"
        "staging = ready.with_suffix('.tmp')\n"
        "staging.write_text(json.dumps([child.pid, int(raw[19]), group]))\n"
        "staging.replace(ready)\n"
        "sys.exit(child.wait(timeout=35))\n"
    )
    monkeypatch.setattr(base, "consume_yield", lambda _: ready.is_file())
    sibling_birth = _process_birth(sibling.pid)
    _responsive(sibling, b"before")
    command = shlex.join([sys.executable, str(payload), str(ready)])
    yielded = json.loads(terminal_tool(command=command, timeout=15, task_id=disposable_vm))
    assert yielded["status"] == "yielded_to_background", yielded
    pid, birth, group = json.loads(ready.read_text())
    assert group.startswith("/") and Path(group).name.startswith("hermes-worker-fg-")
    assert Path(group).suffix == ".scope" and ".." not in Path(group).parts
    assert _process_birth(pid) == birth
    with _owned_pidfd(pid, birth) as child_fd:
        assert os.getsid(pid) == pid, "child must be outside the cooperative process group"
        directory = Path("/sys/fs/cgroup") / group.lstrip("/")
        with (directory / "cgroup.events").open("rb") as events:
            assert b"populated 1" in events.read().splitlines()
            result = process_registry.process_registry.kill_process(yielded["session_id"])
            assert result["status"] == "killed", result
            assert select.select([child_fd], [], [], 5)[0], "the original descendant did not exit"
            assert wait_until(lambda: _original_empty(directory, events.fileno()), timeout=5)
        assert _process_birth(sibling.pid) == sibling_birth
        _responsive(sibling, b"after")
