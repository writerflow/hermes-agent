"""Foreground terminal commands in their own capped ``hermes-worker-fg-*`` systemd scope.

On a managed worker (``python -m agent.managed_worker``) or the supervised gateway, every foreground
command is a descendant of a process that sits in the gateway's cgroup, so no systemd limit reaches
a build's fan-out. Behind ``terminal.foreground_scope.enabled`` (default off) the terminal tool's
foreground command starts in its own transient scope. The scope carries only ``MemoryAccounting=yes``:
CPUQuota, CPUWeight, TasksMax, MemoryHigh and MemorySwapMax come from systemd drop-ins named
``hermes-worker-fg-.scope.d`` so operators tune them without a release. No ``MemoryMax`` is set anywhere:
a hard ceiling needs measured per-command sizes.

``LocalEnvironment._run_bash`` asks :func:`scope_command` for the wrapper; every "no" answer leaves
today's argv and environment untouched.
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from threading import Lock
from typing import BinaryIO, Callable, Mapping

from agent.memory_provider import spawn_context_thread
from hermes_cli._subprocess_compat import process_start_time_stdlib
from hermes_cli.config import cfg_get, read_raw_config_readonly
from hermes_platform.resolver import locate_command

logger = logging.getLogger(__name__)


def _registry():
    """tools.process_registry, imported on use: its import chain reaches tools.environments.local,
    which imports this module."""
    from tools import process_registry
    return process_registry

_GRADLE_PER_COMMAND = "per_command"
_GRADLE_MODES = (_GRADLE_PER_COMMAND, "shared")
_NO_DAEMON_FLAG = "-Dorg.gradle.daemon=false"
_UNIT_PREFIX = "hermes-worker-fg-"
_UNIT_TASK_CHARS = 8

# systemd-run reports a bus-open failure before entering start_transient_scope().
# Match only a complete, standalone launcher diagnostic, not lines in a job error.
_WRAPPER_FAILURE = re.compile(r"Failed to (connect to (the )?(user )?bus|create bus connection|"
                              r"connect to user scope bus): [^\r\n]+", re.IGNORECASE)
_MAX_WRAPPER_STDERR = 8192
_ADMISSION_TIMEOUT = 5.0
_CLEANUP_TIMEOUT = 0.75
_CREDENTIALS = struct.Struct("iII")  # Linux struct ucred: pid, uid, gid
_ANCHOR_PID = struct.Struct("=I")
# Record that the post-placement helper ran before allowing any payload code.
# This byte vetoes replay; it does not prove successful placement or grant cleanup authority.
_EXEC_PAYLOAD = """\
import ctypes, os, signal, socket, struct, sys

def run_payload(gate, status_fd):
    os.close(status_fd)
    with gate:
        try:
            admitted = gate.recv(2) == b'G'
        except OSError:
            admitted = False
    if not admitted:
        os._exit(125)
    os.dup2(1, 2)
    os.execvp(sys.argv[3], sys.argv[3:])

def close_stdio():
    for fd in (0, 1, 2):
        os.close(fd)

def reap_payload(gate, status_fd):
    # Keep the ancestry root out of the command's process-group teardown,
    # without moving either process out of the original scope.
    command_group = os.getpgrp()
    os.setpgid(0, 0)
    # PR_SET_CHILD_SUBREAPER: detached orphans stay attributable to this launch.
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        os._exit(125)
    payload = os.fork()
    if not payload:
        os.setpgid(0, command_group)
        run_payload(gate, status_fd)
    gate.close()
    close_stdio()
    os.write(status_fd, b'R')
    while True:
        try:
            pid, status = os.waitpid(-1, 0)
        except ChildProcessError:
            os._exit(0)
        if pid == payload:
            # Publish the payload result now, without waiting on a deliberately
            # persistent descendant or keeping the terminal's streams open.
            try:
                os.write(status_fd, struct.pack('i', os.waitstatus_to_exitcode(status)))
            except BrokenPipeError:
                pass  # The result waiter has gone away; still reap adopted children.
            os.close(status_fd)

fd, gate_fd = map(int, sys.argv[1:3])
if os.pwrite(fd, b'1', 0) != 1:
    raise SystemExit(125)
os.close(fd)
gate = socket.socket(fileno=gate_fd)
gate.settimeout(10)
read_fd, write_fd = os.pipe()
anchor_pid = os.fork()
if not anchor_pid:
    os.close(read_fd)
    reap_payload(gate, write_fd)
os.close(write_fd)
with os.fdopen(read_fd, 'rb') as status:
    with gate:
        if status.read(1) == b'R':
            try:
                gate.send(b'R' + struct.pack('=I', anchor_pid)
                          + os.environ.get('INVOCATION_ID', '').encode('ascii'))
            except (OSError, UnicodeError):
                pass  # No release can arrive; the held payload will refuse.
    close_stdio()
    result = status.read(4)
if len(result) != 4:
    os._exit(125)
code, = struct.unpack('i', result)
if code < 0:
    if -code != signal.SIGKILL:
        signal.signal(-code, signal.SIG_DFL)
    os.kill(os.getpid(), -code)
os._exit(code)
"""
# The one limit text a probe has confirmed: bash's fork failure under TasksMax (EAGAIN).
_TASKS_LIMIT_TEXT = "Resource temporarily unavailable"
LIMIT_NOTE = ("[hermes-worker-fg scope: this command hit the scope's task limit (TasksMax); to turn per-command "
              "scopes off set terminal.foreground_scope.enabled: false]")

_invalid_modes_logged: set[str] = set()


@dataclass(frozen=True)
class ScopedSpawn:
    """The wrapped argv and environment for one command, and the scope's unit name (with ``.scope``)."""
    argv: list[str]
    env: dict[str, str]
    unit: str


@dataclass(frozen=True)
class _Settings:
    enabled: bool
    gradle_mode: str


@dataclass(frozen=True)
class _LaunchStreams:
    stderr: BinaryIO
    entered: BinaryIO


@dataclass(frozen=True)
class _PinnedScope:
    boot: str
    launcher: tuple[int, int]
    anchor: tuple[int, int]
    placement: tuple[str, str, str, str, int]
    handle: tuple[int, int]
    directory: int
    events: int
    freeze: int
    kill: int
    files: ExitStack

    def close(self) -> None:
        self.files.close()

    def _events(self) -> tuple[bytes, bytes]:
        rows = [row.split() for row in os.pread(self.events, 4096, 0).splitlines()]
        values = []
        for key in (b"populated", b"frozen"):
            (row,) = [row for row in rows if row[:1] == [key]]
            if len(row) != 2 or row[1] not in (b"0", b"1"):
                raise ValueError("invalid scope event")
            values.append(row[1])
        return values[0], values[1]

    def _wait_for_event(self, index: int, value: bytes, deadline: float) -> bool:
        while self._events()[index] != value:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(remaining, 0.01))
        return True

    def drain(self, *, timeout: float = _CLEANUP_TIMEOUT) -> str | None:
        # Admission owns these FDs. No fresh unit or path resolution may grant
        # authority; an invalidated original descriptor only permits refusal.
        deadline = time.monotonic() + timeout
        try:
            populated, frozen = self._events()
            if populated == b"0":
                return None
            if frozen != b"0":
                return "original scope already frozen"
            try:
                _write_control(self.freeze, b"1")
                if not self._wait_for_event(1, b"1", deadline):
                    return "original scope still populated; freeze unconfirmed"
                if not _verified_frozen_members(self, deadline) or self._events()[1] != b"1":
                    return "original scope member provenance unverified"
                _write_control(self.kill, b"1")
                if not self._wait_for_event(0, b"0", deadline):
                    return "original scope still populated after teardown"
            finally:
                # Always release our own freeze request, even on refusal.
                _write_control(self.freeze, b"0")
        except (OSError, ValueError, LookupError, RecursionError):
            return "original scope state unverified"
        return None


def _read_placement(pid: int, unit: str, env: Mapping[str, str]) -> tuple[str, str, str, str, int]:
    # Resolve by the held PID, not by the requested name. Pin the D-Bus bus
    # epoch and unique service owner so a replacement manager cannot stand in.
    command = locate_command("busctl").command
    if not command:
        raise ValueError("no manager client")

    def call(signature, *args):
        reply = subprocess.run(
            [*command, "--user", "--json=short", "--auto-start=no", "call", *args],
            env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=0.75, check=True,
        )
        message = json.loads(reply.stdout)
        if message["type"] != signature:
            raise ValueError("unexpected manager reply")
        value, = message["data"]
        return value

    def property_value(properties, name, signature, expected_type):
        field = properties[name]
        value = field["data"]
        if field["type"] != signature or type(value) is not expected_type:
            raise ValueError("unexpected manager property")
        return value

    bus = "org.freedesktop.DBus"
    dbus = (bus, "/org/freedesktop/DBus", bus)
    systemd = "org.freedesktop.systemd1"
    bus_id = call("s", *dbus, "GetId")
    owner = call("s", *dbus, "GetNameOwner", "s", systemd)
    if not re.fullmatch(r":[0-9]+\.[0-9]+", owner) or not re.fullmatch(r"[0-9a-f]{32}", bus_id):
        raise ValueError("unverified manager connection")
    obj = call("o", owner, "/org/freedesktop/systemd1", f"{systemd}.Manager", "GetUnitByPID", "u", str(pid))
    props = (owner, obj, "org.freedesktop.DBus.Properties", "GetAll", "s")
    info = call("a{sv}", *props, f"{systemd}.Unit")
    if any(property_value(info, key, signature, type(value)) != value for key, signature, value in
           (("Id", "s", unit), ("ActiveState", "s", "active"), ("Transient", "b", True))):
        raise ValueError("held PID not in active transient scope")
    scope = call("a{sv}", *props, f"{systemd}.Scope")
    # Each busctl call opens its own connection. Reject a change across reads
    # rather than interpreting a reused unique owner as the same manager.
    if (call("s", *dbus, "GetNameOwner", "s", systemd) != owner
            or call("s", *dbus, "GetId") != bus_id):
        raise ValueError("manager connection changed")
    path = property_value(scope, "ControlGroup", "s", str)
    invocation = property_value(info, "InvocationID", "ay", list)
    generation = property_value(scope, "ControlGroupId", "t", int)
    if len(invocation) != 16 or any(type(byte) is not int for byte in invocation) or not 0 < generation < 2**64:
        raise ValueError("invalid scope identity")
    return bus_id, owner, path, bytes(invocation).hex(), generation


def _cgroup_mount() -> tuple[int, str]:
    # Only the full, canonical cgroup-v2 hierarchy is supported. Do not guess
    # path translation for a subtree bind or an unfamiliar mount namespace.
    with open("/proc/self/mountinfo", encoding="utf-8") as file:
        for line in file:
            left, _, right = line.partition(" - ")
            mount_id, _, _, root, mount, *_ = left.split()
            if root == "/" and mount == "/sys/fs/cgroup" and right.startswith("cgroup2 "):
                return int(mount_id), mount
    raise ValueError("no canonical cgroup v2 mount")


def _unified_membership(pid: int) -> str:
    with open(f"/proc/{pid}/cgroup", encoding="utf-8") as file:
        rows = file.read().splitlines()
    if len(rows) != 1 or not rows[0].startswith("0::/"):
        raise ValueError("no unified cgroup membership")
    return rows[0][3:]


def _kernel_handle(fd: int) -> tuple[int, int]:
    # Linux name_to_handle_at: kernfs returns its 64-bit object generation
    # identifier, not just st_ino. We never reopen an object by handle.
    handle = ctypes.create_string_buffer(struct.pack("Ii8x", 8, 0), 16)
    mount_id = ctypes.c_int()
    call = ctypes.CDLL(None, use_errno=True).name_to_handle_at
    if call(fd, b".", ctypes.byref(handle), ctypes.byref(mount_id), 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot identify cgroup object")
    size, kind = struct.unpack_from("Ii", handle)
    if (size, kind) != (8, 0xFE):  # FILEID_KERNFS
        raise ValueError("no kernfs generation handle")
    return mount_id.value, struct.unpack_from("Q", handle, 8)[0]


def _fd_mount_id(fd: int) -> int:
    with open(f"/proc/self/fdinfo/{fd}", encoding="ascii") as file:
        for line in file:
            if line.startswith("mnt_id:"):
                return int(line.split()[1])
    raise ValueError("no mount identity for cgroup control")


def _open_cgroup(mount: str, parts: list[str], files: ExitStack) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(mount, flags)
    try:
        for part in parts:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        files.callback(os.close, fd)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_control(directory: int, name: str) -> str:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
    with os.fdopen(fd, encoding="ascii") as file:
        return file.read()


def _process_parent_birth(pid: int) -> tuple[int, int]:
    """Read parent PID and process start ticks in one procfs snapshot."""
    with open(f"/proc/{pid}/stat", "rb") as file:
        fields = file.read().rsplit(b")", 1)[1].split()
    if fields[0] in (b"Z", b"X", b"x"):
        raise ValueError("process no longer live")
    return int(fields[1]), int(fields[19])


def _write_control(fd: int, value: bytes) -> None:
    if os.pwrite(fd, value, 0) != len(value):
        raise OSError("incomplete scope control write")


def _member_snapshot(directory: int, path: str, mount: int, deadline: float,
                     members: dict[int, tuple[int, int]]) -> None:
    if time.monotonic() >= deadline or _fd_mount_id(directory) != mount:
        raise ValueError("scope traversal unverified")
    fd = os.open("cgroup.procs", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
    with os.fdopen(fd, encoding="ascii") as file:
        for line in file:
            pid = int(line)
            if time.monotonic() >= deadline or pid <= 0 or pid in members:
                raise ValueError("scope membership changed")
            snapshot = _process_parent_birth(pid)
            if _unified_membership(pid) != path or _process_parent_birth(pid) != snapshot:
                raise ValueError("scope process identity changed")
            members[pid] = snapshot
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                child = os.open(entry.name, flags, dir_fd=directory)
                try:
                    _member_snapshot(child, f"{path}/{entry.name}", mount, deadline, members)
                finally:
                    os.close(child)


def _verified_frozen_members(scope: _PinnedScope, deadline: float) -> bool:
    path, mount = scope.placement[2], scope.handle[0]
    members: dict[int, tuple[int, int]] = {}
    _member_snapshot(scope.directory, path, mount, deadline, members)
    anchor, birth = scope.anchor
    if anchor not in members or members[anchor][1] != birth:
        return False
    owned = {anchor}
    launcher, birth = scope.launcher
    if launcher in members and members[launcher][1] == birth:
        owned.add(launcher)
    for pid in members:
        chain = set()
        while pid not in owned:
            if pid not in members or pid in chain or time.monotonic() >= deadline:
                return False
            chain.add(pid)
            parent, birth = members[pid]
            if parent not in members or members[parent][1] > birth:
                return False
            pid = parent
        owned.update(chain)
    # A freeze does not prevent external cgroup writers from moving a process.
    # Reject any drift we observe; this is not a sandbox for other writers.
    current: dict[int, tuple[int, int]] = {}
    _member_snapshot(scope.directory, path, mount, deadline, current)
    return current == members and time.monotonic() < deadline


def _verify_held_members(directory: int, path: str, launcher: tuple[int, int],
                         anchor: tuple[int, int]) -> bool:
    pid, birth = launcher
    anchor_pid, anchor_birth = anchor
    members = set(_read_control(directory, "cgroup.procs").splitlines())
    return ({str(pid), str(anchor_pid)} <= members
            and _unified_membership(pid) == _unified_membership(anchor_pid) == path
            and _process_parent_birth(anchor_pid) == (pid, anchor_birth)
            and process_start_time_stdlib(pid) == birth)


def _pin_original(proc: subprocess.Popen, unit: str, invocation_id: bytes,
                  anchor_pid: int) -> _PinnedScope | None:
    if not re.fullmatch(rb"[0-9a-f]{32}", invocation_id) or invocation_id == b"0" * 32:
        return None
    try:
        env, birth = getattr(proc, "_hermes_scope_launch")
        if birth is None or process_start_time_stdlib(proc.pid) != birth:
            return None
        parent, anchor_birth = _process_parent_birth(anchor_pid)
        if parent != proc.pid or anchor_birth < birth:
            return None
        with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as file:
            boot = file.read().strip()
        placement = _read_placement(proc.pid, unit, env)
        _, _, path, invocation, cgroup_id = placement
        parts = path[1:].split("/")
        if (invocation_id.decode("ascii") != invocation or not path.startswith("/") or parts[-1] != unit
                or any(part in ("", ".", "..") for part in parts) or _unified_membership(proc.pid) != path):
            return None
        mount_id, mount = _cgroup_mount()
        with ExitStack() as files:
            directory = _open_cgroup(mount, parts, files)
            handle = _kernel_handle(directory)
            if (not cgroup_id or handle != (mount_id, cgroup_id) or
                    _read_control(directory, "cgroup.type").strip() != "domain"):
                return None
            controls = []
            for name, access in (("events", os.O_RDONLY), ("freeze", os.O_WRONLY), ("kill", os.O_WRONLY)):
                fd = os.open(f"cgroup.{name}", access | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
                files.callback(os.close, fd)
                if _fd_mount_id(fd) != mount_id:
                    return None
                controls.append(fd)
            if _read_placement(proc.pid, unit, env) != placement:
                return None
            with ExitStack() as current:
                if _kernel_handle(_open_cgroup(mount, parts, current)) != handle:
                    return None
            launcher, anchor = (proc.pid, birth), (anchor_pid, anchor_birth)
            if not _verify_held_members(directory, path, launcher, anchor):
                return None
            events, freeze, kill = controls
            return _PinnedScope(boot, launcher, anchor, placement, handle, directory,
                                events, freeze, kill, files.pop_all())
    except (OSError, ValueError, TypeError, LookupError, AttributeError, subprocess.SubprocessError):
        # No command, env, or manager output in logs. A missing capability is
        # indistinguishable from uncertain placement for admission purposes.
        return None


class _HeldAdmission:
    def __init__(self, channel: socket.socket):
        self._channel = channel
        self._fence = Lock()
        self._closed = False
        self._original: _PinnedScope | None = None
        self._drained = False

    def verify_and_release(self, proc: subprocess.Popen, unit: str) -> None:
        try:
            receipt = self._receive(proc.pid)
            if receipt is None or proc.poll() is not None:
                return
            invocation_id, anchor_pid = receipt
            original = _pin_original(proc, unit, invocation_id, anchor_pid)
            if original is None:
                logger.warning("Foreground scope %s: placement unverified; payload not released", unit)
                return
            with self._fence:
                if not self._closed and proc.poll() is None:
                    self._original = original
                    self._channel.sendall(b"G")
                else:
                    original.close()
        except OSError:
            logger.debug("Foreground scope %s: admission channel closed", unit)
        finally:
            with self._fence:
                self._closed = True
                self._channel.close()

    def _receive(self, pid: int) -> tuple[bytes, int] | None:
        header = 1 + _ANCHOR_PID.size
        data, ancillary, flags, _ = self._channel.recvmsg(header + 32, socket.CMSG_SPACE(_CREDENTIALS.size))
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or not data.startswith(b"R") or len(data) < header:
            return None
        for level, kind, credentials in ancillary:
            if (level, kind, len(credentials)) == (socket.SOL_SOCKET, socket.SCM_CREDENTIALS, _CREDENTIALS.size):
                sender, uid, _ = _CREDENTIALS.unpack(credentials)
                if (sender, uid) == (pid, os.geteuid()):
                    anchor_pid, = _ANCHOR_PID.unpack_from(data, 1)
                    if 0 < anchor_pid < 2**31 and anchor_pid != pid:
                        return data[header:], anchor_pid
        return None

    def cancel(self) -> bool:
        # The verifier may be doing I/O. Serialize only the final release with
        # cancellation, never wait for verification while holding the fence.
        with self._fence:
            if not self._closed:
                self._closed = True
                with suppress(OSError):
                    self._channel.shutdown(socket.SHUT_RDWR)
            return self._original is not None

    def close(self) -> None:
        self.cancel()
        with self._fence:
            if self._original is not None:
                self._original.close()
                self._original = None

    def cleanup_original(self, unit: str, *, timeout: float,
                         prepare: Callable[[], None] | None = None) -> str | None:
        with self._fence:
            if self._drained:
                return None
            if self._original is None:
                return "original scope unavailable" if prepare is not None else None
            if prepare is not None:
                # The reader can reap the launcher during cooperative shutdown,
                # but must not close the pin until this explicit stop has used it.
                with suppress(OSError):
                    prepare()
            reason = self._original.drain(timeout=timeout)
            self._drained = reason is None
        if reason is not None:
            logger.warning("Foreground scope %s: cleanup incomplete: %s", unit, reason)
        return reason


def foreground_scope_host() -> bool:
    """Whether this process is a place whose foreground commands belong in a scope: a managed worker or the
    live supervised gateway. The supervisor markers are inherited by every descendant (a ``hermes`` CLI started
    inside a build has them), so they alone cannot decide; the worker flag is process-scoped and not inherited.
    Both gates only choose a resource cap, never a permission."""
    if not _registry()._IS_LINUX:
        return False
    from gateway.restart import is_supervised_gateway_launch
    if not is_supervised_gateway_launch():
        return False
    # Imported here: the worker store pulls in websockets, which the tool layer must not load at import.
    from agent.runtime_session_store import is_worker_process
    return is_worker_process() or _registry()._is_supervised_gateway_process()


def _settings() -> _Settings:
    """Read per command so flipping the flag needs no restart. ``read_raw_config`` returns ``{}`` for a missing
    or unparseable file, which means off."""
    section = cfg_get(read_raw_config_readonly(), "terminal", "foreground_scope", default={})
    if not isinstance(section, Mapping):
        return _Settings(False, _GRADLE_PER_COMMAND)
    mode = section.get("gradle_daemon", _GRADLE_PER_COMMAND)
    if mode not in _GRADLE_MODES:
        if str(mode) not in _invalid_modes_logged:
            _invalid_modes_logged.add(str(mode))
            logger.warning("terminal.foreground_scope.gradle_daemon %r is not one of %s; using %s",
                           mode, ", ".join(_GRADLE_MODES), _GRADLE_PER_COMMAND)
        mode = _GRADLE_PER_COMMAND
    return _Settings(section.get("enabled") is True, mode)


def _unit_name() -> str:
    task = re.sub(r"[^A-Za-z0-9_]", "", os.environ.get("HERMES_KANBAN_TASK", ""))[:_UNIT_TASK_CHARS] or "x"
    return f"{_UNIT_PREFIX}{task}-{uuid.uuid4().hex[:8]}"


def _gradle_env(env: dict[str, str], mode: str) -> dict[str, str]:
    """``per_command``: Gradle forks a single-use daemon inside this command's scope and stops it at the end of
    the build, so the build's java is charged to this command. A user's own ``org.gradle.daemon`` choice wins."""
    if mode != _GRADLE_PER_COMMAND:
        return env
    opts = env.get("GRADLE_OPTS", "")
    if "org.gradle.daemon" not in opts:
        env["GRADLE_OPTS"] = f"{opts} {_NO_DAEMON_FLAG}".strip()
    return env


def scope_command(args: list[str], run_env: Mapping[str, str]) -> ScopedSpawn | None:
    """The scope-wrapped spawn for one foreground command, or None for today's unscoped spawn: feature off,
    not a worker/gateway, or no usable systemd (the last warns once per process)."""
    settings = _settings()
    if not settings.enabled or not foreground_scope_host():
        return None
    binary = shutil.which("systemd-run") if _registry()._systemd_run_user_scope_available() else None
    if binary is None:
        _registry()._warn_scope_degraded_once(
            "foreground terminal scope enabled but systemd-run --user --scope is unavailable",
            consequence="foreground commands run unscoped in the worker's own cgroup, without per-command limits.")
        return None
    unit = _unit_name()
    no_expand = ["--expand-environment=no"] if _registry()._SYSTEMD_RUN_NO_EXPAND else []
    argv = [binary, "--user", "--scope", "--quiet", *no_expand, "--unit", unit, "--collect",
            "--property", "MemoryAccounting=yes", "--", *args]
    env = _gradle_env(_registry().systemd_user_bus_env(dict(run_env)), settings.gradle_mode)
    return ScopedSpawn(argv, env, f"{unit}.scope")


def scope_unit_of(proc: object) -> str:
    """The foreground scope a process was spawned into (<unit>.scope), or "" for any unscoped handle
    (a test double's attributes are never strings)."""
    unit = getattr(proc, "_hermes_scope_unit", "")
    return unit if isinstance(unit, str) else ""


def spawn_scoped(argv: list[str], **kwargs) -> subprocess.Popen:
    """Keep launcher diagnostics and the no-replay latch private from the payload."""
    split = argv.index("--") + 1
    with ExitStack() as files:
        diagnostics = files.enter_context(tempfile.TemporaryFile(mode="w+b"))
        entered = files.enter_context(tempfile.TemporaryFile(mode="w+b"))
        parent, child = socket.socketpair(type=socket.SOCK_SEQPACKET)
        files.enter_context(parent)
        parent.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        parent.settimeout(_ADMISSION_TIMEOUT)
        fd = entered.fileno()
        # Isolated interpreter: no user startup imports before the latch is set.
        with child:
            gate_fd = child.fileno()
            cmd = [*argv[:split], sys.executable, "-I", "-S", "-c", _EXEC_PAYLOAD,
                   str(fd), str(gate_fd), *argv[split:]]
            # systemd-run merges inherited environment with the invocation it
            # reads from the manager. Do not accept an inherited value as evidence.
            env = {key: value for key, value in kwargs.pop("env").items() if key != "INVOCATION_ID"}
            kwargs.update(stderr=diagnostics, pass_fds=(fd, gate_fd), env=env)
            proc = subprocess.Popen(cmd, **kwargs)
        setattr(proc, "_hermes_scope_launch", (env, process_start_time_stdlib(proc.pid)))
        setattr(proc, "_hermes_scope_streams", _LaunchStreams(diagnostics, entered))
        setattr(proc, "_hermes_scope_admission", _HeldAdmission(parent))
        files.pop_all()
    return proc


def start_admission(proc: subprocess.Popen, unit: str) -> None:
    admission: _HeldAdmission = getattr(proc, "_hermes_scope_admission")
    thread = spawn_context_thread(lambda: admission.verify_and_release(proc, unit), name="foreground-scope-admission")
    thread.start()


def cancel_admission(proc: object) -> bool:
    """Close the release gate; return whether cleanup owns a verified original."""
    admission = getattr(proc, "_hermes_scope_admission", None)
    return admission is not None and admission.cancel()


def cleanup_original(proc: object, *, timeout: float = _CLEANUP_TIMEOUT) -> None:
    if (admission := getattr(proc, "_hermes_scope_admission", None)) is not None:
        admission.cleanup_original(scope_unit_of(proc), timeout=timeout)


def stop_adopted_scope(proc: subprocess.Popen, prepare: Callable[[], None]) -> str | None:
    admission: _HeldAdmission = getattr(proc, "_hermes_scope_admission")
    admission.cancel()
    return admission.cleanup_original(scope_unit_of(proc), timeout=_CLEANUP_TIMEOUT, prepare=prepare)


def wrapper_diagnostic(proc: subprocess.Popen, returncode: int | None) -> tuple[str, bool]:
    """Return diagnostics and whether they prove the legacy bus-open failure."""
    streams: _LaunchStreams = getattr(proc, "_hermes_scope_streams")
    raw = os.pread(streams.stderr.fileno(), _MAX_WRAPPER_STDERR + 1, 0)
    text = raw[:_MAX_WRAPPER_STDERR].decode("utf-8", "replace").removesuffix("\n")
    pre_start = (returncode == 1 and len(raw) <= _MAX_WRAPPER_STDERR
                 and _WRAPPER_FAILURE.fullmatch(text) is not None
                 and not os.pread(streams.entered.fileno(), 1, 0))
    return text, pre_start


def close_admission(proc: object) -> None:
    if (admission := getattr(proc, "_hermes_scope_admission", None)) is not None:
        admission.close()


def close_spawn_streams(proc: subprocess.Popen) -> None:
    streams: _LaunchStreams = getattr(proc, "_hermes_scope_streams")
    streams.stderr.close()
    streams.entered.close()



def limit_note(returncode: int | None, output: str) -> str:
    """One line for the agent when a scoped command failed on the task limit, else empty."""
    return LIMIT_NOTE if returncode and _TASKS_LIMIT_TEXT in output else ""


def warn_bus_lost(spawn_env: Mapping[str, str]) -> None:
    """The wrapper died before starting the command: let the shared probe verdict follow the bus."""
    _registry().scoped_spawn_lost_user_bus(dict(spawn_env))
    _registry()._warn_scope_degraded_once(
        "foreground terminal scope could not be created for a command",
        consequence="that command ran once unscoped; the next command re-checks.")
