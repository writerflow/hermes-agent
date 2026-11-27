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
import uuid
from contextlib import ExitStack, suppress
from dataclasses import dataclass
from threading import Lock
from typing import BinaryIO, Mapping

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
_CREDENTIALS = struct.Struct("iII")  # Linux struct ucred: pid, uid, gid
# Record that the post-placement helper ran before allowing any payload code.
# This byte vetoes replay; it does not prove successful placement or grant cleanup authority.
_EXEC_PAYLOAD = """\
import os, socket, sys
fd, gate_fd = map(int, sys.argv[1:3])
if os.pwrite(fd, b'1', 0) != 1:
    raise SystemExit(125)
os.close(fd)
with socket.socket(fileno=gate_fd) as gate:
    gate.settimeout(10)
    try:
        gate.send(b'R' + os.environ.get('INVOCATION_ID', '').encode('ascii'))
        admitted = gate.recv(2) == b'G'
    except (OSError, UnicodeError):
        admitted = False
if not admitted:
    raise SystemExit(125)
os.dup2(1, 2)
os.execvp(sys.argv[3], sys.argv[3:])
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
    placement: tuple[str, str, str, str, int]
    handle: tuple[int, int]
    directory: int
    events: int
    freeze: int
    kill: int
    files: ExitStack

    def close(self) -> None:
        self.files.close()


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


def _pin_original(proc: subprocess.Popen, unit: str, invocation_id: bytes) -> _PinnedScope | None:
    if not re.fullmatch(rb"[0-9a-f]{32}", invocation_id) or invocation_id == b"0" * 32:
        return None
    try:
        env, birth = getattr(proc, "_hermes_scope_launch")
        if birth is None or process_start_time_stdlib(proc.pid) != birth:
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
            if (_unified_membership(proc.pid) != path or
                    str(proc.pid) not in _read_control(directory, "cgroup.procs").splitlines() or
                    process_start_time_stdlib(proc.pid) != birth):
                return None
            events, freeze, kill = controls
            return _PinnedScope(boot, (proc.pid, birth), placement, handle, directory,
                                events, freeze, kill, files.pop_all())
    except (OSError, ValueError, TypeError, KeyError, AttributeError, subprocess.SubprocessError):
        # No command, env, or manager output in logs. A missing capability is
        # indistinguishable from uncertain placement for admission purposes.
        return None


class _HeldAdmission:
    def __init__(self, channel: socket.socket):
        self._channel = channel
        self._fence = Lock()
        self._closed = False
        self._original: _PinnedScope | None = None

    def verify_and_release(self, proc: subprocess.Popen, unit: str) -> None:
        try:
            invocation_id = self._receive(proc.pid)
            if invocation_id is None or proc.poll() is not None:
                return
            original = _pin_original(proc, unit, invocation_id)
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

    def _receive(self, pid: int) -> bytes | None:
        data, ancillary, flags, _ = self._channel.recvmsg(34, socket.CMSG_SPACE(_CREDENTIALS.size))
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or not data.startswith(b"R"):
            return None
        for level, kind, credentials in ancillary:
            if (level, kind, len(credentials)) == (socket.SOL_SOCKET, socket.SCM_CREDENTIALS, _CREDENTIALS.size):
                sender, uid, _ = _CREDENTIALS.unpack(credentials)
                if (sender, uid) == (pid, os.geteuid()):
                    return data[1:]
        return None

    def cancel(self) -> None:
        # The verifier may be doing I/O. Serialize only the final release with
        # cancellation, never wait for verification while holding the fence.
        with self._fence:
            if not self._closed:
                self._closed = True
                with suppress(OSError):
                    self._channel.shutdown(socket.SHUT_RDWR)

    def close(self) -> None:
        self.cancel()
        with self._fence:
            if self._original is not None:
                self._original.close()
                self._original = None


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


def cancel_admission(proc: object) -> None:
    if (admission := getattr(proc, "_hermes_scope_admission", None)) is not None:
        admission.cancel()


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
