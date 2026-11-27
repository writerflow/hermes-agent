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

import logging
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from typing import Mapping

from hermes_cli.config import cfg_get, read_raw_config_readonly

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

# What the wrapper prints when it fails before the command could start (bus lost after the probe).
_WRAPPER_FAILURE = re.compile(r"Failed to (connect to (the )?(user )?bus|start transient scope|create bus connection|"
                              r"connect to user scope bus)", re.IGNORECASE)
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


def scope_never_started(returncode: int | None, output: str, cwd_marker: str) -> bool:
    """The wrapper failed before the command began: non-zero, a wrapper failure text, and no end marker. A
    command that started always reaches its marker, so it is never re-run."""
    return bool(returncode) and cwd_marker not in output and _WRAPPER_FAILURE.search(output) is not None


def limit_note(returncode: int | None, output: str) -> str:
    """One line for the agent when a scoped command failed on the task limit, else empty."""
    return LIMIT_NOTE if returncode and _TASKS_LIMIT_TEXT in output else ""


def warn_bus_lost(spawn_env: Mapping[str, str]) -> None:
    """The wrapper died before starting the command: let the shared probe verdict follow the bus."""
    _registry().scoped_spawn_lost_user_bus(dict(spawn_env))
    _registry()._warn_scope_degraded_once(
        "foreground terminal scope could not be created for a command",
        consequence="that command ran once unscoped; the next command re-checks.")
