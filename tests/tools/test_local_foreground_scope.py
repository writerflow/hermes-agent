"""Foreground terminal commands run in their own capped ``hermes-worker-fg-*`` systemd scope.

On a managed MacPro worker every foreground command (``./gradlew test``, ``vitest``) is a child of
the worker process, so no systemd limit reaches the build fan-out. Behind
``terminal.foreground_scope.enabled`` (default off) the terminal tool's foreground command starts in
its own transient scope whose limits come from systemd drop-ins.

Gate rule for this file: no test patches an identity gate (``foreground_scope_host``,
``_is_supervised_gateway_process``, ``is_worker_process``, ``is_supervised_gateway_launch``,
``get_running_pid``). Host shapes are built from what the real gate reads: environment variables,
the real ``gateway.pid`` / runtime-lock writers in a temp ``HERMES_HOME``, and the module flag a
successful worker adopt sets. Host capability (the systemd-run probe) is seeded through its own
cache, not by replacing the function. ``test_gate_patch_guard`` enforces the rule.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest

from tools import foreground_scope as fs
from tools import process_registry as pr
from tools.environments import local as local_env
from tools.environments.local import LocalEnvironment

pytestmark = pytest.mark.spawns_gateway_lookalike  # host shapes run real `gateway run` lookalike holders

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_NAMES = (
    "foreground_scope_host",
    "_is_supervised_gateway_process",
    "is_worker_process",
    "is_supervised_gateway_launch",
    "get_running_pid",
)
D6_PROPERTIES = ("MemoryHigh=5G", "MemorySwapMax=2G", "TasksMax=4096", "CPUQuota=800%", "CPUWeight=50")


# --------------------------------------------------------------------------- stub systemd-run
@dataclass
class Call:
    argv: list[str]
    env: dict[str, str]

    @property
    def properties(self) -> list[str]:
        return [self.argv[i + 1] for i, a in enumerate(self.argv[:-1]) if a == "--property"]

    @property
    def unit(self) -> str:
        return self.argv[self.argv.index("--unit") + 1]

    @property
    def command(self) -> list[str]:
        return self.argv[self.argv.index("--") + 1:]


class SystemdStub:
    """A ``systemd-run`` on PATH that records argv and environment, then behaves per ``mode``."""

    SCRIPTS = {
        # run the wrapped command in place, as the real --scope mode does
        "passthrough": 'exec "${args[@]:$((sep+1))}"',
        "unplaced": 'exec "${args[@]:$((sep+1))}"',
        # the bus is gone: the wrapper fails before the command could start
        "no_bus": 'echo "Failed to connect to bus: No such file or directory" >&2; exit 1',
        "lost_create_reply": 'echo "Failed to start transient scope unit: Connection timed out" >&2; exit 1',
        "unit_exists": 'echo "Failed to start transient scope unit: Unit already exists" >&2; exit 1',
        "multiline_create_error": ('echo "Failed to start transient scope unit: Connection dropped" >&2; '
                                   'echo "Failed to connect to bus: No such file or directory" >&2; exit 1'),
        # As with systemd-run, exec in place; the payload itself reports failure.
        "ran_then_fail": 'args[-1]="trap \'exit 1\' EXIT; ${args[-1]}"; exec "${args[@]:$((sep+1))}"',
        "ran_then_bus_error": ('"${args[@]:$((sep+1))}"; '
                               'echo "Failed to connect to bus: No such file or directory" >&2; exit 1'),
    }

    def __init__(self, root: Path, mode: str = "passthrough", *, real: str | None = None):
        self.root = root
        self.records = root / "calls"
        self.records.mkdir(parents=True)
        self.extra = root / "extra_properties"
        body = self.SCRIPTS[mode]
        if real is not None:  # inject extra --property values, then hand over to the real systemd-run
            body = (
                f'pre=("${{args[@]:0:$sep}}"); '
                f"if [ -f '{self.extra}' ]; then "
                f"while IFS= read -r p; do pre+=(--property \"$p\"); done < '{self.extra}'; fi; "
                f"exec '{real}' \"${{pre[@]}}\" -- \"${{args[@]:$((sep+1))}}\""
            )
        script = textwrap.dedent(f"""\
            #!/usr/bin/env bash
            printf '%s\\0' "$@" > '{self.records}'/$$.argv
            env -0 > '{self.records}'/$$.env
            args=("$@"); sep=0
            for i in "${{!args[@]}}"; do [ "${{args[$i]}}" = "--" ] && {{ sep=$i; break; }}; done
            {body}
        """)
        path = root / "bin" / "systemd-run"
        path.parent.mkdir()
        path.write_text(script)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        self.bindir = path.parent

    def set_extra_properties(self, *props: str) -> None:
        self.extra.write_text("\n".join(props) + "\n")

    def calls(self) -> list[Call]:
        out = []
        for argv_file in sorted(self.records.glob("*.argv"), key=lambda p: p.stat().st_mtime_ns):
            argv = argv_file.read_bytes().decode().split("\0")[:-1]
            env_raw = argv_file.with_suffix(".env").read_bytes().decode(errors="replace").split("\0")
            env = dict(item.split("=", 1) for item in env_raw if "=" in item)
            out.append(Call(argv, env))
        return out


@pytest.fixture()
def stub_factory(tmp_path, monkeypatch):
    """``stub_factory(mode)`` puts a recording ``systemd-run`` first on PATH and reports the probe as up."""
    made = []

    def make(mode: str = "passthrough", *, real: str | None = None) -> SystemdStub:
        stub = SystemdStub(tmp_path / f"stub{len(made)}", mode, real=real)
        made.append(stub)
        monkeypatch.setenv("PATH", f"{stub.bindir}{os.pathsep}{os.environ['PATH']}")
        # Host capability, not identity: seed the probe's own cache instead of replacing the probe.
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", True)
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_PROBED_AT", time.monotonic())
        if real is None and mode in ("passthrough", "ran_then_fail"):
            monkeypatch.setattr(fs, "_pin_original", _fixture_pin)
        return stub

    return make


@pytest.fixture()
def stub(stub_factory) -> SystemdStub:
    return stub_factory("passthrough")


class _FixturePin:
    def close(self):
        pass


def _fixture_pin(*_):
    # These tests exercise behavior after admission. The pin-specific tests
    # exercise actual directory/control opens with a simulated manager/kernel.
    return _FixturePin()


@pytest.fixture()
def real_systemd_stub(stub_factory, monkeypatch):
    """Recording shim in front of the REAL systemd-run (extra properties injectable); skips without a user bus."""
    real = shutil.which("systemd-run")
    if real is None:
        pytest.skip("systemd-run not installed")
    made = stub_factory("passthrough", real=real)
    monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)  # probe for real, through the shim
    if not pr._systemd_run_user_scope_available():
        pytest.skip("no systemd user bus for transient scopes")
    shutil.rmtree(made.records)  # the probe's own call is not part of any test
    made.records.mkdir()
    return made


# --------------------------------------------------------------------------- host shapes
_SUPERVISED_ENV = {"_HERMES_GATEWAY": "1", "HERMES_SUPERVISED_CHILD": "1", "INVOCATION_ID": "x"}


def _set_supervised_env(monkeypatch) -> None:
    for key, value in _SUPERVISED_ENV.items():
        monkeypatch.setenv(key, value)


def _clear_supervised_env(monkeypatch) -> None:
    for key in (*_SUPERVISED_ENV, "HERMES_GATEWAY_SUPERVISOR"):
        monkeypatch.delenv(key, raising=False)


_GATEWAY_PROLOGUE = '''
import json, os, sys, time
sys.path.insert(0, {root!r})
from gateway import status
if not status.acquire_gateway_runtime_lock():
    sys.exit("could not acquire gateway.lock")
status.write_pid_file()
from tools import process_registry as pr
pr._SYSTEMD_SCOPE_AVAILABLE = True
pr._SYSTEMD_SCOPE_PROBED_AT = time.monotonic()
'''


def _gateway_script(tmp_path: Path, body: str) -> Path:
    """A script that IS a live gateway to the real identity code: it runs as ``<script> gateway run`` and holds
    the real runtime lock and gateway.pid, as the existing strict-identity tests do."""
    script = tmp_path / "gwbin" / "hermes"
    script.parent.mkdir(exist_ok=True)
    script.write_text(_GATEWAY_PROLOGUE.format(root=str(REPO_ROOT)) + textwrap.dedent(body))
    return script


def _gateway_env() -> dict[str, str]:
    return {**os.environ, **_SUPERVISED_ENV}


def run_as_gateway(tmp_path: Path, body: str) -> dict:
    """Run ``body`` inside a live-gateway-shaped process; its last stdout line is a JSON result."""
    done = subprocess.run([sys.executable, str(_gateway_script(tmp_path, body)), "gateway", "run"],
                          env=_gateway_env(), capture_output=True, text=True, timeout=180, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture()
def foreign_gateway(tmp_path):
    """A live gateway in ANOTHER process (the worker's view of it), through the real lock and pid writers."""
    body = "print('ready', flush=True)\nsys.stdin.read()\n"
    proc = subprocess.Popen([sys.executable, str(_gateway_script(tmp_path, body)), "gateway", "run"],
                            env=_gateway_env(), text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdout.readline().strip() == "ready", proc.stderr.read()
        yield proc
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)


@pytest.fixture()
def worker_host(monkeypatch, foreign_gateway):
    """Managed worker: supervised env, the gateway is another pid, this interpreter adopted (flag set)."""
    _set_supervised_env(monkeypatch)
    monkeypatch.setattr("agent.runtime_session_store._worker_process", True)
    return foreign_gateway


@pytest.fixture()
def cli_host(monkeypatch):
    _clear_supervised_env(monkeypatch)
    monkeypatch.setattr("agent.runtime_session_store._worker_process", False)


@pytest.fixture()
def descendant_cli_host(monkeypatch, foreign_gateway):
    """A ``hermes`` CLI started inside a build: it inherited the worker's env but never adopted."""
    _set_supervised_env(monkeypatch)
    monkeypatch.setattr("agent.runtime_session_store._worker_process", False)


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=60, check=False,
                          env=pr.systemd_user_bus_env())


@pytest.fixture()
def fg_config(tmp_path):
    """``fg_config(enabled=True, gradle_daemon=None)`` writes the profile config.yaml the real reader sees."""
    from hermes_constants import get_hermes_home

    def write(enabled: bool = True, gradle_daemon: str | None = None, raw: str | None = None, **extra) -> None:
        section: dict = {"enabled": enabled, **extra}
        if gradle_daemon is not None:
            section["gradle_daemon"] = gradle_daemon
        lines = ["terminal:", "  foreground_scope:"]
        lines += [f"    {key}: {json.dumps(value)}" for key, value in section.items()]  # JSON scalars are YAML
        (get_hermes_home() / "config.yaml").write_text(raw if raw is not None else "\n".join(lines) + "\n")

    return write


@pytest.fixture()
def make_env(monkeypatch, tmp_path):
    envs = []

    def make(**kwargs) -> LocalEnvironment:
        kwargs.setdefault("cwd", str(tmp_path))
        kwargs.setdefault("timeout", 60)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(LocalEnvironment, "init_session", lambda self, *a, **k: None)
            env = LocalEnvironment(**kwargs)
        envs.append(env)
        return env

    yield make
    for env in envs:
        env.cleanup()


# --------------------------------------------------------------------------- T0: the gate, unpatched
@pytest.mark.platforms("linux")
class TestGate:
    def test_worker_shape_passes_gate(self, worker_host):
        assert fs.foreground_scope_host() is True

    def test_gateway_shape_passes_gate(self, tmp_path):
        out = run_as_gateway(tmp_path, """
            import tools.foreground_scope as fs
            print(json.dumps({"host": fs.foreground_scope_host(), "r0": pr._is_supervised_gateway_process()}))
        """)
        assert out == {"host": True, "r0": True}

    def test_cli_shapes_fail_gate(self, cli_host, monkeypatch, foreign_gateway):
        assert fs.foreground_scope_host() is False
        _set_supervised_env(monkeypatch)  # descendant CLI: inherited env, another pid owns the gateway, no adopt
        assert fs.foreground_scope_host() is False

    def test_r0_gate_is_false_in_worker_shape(self, worker_host):
        """Pins the premise of the gate design: the gateway-identity test alone is false inside a worker."""
        assert pr._is_supervised_gateway_process() is False


def _gate_patch_violations(source: str) -> list[str]:
    """Calls that patch an identity gate: ``monkeypatch.setattr``/``patch``/``patch.object``/``patch.multiple``/
    ``patch.dict`` or ``mock.patch...`` naming a gate, by string target, attribute or keyword."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        callee = ast.unparse(node.func)
        if not (callee.endswith("setattr") or "patch" in callee):
            continue
        text = ast.unparse(node)
        found.extend(f"{callee}: {name}" for name in GATE_NAMES if name in text)
    return found


class TestGatePatchGuard:
    def test_gate_patch_guard(self):
        violations = _gate_patch_violations(Path(__file__).read_text())
        assert not violations, f"tests must build host shapes, not patch the gate: {violations}"

    @pytest.mark.parametrize("snippet", [
        'monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)',
        'monkeypatch.setattr(pr, "is_worker_process", lambda: True)',
        'patch("tools.foreground_scope.foreground_scope_host", return_value=True)',
        'patch.object(fs, "foreground_scope_host", return_value=True)',
        'with mock.patch.object(restart, "is_supervised_gateway_launch"): pass',
        'monkeypatch.setattr("gateway.status.get_running_pid", lambda **k: 1)',
        'patch.multiple(pr, is_worker_process=lambda: True)',
    ])
    def test_guard_catches_each_patch_form(self, snippet):
        assert _gate_patch_violations(snippet)

    @pytest.mark.parametrize("snippet", [
        'monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", True)',
        'monkeypatch.setattr("agent.runtime_session_store._worker_process", True)',
        'assert fs.foreground_scope_host() is True',
    ])
    def test_guard_allows_host_shape_and_capability_seeding(self, snippet):
        assert not _gate_patch_violations(snippet)


# --------------------------------------------------------------------------- T1: tracer
@pytest.mark.platforms("linux")
class TestScopedCommand:
    def _assert_scoped(self, stub, result):
        assert result["returncode"] == 0, result
        assert "hi" in result["output"]
        (call,) = stub.calls()
        assert call.argv[:3] == ["--user", "--scope", "--quiet"]  # recorded argv excludes the binary itself
        assert call.unit.startswith("hermes-worker-fg-")
        assert "--collect" in call.argv
        assert call.properties == ["MemoryAccounting=yes"]
        assert call.command[-2] == "-c", call.command  # <bash> [-l] -c <wrapped>

    def test_worker_command_is_scoped(self, worker_host, fg_config, stub, make_env):
        fg_config()
        self._assert_scoped(stub, make_env().execute("echo hi", resource_scope=True))

    def test_gateway_command_is_scoped(self, fg_config, stub, tmp_path):
        fg_config()
        result = run_as_gateway(tmp_path, f"""
            from tests.tools.test_local_foreground_scope import _fixture_pin
            from tools import foreground_scope as fs
            from tools.environments.local import LocalEnvironment
            fs._pin_original = _fixture_pin
            LocalEnvironment.init_session = lambda self, *a, **k: None
            env = LocalEnvironment(cwd={str(tmp_path)!r}, timeout=60)
            r = env.execute("echo hi", resource_scope=True)
            print(json.dumps({{"returncode": r["returncode"], "output": r["output"]}}))
        """)
        self._assert_scoped(stub, result)

    def test_cli_commands_are_not_scoped(self, cli_host, fg_config, stub, make_env, monkeypatch, foreign_gateway):
        fg_config()
        env = make_env()
        assert env.execute("echo hi", resource_scope=True)["returncode"] == 0
        _set_supervised_env(monkeypatch)  # descendant CLI
        assert env.execute("echo hi", resource_scope=True)["returncode"] == 0
        assert stub.calls() == []

    def test_unit_name_carries_the_card_prefix(self, worker_host, fg_config, stub, make_env, monkeypatch):
        fg_config()
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_ad1ebdaa")
        make_env().execute("echo hi", resource_scope=True)
        make_env().execute("echo hi", resource_scope=True)
        first, second = stub.calls()
        assert re.fullmatch(r"hermes-worker-fg-t_ad1ebd-[0-9a-f]{8}", first.unit), first.unit
        assert first.unit != second.unit

    def test_real_scope_cgroup(self, worker_host, fg_config, real_systemd_stub, make_env):
        fg_config()
        result = make_env().execute("cat /proc/self/cgroup", resource_scope=True)
        assert "hermes-worker-fg-" in result["output"], result


# --------------------------------------------------------------------------- T2: only the terminal tool opts in
@pytest.mark.platforms("linux")
class TestOnlyTheTerminalToolOptsIn:
    @pytest.fixture()
    def local_terminal(self, tmp_path, monkeypatch):
        from tools.terminal_tool_lifecycle import cleanup_vm
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        task = f"fg-scope-{os.getpid()}"
        yield task
        cleanup_vm(task)

    def test_terminal_tool_foreground_is_scoped(self, worker_host, fg_config, stub, local_terminal):
        from tools.terminal_tool import terminal_tool
        fg_config()
        out = json.loads(terminal_tool(command="echo hi", task_id=local_terminal))
        assert out["exit_code"] == 0, out
        assert [c.unit.startswith("hermes-worker-fg-") for c in stub.calls()] == [True]

    def test_control_plane_command_is_not_scoped(self, worker_host, fg_config, stub, local_terminal):
        from tools.terminal_tool import terminal_tool
        fg_config()
        out = json.loads(terminal_tool(command="echo hi", task_id=local_terminal, _host_local=True))
        assert out["exit_code"] == 0, out
        assert stub.calls() == []

    def test_session_bootstrap_probes_and_plain_execute_are_not_scoped(self, worker_host, fg_config, stub, tmp_path):
        fg_config()
        env = LocalEnvironment(cwd=str(tmp_path), timeout=60)  # real init_session snapshot bootstrap
        try:
            assert env.execute("echo hi")["returncode"] == 0  # file operations, RPC and log reads
            assert env._run_bash("sudo -n true", timeout=30).wait(timeout=30) is not None  # capability probe
        finally:
            env.cleanup()
        assert stub.calls() == []

    def test_background_command_is_left_alone(self, worker_host, fg_config, stub, tmp_path):
        """Scoping background commands started from a worker is a separate follow-up."""
        fg_config()
        session = pr.process_registry.spawn_local("sleep 30", cwd=str(tmp_path))
        try:
            assert not [c for c in stub.calls() if c.unit.startswith("hermes-worker-fg-")]
        finally:
            pr.process_registry.kill_process(session.id)

    def test_backend_without_scope_support_never_receives_the_keyword(self, tmp_path):
        from tools.environments.base import BaseEnvironment

        class OldSignature(BaseEnvironment):
            seen: list[str] = []

            def __init__(self):
                super().__init__(cwd=str(tmp_path), timeout=30)

            def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
                OldSignature.seen.append(cmd_string)
                return subprocess.Popen(["bash", "-c", cmd_string], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, text=True)

            def cleanup(self):
                pass

            def init_session(self, *a, **k):
                self._snapshot_ready = False

        env = OldSignature()
        assert env.execute("echo hi", resource_scope=True)["returncode"] == 0
        assert BaseEnvironment.supports_resource_scope is False
        assert LocalEnvironment.supports_resource_scope is True


# --------------------------------------------------------------------------- T3: property set and limits
@pytest.mark.platforms("linux")
class TestPropertySet:
    def test_argv_property_set_is_exactly_accounting(self, worker_host, fg_config, stub, make_env):
        fg_config()
        make_env().execute("echo hi", resource_scope=True)
        (call,) = stub.calls()
        assert call.properties == ["MemoryAccounting=yes"]
        for limit in ("MemoryMax", "MemoryHigh", "CPUQuota", "TasksMax", "MemorySwapMax"):
            assert not any(limit in element for element in call.argv), (limit, call.argv)

    def test_no_memory_max_code_path(self, worker_host, stub, fg_config, make_env):
        """A leftover ``memory_max_mb`` key is ignored: no MemoryMax reaches the scope, however configured."""
        fg_config(memory_max_mb=4096)
        make_env().execute("echo hi", resource_scope=True)
        (call,) = stub.calls()
        assert not any("MemoryMax" in element for element in call.argv), call.argv
        assert not hasattr(fs, "_foreground_memory_max_bytes")

    def test_real_scope_property_set(self, worker_host, fg_config, real_systemd_stub, make_env):
        fg_config()
        real_systemd_stub.set_extra_properties(*D6_PROPERTIES)
        env = make_env()
        out = env.execute(
            "systemctl --user show \"$(basename $(cut -d: -f3 /proc/self/cgroup))\" "
            "-p MemoryHigh -p MemorySwapMax -p MemoryMax -p TasksMax -p CPUQuotaPerSecUSec -p CPUWeight",
            resource_scope=True)["output"]
        shown = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
        assert shown["MemoryHigh"] == str(5 * 1024 ** 3), out
        assert shown["MemorySwapMax"] == str(2 * 1024 ** 3), out
        assert shown["TasksMax"] == "4096", out
        assert shown["CPUQuotaPerSecUSec"] == "8s", out
        assert shown["CPUWeight"] == "50", out
        assert shown["MemoryMax"] == "infinity", out

    def test_real_tasksmax_fails_the_fork_not_waits(self, worker_host, fg_config, real_systemd_stub, make_env):
        fg_config()
        real_systemd_stub.set_extra_properties("TasksMax=20")
        result = make_env(timeout=120).execute(
            "for i in $(seq 60); do sleep 30 & done; wait", resource_scope=True, timeout=120)
        assert "Resource temporarily unavailable" in result["output"], result
        assert result["returncode"] != 0, result

    def test_real_cpuquota_throttles_not_fails(self, worker_host, fg_config, real_systemd_stub, make_env):
        fg_config()
        real_systemd_stub.set_extra_properties("CPUQuota=100%")
        burn = "timeout 2 sh -c 'while :; do :; done'"
        result = make_env().execute(
            f"TIMEFORMAT=%U; time {{ for i in 1 2 3 4; do {burn} & done; wait; }}; echo done", resource_scope=True)
        user_seconds = float(next(line for line in result["output"].splitlines() if re.fullmatch(r"\d+\.\d+", line)))
        assert user_seconds <= 2.5, result  # four loops would burn ~8 s unthrottled


# --------------------------------------------------------------------------- T4: kill, timeout, yield
@pytest.mark.platforms("linux")
class TestScopeLifecycle:
    @pytest.mark.parametrize("redirect", ["", " >&2"])
    def test_payload_bus_diagnostic_does_not_trigger_fallback(self, redirect, worker_host, fg_config,
                                                             stub, make_env, tmp_path):
        fg_config()
        counter = tmp_path / "counter"
        diagnostic = "Failed to connect to bus: No such file or directory"
        result = make_env().execute(
            f"echo ran >> {counter}; echo '{diagnostic}'{redirect}; exit 1", resource_scope=True)
        assert counter.read_text() == "ran\n"
        assert result["returncode"] == 1
        assert result["output"].strip() == diagnostic
        assert len(stub.calls()) == 1

    def test_launcher_bus_diagnostic_after_untrusted_helper_does_not_replay(self, worker_host, fg_config,
                                                                          stub_factory, make_env, tmp_path):
        fg_config()
        stub = stub_factory("ran_then_bus_error")
        counter = tmp_path / "counter"
        result = make_env().execute(f"echo ran >> {counter}", resource_scope=True)
        # The helper has the wrong sender PID; neither admission nor replay is allowed.
        assert not counter.exists()
        assert result["returncode"] == 1
        assert len(stub.calls()) == 1

    def test_post_job_helper_without_scope_membership_cannot_run_payload(
        self, worker_host, fg_config, stub_factory, make_env, monkeypatch, tmp_path,
    ):
        fg_config()
        stub = stub_factory("unplaced")
        # The pass-through launcher execs the helper with the right PID but
        # never creates a cgroup or puts the helper into an original object.
        stopped = self._spy_stop(monkeypatch)
        marker = tmp_path / "payload"
        result = make_env().execute(f"touch {marker}", resource_scope=True, timeout=5)
        assert not marker.exists(), result
        assert result["returncode"] == 125
        assert len(stub.calls()) == 1
        assert stopped == []

    @pytest.mark.parametrize("cancel", [False, True], ids=["release-once", "cancel-before-release"])
    def test_held_payload_waits_for_original_scope_verification(
        self, cancel, worker_host, fg_config, stub, make_env, monkeypatch, tmp_path,
    ):
        fg_config()
        marker = tmp_path / "payload"
        verifying, finish_verification = Event(), Event()
        stopped = self._spy_stop(monkeypatch)
        held = []

        class FixtureOriginal:
            def close(self):
                pass

        def pin_original(proc, unit, invocation_id):
            held.append(proc)
            verifying.set()
            assert finish_verification.wait(5), "verification barrier was not released"
            return FixtureOriginal()

        # Substitute only the manager/kernel object verification: run the actual
        # launcher, helper, admission channel, and terminal waiter.
        monkeypatch.setattr(fs, "_pin_original", pin_original)
        env = make_env()
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(copy_context().run, env.execute, f"echo ran >> {marker}",
                                 resource_scope=True, timeout=8)
            try:
                assert verifying.wait(5), (marker.exists(), result.result(timeout=10))
                assert not marker.exists()
                if cancel:
                    # Force the cooperative PGID route to refuse. A rejected
                    # group signal must not substitute for closing admission.
                    monkeypatch.setattr(local_env, "_leader_is_ours", lambda *_: False)
                    env._kill_process(held[0])
            finally:
                finish_verification.set()
            reply = result.result(timeout=8)
        if cancel:
            assert not marker.exists(), reply
            assert reply["returncode"] == 125
        else:
            assert marker.read_text() == "ran\n", reply
            assert reply["returncode"] == 0
        assert len(stub.calls()) == 1
        assert stopped == []

    @pytest.mark.parametrize("change", [None, "invocation", "membership", "generation", "control"])
    def test_held_launch_pins_controls_only_for_the_original_placement(
        self, change, worker_host, fg_config, stub_factory, make_env, monkeypatch, tmp_path,
    ):
        fg_config()
        invocation = "12" * 16
        monkeypatch.setitem(SystemdStub.SCRIPTS, "placed", f'export INVOCATION_ID={invocation}; '
                            + SystemdStub.SCRIPTS["passthrough"])
        stub_factory("placed")
        mount = tmp_path / "cgroup"
        mount.mkdir()
        placements = []
        marker = tmp_path / "payload"

        def placement(pid, unit, env):
            path = f"/app.slice/{unit}"
            cg = mount / path.lstrip("/")
            if not placements:
                cg.mkdir(parents=True)
                (cg / "cgroup.type").write_text("domain\n")
                (cg / "cgroup.procs").write_text(f"{pid}\n")
                for control in ("events", "freeze", "kill"):
                    if (control, change) != ("kill", "control"):
                        (cg / f"cgroup.{control}").touch()
            elif change == "membership":
                (cg / "cgroup.procs").write_text("")
            elif change == "generation":
                cg.rename(cg.with_suffix(".old"))
                shutil.copytree(cg.with_suffix(".old"), cg)
            placements.append(path)
            assert not marker.exists(), "payload ran before the pin was verified"
            current = "34" * 16 if change == "invocation" and len(placements) > 1 else invocation
            original_id = cg.stat().st_ino if change != "generation" or len(placements) == 1 else cg.with_suffix(".old").stat().st_ino
            return ("bus-id", ":1.2", path, current, original_id)

        monkeypatch.setattr(fs, "_read_placement", placement, raising=False)
        monkeypatch.setattr(fs, "_cgroup_mount", lambda: (17, str(mount)), raising=False)
        monkeypatch.setattr(fs, "_unified_membership", lambda pid: placements[-1], raising=False)
        # Fake the kernfs generation at the syscall boundary; directory/control
        # opens (including the replacement) and the held-launch channel are real.
        monkeypatch.setattr(fs, "_kernel_handle", lambda fd: (17, os.fstat(fd).st_ino), raising=False)
        monkeypatch.setattr(fs, "_fd_mount_id", lambda fd: 17, raising=False)
        reply = make_env().execute(f"echo ran >> {marker}", resource_scope=True, timeout=10)
        if change is None:
            assert reply["returncode"] == 0, reply
            assert marker.read_text() == "ran\n"
        else:
            assert reply["returncode"] == 125, reply
            assert not marker.exists()
        assert len(placements) == (1 if change == "control" else 2)

    @pytest.mark.parametrize("change", [None, "bus", "manager", "wrong-id-type"])
    def test_placement_requires_one_bus_epoch_and_typed_generation(self, change, monkeypatch):
        unit = "hermes-worker-fg-test.scope"
        obj = "/org/freedesktop/systemd1/unit/hermes_2dworker_2dfg_2dtest_2escope"
        systemd = "org.freedesktop.systemd1"
        bus_id, owner = "a" * 32, ":1.7"
        path, generation = f"/app.slice/{unit}", 4294967373
        invocation = [18] * 16
        read_scope = False

        def reply(args, **kwargs):
            nonlocal read_scope
            destination, target, interface, member, *parameters = args[args.index("call") + 1:]
            if member == "GetId":
                signature, value = "s", "b" * 32 if read_scope and change == "bus" else bus_id
            elif member == "GetNameOwner":
                signature, value = "s", ":1.8" if read_scope and change == "manager" else owner
            else:
                assert destination == owner
                if member == "GetUnitByPID":
                    assert parameters == ["u", "12345"]
                    signature, value = "o", obj
                elif parameters == ["s", f"{systemd}.Unit"]:
                    assert target == obj and member == "GetAll"
                    signature, value = "a{sv}", {
                        "Id": {"type": "s", "data": unit},
                        "ActiveState": {"type": "s", "data": "active"},
                        "Transient": {"type": "b", "data": True},
                        "InvocationID": {"type": "ay", "data": invocation},
                    }
                else:
                    assert target == obj and member == "GetAll" and parameters == ["s", f"{systemd}.Scope"]
                    read_scope = True
                    signature, value = "a{sv}", {
                        "ControlGroup": {"type": "s", "data": path},
                        "ControlGroupId": {"type": "x" if change == "wrong-id-type" else "t", "data": generation},
                    }
            return subprocess.CompletedProcess(args, 0, json.dumps({"type": signature, "data": [value]}).encode())

        monkeypatch.setattr(fs, "locate_command", lambda _: type("Located", (), {"command": ("busctl",)})())
        monkeypatch.setattr(fs.subprocess, "run", reply)
        if change is None:
            assert fs._read_placement(12345, unit, {}) == (bus_id, owner, path, "12" * 16, generation)
        else:
            with pytest.raises(ValueError):
                fs._read_placement(12345, unit, {})

    @pytest.mark.parametrize("mode", ["lost_create_reply", "unit_exists", "multiline_create_error"])
    def test_create_call_failure_does_not_run_payload_unscoped(self, mode, worker_host, fg_config,
                                                             stub_factory, make_env, monkeypatch, tmp_path):
        fg_config()
        stub = stub_factory(mode)
        stopped = self._spy_stop(monkeypatch)
        marker = tmp_path / "payload"
        result = make_env().execute(f"touch {marker}", resource_scope=True)
        assert not marker.exists(), result
        assert result["returncode"] == 1
        assert len(stub.calls()) == 1
        assert stopped == []

    def test_unresolved_creation_never_stops_a_same_name_unit(self, worker_host, fg_config, stub_factory,
                                                               make_env, monkeypatch, tmp_path):
        fg_config()
        stub = stub_factory("no_bus")
        # A request with no reply: the wrapper stays alive without executing its payload.
        (stub.bindir / "systemd-run").write_text("#!/usr/bin/env bash\nexec sleep 30\n")
        stopped = self._spy_stop(monkeypatch)
        marker = tmp_path / "payload"
        result = make_env().execute(f"touch {marker}", resource_scope=True, timeout=1)
        assert result["returncode"] == 124
        assert not marker.exists()
        assert stopped == []

    def _spy_stop(self, monkeypatch):
        stopped: list[str] = []
        monkeypatch.setattr(pr, "_stop_systemd_unit", lambda unit: stopped.append(unit) or True)
        return stopped

    def test_timeout_stops_the_whole_scope(self, worker_host, fg_config, real_systemd_stub, make_env, monkeypatch):
        fg_config()
        stopped = self._spy_stop(monkeypatch)
        result = make_env().execute("sleep 30 & setsid sleep 30 & wait", resource_scope=True, timeout=1)
        assert result["returncode"] == 124, result
        (call,) = real_systemd_stub.calls()
        assert stopped == []
        state = _systemctl("is-active", f"{call.unit}.scope")
        assert state.stdout.strip() == "inactive", state

    @pytest.mark.live_system_guard_bypass  # stops its own hermes-worker-fg scope, named at run time
    def test_normal_exit_leaves_the_scope_to_systemd(self, worker_host, fg_config, real_systemd_stub, make_env,
                                                     monkeypatch):
        fg_config()
        stopped = self._spy_stop(monkeypatch)
        result = make_env().execute("setsid sleep 30 >/dev/null 2>&1 & echo started", resource_scope=True)
        (call,) = real_systemd_stub.calls()
        try:
            assert result["returncode"] == 0 and "started" in result["output"], result
            assert stopped == []
            state = _systemctl("is-active", f"{call.unit}.scope")
            assert state.stdout.strip() == "active", state
        finally:
            _systemctl("stop", f"{call.unit}.scope")

    def test_yielded_command_never_stops_a_scope_by_name(self, worker_host, fg_config, stub, make_env, monkeypatch):
        from tools.terminal_tool_background import yield_to_background_handler
        fg_config()
        stopped = self._spy_stop(monkeypatch)
        command = "echo ready; sleep 30"
        proc = make_env()._run_bash(command, resource_scope=True)
        try:
            assert proc.stdout.readline().strip() == "ready"
            handler = yield_to_background_handler(
                command=command, env_type="local", cwd=None, effective_task_id="fg-yield",
                task_id="fg-yield", session_key="")
            handed = handler(proc, "")
            pr.process_registry.kill_process(handed["yielded_session_id"])
            assert len(stub.calls()) == 1
            assert stopped == []
        finally:
            proc.kill()
            proc.wait(timeout=5)


# --------------------------------------------------------------------------- T4g: Gradle daemon policy
@pytest.mark.platforms("linux")
class TestGradleDaemonPolicy:
    def _gradle_opts(self, stub):
        (call,) = stub.calls()
        return call.env.get("GRADLE_OPTS")

    def test_default_adds_no_daemon(self, worker_host, fg_config, stub, make_env, monkeypatch):
        fg_config()
        monkeypatch.delenv("GRADLE_OPTS", raising=False)
        make_env().execute("true", resource_scope=True)
        assert self._gradle_opts(stub).split()[-1] == "-Dorg.gradle.daemon=false"

    def test_existing_value_is_extended_not_replaced(self, worker_host, fg_config, stub, make_env, monkeypatch):
        fg_config()
        monkeypatch.setenv("GRADLE_OPTS", "-Xmx1g")
        make_env().execute("true", resource_scope=True)
        assert self._gradle_opts(stub) == "-Xmx1g -Dorg.gradle.daemon=false"

    def test_user_choice_is_left_alone(self, worker_host, fg_config, stub, make_env, monkeypatch):
        fg_config()
        monkeypatch.setenv("GRADLE_OPTS", "-Dorg.gradle.daemon=true -Xmx1g")
        make_env().execute("true", resource_scope=True)
        assert self._gradle_opts(stub) == "-Dorg.gradle.daemon=true -Xmx1g"

    def test_shared_mode_leaves_env_alone(self, worker_host, fg_config, stub, make_env, monkeypatch):
        fg_config(gradle_daemon="shared")
        monkeypatch.setenv("GRADLE_OPTS", "-Xmx1g")
        make_env().execute("true", resource_scope=True)
        assert self._gradle_opts(stub) == "-Xmx1g"

    def test_unscoped_commands_are_untouched(self, cli_host, fg_config, stub, make_env, monkeypatch, tmp_path):
        fg_config()
        monkeypatch.setenv("GRADLE_OPTS", "-Xmx1g")
        result = make_env().execute("printenv GRADLE_OPTS", resource_scope=True)
        assert result["output"].strip() == "-Xmx1g"
        assert stub.calls() == []

    def test_invalid_value_means_per_command_with_one_log_line(self, worker_host, fg_config, stub, make_env,
                                                               monkeypatch, caplog):
        fg_config(gradle_daemon="bogus")
        monkeypatch.delenv("GRADLE_OPTS", raising=False)
        env = make_env()
        with caplog.at_level("INFO", logger="tools.foreground_scope"):
            env.execute("true", resource_scope=True)
            env.execute("true", resource_scope=True)
        assert [c.env["GRADLE_OPTS"].split()[-1] for c in stub.calls()] == ["-Dorg.gradle.daemon=false"] * 2
        assert sum("gradle_daemon" in r.getMessage() for r in caplog.records) == 1


# --------------------------------------------------------------------------- T5: byte-identical fallback
class TestFallbackIsByteIdentical:
    @pytest.fixture()
    def popen_spy(self, monkeypatch):
        seen: list[dict] = []
        real = subprocess.Popen

        def spy(args, **kwargs):
            seen.append({"args": list(args), "env": dict(kwargs.get("env") or {}), "cwd": kwargs.get("cwd")})
            return real(args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", spy)
        return seen

    @staticmethod
    def _command_call(popen_spy):
        """The spawn that carries the test's command (other spawns, probes or cleanup, are not under test)."""
        (call,) = [c for c in popen_spy if "echo hi" in c["args"][-1]][-1:]
        popen_spy.clear()
        return call

    def _golden(self, make_env, popen_spy):
        """What today's ``_run_bash`` spawns for the same command: no resource_scope keyword at all."""
        make_env().execute("echo hi")
        return self._command_call(popen_spy)

    def _assert_unchanged(self, make_env, popen_spy, stub):
        golden = self._golden(make_env, popen_spy)
        assert make_env().execute("echo hi", resource_scope=True)["returncode"] == 0
        scoped_call = self._command_call(popen_spy)
        assert scoped_call["args"][:2] == golden["args"][:2]
        assert scoped_call["env"] == golden["env"]
        assert scoped_call["cwd"] == golden["cwd"]
        assert stub.calls() == []

    @pytest.mark.platforms("linux")
    def test_cli_shape(self, cli_host, fg_config, stub, make_env, popen_spy):
        fg_config()
        self._assert_unchanged(make_env, popen_spy, stub)

    @pytest.mark.platforms("linux")
    def test_descendant_cli_shape(self, descendant_cli_host, fg_config, stub, make_env, popen_spy):
        fg_config()
        self._assert_unchanged(make_env, popen_spy, stub)

    @pytest.mark.platforms("linux")
    def test_flag_off(self, worker_host, fg_config, stub, make_env, popen_spy):
        fg_config(enabled=False)
        self._assert_unchanged(make_env, popen_spy, stub)

    @pytest.mark.platforms("linux")
    def test_config_unreadable_means_off(self, worker_host, fg_config, stub, make_env, popen_spy):
        fg_config(raw="terminal: [unbalanced\n  : :")
        self._assert_unchanged(make_env, popen_spy, stub)

    @pytest.mark.platforms("linux")
    def test_probe_false_warns_once_per_process(self, worker_host, fg_config, stub, make_env, popen_spy,
                                                monkeypatch, caplog):
        fg_config()
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", False)
        monkeypatch.setattr(pr, "_scope_degraded_warned", False)
        with caplog.at_level("WARNING"):
            self._assert_unchanged(make_env, popen_spy, stub)
            make_env().execute("echo hi", resource_scope=True)
        assert sum("foreground" in r.getMessage() for r in caplog.records) == 1

    @pytest.mark.platforms("linux")
    def test_systemd_run_missing_from_path(self, worker_host, fg_config, stub, make_env, popen_spy, monkeypatch):
        fg_config()
        real_which = shutil.which
        monkeypatch.setattr(shutil, "which", lambda name, *a, **k: None if name == "systemd-run" else real_which(name, *a, **k))
        self._assert_unchanged(make_env, popen_spy, stub)

    @pytest.mark.platforms("macos")
    def test_macos_never_probes(self, fg_config, make_env, popen_spy, monkeypatch):
        fg_config()
        probes = []
        monkeypatch.setattr(pr, "_SYSTEMD_SCOPE_AVAILABLE", None)
        real_run = subprocess.run
        monkeypatch.setattr(subprocess, "run", lambda argv, *a, **k: probes.append(argv) or real_run(argv, *a, **k))
        assert make_env().execute("echo hi", resource_scope=True)["returncode"] == 0
        assert not [p for p in probes if "systemd-run" in str(p)]

    @pytest.mark.platforms("linux")
    def test_bus_lost_after_probe_runs_once_unscoped(self, worker_host, fg_config, stub_factory, make_env, monkeypatch,
                                                     tmp_path):
        fg_config()
        stub = stub_factory("no_bus")
        lost = []
        real = pr.scoped_spawn_lost_user_bus
        monkeypatch.setattr(pr, "scoped_spawn_lost_user_bus", lambda env: lost.append(True) or real(env))
        counter = tmp_path / "counter"
        result = make_env().execute(f"echo ran >> {counter}; echo hi", resource_scope=True)
        assert result["returncode"] == 0 and "hi" in result["output"], result
        assert counter.read_text() == "ran\n"
        assert len(stub.calls()) == 1 and lost == [True]

    @pytest.mark.platforms("linux")
    def test_command_that_started_is_never_rerun(self, worker_host, fg_config, stub_factory, make_env, tmp_path):
        fg_config()
        stub = stub_factory("ran_then_fail")
        counter = tmp_path / "counter"
        result = make_env().execute(f"echo ran >> {counter}", resource_scope=True)
        assert result["returncode"] == 1, result
        assert counter.read_text() == "ran\n"
        assert len(stub.calls()) == 1


# --------------------------------------------------------------------------- T6: limits visible to the agent
@pytest.mark.platforms("linux")
class TestLimitNote:
    EAGAIN = "bash: fork: retry: Resource temporarily unavailable"

    def test_scoped_failure_names_the_limit_and_the_switch(self, worker_host, fg_config, stub, make_env):
        fg_config()
        result = make_env().execute(f"echo '{self.EAGAIN}' >&2; exit 2", resource_scope=True)
        assert result["returncode"] == 2
        lines = [line for line in result["output"].splitlines() if "terminal.foreground_scope" in line]
        assert len(lines) == 1 and "TasksMax" in lines[0], result["output"]

    def test_unscoped_command_output_is_unchanged(self, cli_host, fg_config, stub, make_env):
        fg_config()
        result = make_env().execute(f"echo '{self.EAGAIN}' >&2; exit 2", resource_scope=True)
        assert "terminal.foreground_scope" not in result["output"]

    def test_clean_exit_never_gets_the_note(self, worker_host, fg_config, stub, make_env):
        fg_config()
        result = make_env().execute(f"echo '{self.EAGAIN}'", resource_scope=True)
        assert result["returncode"] == 0 and "terminal.foreground_scope" not in result["output"]

    def test_unconfirmed_markers_are_not_matched(self, worker_host, fg_config, stub, make_env):
        fg_config()
        result = make_env().execute("echo 'java.lang.OutOfMemoryError: unable to create native thread' >&2; exit 1",
                                    resource_scope=True)
        assert "terminal.foreground_scope" not in result["output"]
