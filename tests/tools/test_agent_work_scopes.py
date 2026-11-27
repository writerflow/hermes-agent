"""Focused WRI-2560 policy checks with an inert manager boundary."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_abort_recovery, update_cmd_commit, update_cmd_posix_pause as pause
from tools import process_registry, work_scope


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("case", ["verified", "wrong policy", "wrong invocation", "missing invocation"])
def test_abort_recovery_checks_the_scope_before_starting_the_worker(monkeypatch, tmp_path, case):
    monkeypatch.setattr(update_abort_recovery.shutil, "which", lambda name: "/usr/bin/systemd-run")
    monkeypatch.setenv("INVOCATION_ID", "not-the-new-scope")
    monkeypatch.setattr(process_registry, "systemd_user_bus_env", lambda env: {
        **env, "DBUS_SESSION_BUS_ADDRESS": "unix:path=/fixture/bus",
    })
    seen = []
    started = []
    invocation = b"\x01" * 16
    scope = work_scope.ScopeBus.__new__(work_scope.ScopeBus)
    verified = []
    monkeypatch.setattr(scope, "verify_owner", lambda: verified.append("owner"))
    monkeypatch.setattr(scope, "scope_identity", lambda name, membership: ("/fixture", invocation))
    monkeypatch.setattr(scope, "property", lambda *args: "stop" if case == "wrong policy" else "continue")
    monkeypatch.setattr(work_scope, "ScopeBus", lambda route, deadline: scope)
    monkeypatch.setattr(work_scope, "_CGROUP_ROOT", tmp_path)

    def exec_worker(path, argv, env):
        assert verified == ["owner"]
        started.append(argv)

    monkeypatch.setattr(work_scope.os, "execvpe", exec_worker)

    def manager(argv, **kwargs):
        separator = argv.index("--")
        wrapper, launch = argv[:separator], argv[separator + 1:]
        assert "OOMPolicy=continue" in wrapper
        name = wrapper[wrapper.index("--unit") + 1]
        assert launch[3] == name
        assert "INVOCATION_ID" not in kwargs["env"]
        seen.append(json.loads(kwargs["input"]))
        target = tmp_path / name
        target.mkdir()
        (target / "cgroup.procs").write_text(f"{os.getpid()}\n", encoding="ascii")
        read_text = Path.read_text
        def read_membership(path, **options):
            return f"0::/{name}\n" if path == Path("/proc/self/cgroup") else read_text(path, **options)

        monkeypatch.setattr(Path, "read_text", read_membership)
        if case == "missing invocation":
            monkeypatch.delenv("INVOCATION_ID")
        else:
            monkeypatch.setenv("INVOCATION_ID", "02" * 16 if case == "wrong invocation" else invocation.hex())
        return subprocess.CompletedProcess(argv, work_scope._run_scope_launcher(name, launch[4:]), "", "")

    monkeypatch.setattr(update_abort_recovery.subprocess, "run", manager)
    result = update_abort_recovery._run_fresh_recovery_process(
        ["fixture"], {"fixture": "systemd"}, gateway_mode=True, recover_serve=False, skip_units=[],
    )
    assert result is not None
    assert result.returncode == (0 if case == "verified" else 1)
    request, = seen
    assert request["profiles"] == ["fixture"]
    assert [argv[1:] for argv in started] == (
        [["-m", "hermes_cli.update_restart_recovery", "--stdin"]] if case == "verified" else []
    )


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("stderr", [b"OOMPolicy refused", b""])
def test_failed_retained_scope_probe_cannot_degrade_on_warm_cache(monkeypatch, stderr):
    monkeypatch.setenv("INVOCATION_ID", "fixture-invocation")
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(process_registry, "_SYSTEMD_SCOPE_AVAILABLE", None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemd-run")

    probes = []

    def manager(argv, **kwargs):
        probes.append(argv)
        return subprocess.CompletedProcess(argv, 1, b"", stderr)

    monkeypatch.setattr(process_registry.subprocess, "run", manager)
    for _ in range(2):
        with pytest.raises(process_registry.RestartSafeScopeUnavailable, match="OOMPolicy"):
            process_registry.restart_safe_gateway_child_argv(
                ["payload"], unit_suffix="warm-probe", require_restart_safe_scope=False,
            )
    assert len(probes) == 1
    assert "OOMPolicy=continue" in probes[0]


@pytest.mark.platforms("linux")
def test_retained_dispatch_refuses_if_scope_helper_disappears_after_probe(monkeypatch):
    monkeypatch.setenv("INVOCATION_ID", "fixture-invocation")
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(process_registry, "_SYSTEMD_SCOPE_AVAILABLE", None)
    binaries = iter(("/usr/bin/systemd-run", None))
    monkeypatch.setattr("shutil.which", lambda name: next(binaries))

    def manager(argv, **kwargs):
        assert "OOMPolicy=continue" in argv
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(process_registry.subprocess, "run", manager)
    with pytest.raises(process_registry.RestartSafeScopeUnavailable, match="disappeared"):
        process_registry.restart_safe_gateway_child_argv(
            ["payload"], unit_suffix="lost-helper", require_restart_safe_scope=False,
        )


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("bus, scope", [("--user", "user"), ("--system", "system")])
@pytest.mark.parametrize("policy", [
    "continue", "stop", "kill", "changed invocation", "replaced cgroup",
    "missing member", "unreadable membership", None,
])
def test_escape_checks_policy_before_update_can_switch(monkeypatch, tmp_path, bus, scope, policy):
    original = "/hermes-gateway.service"
    action = []
    moved_to = []
    requested = []
    cgroup_root = tmp_path / "cgroup"
    monkeypatch.setattr(work_scope, "_CGROUP_ROOT", cgroup_root, raising=False)
    invocation_reads = iter(([31] * 16, [32 if policy == "changed invocation" else 31] * 16))
    unit = {"pid": 1, "unit": "hermes-gateway.service", "cgroup": original, "scope": scope}
    monkeypatch.setattr(pause, "_pid_cgroup", lambda pid="self": (moved_to or [original])[0])
    monkeypatch.setattr(pause, "_ppid", lambda pid: 1)
    monkeypatch.setattr(pause, "_discover_systemd", lambda: [unit])
    monkeypatch.setattr(pause, "_discover_launchd", list)
    monkeypatch.setattr(pause, "_discover_bare", lambda excluded: ([], []))
    monkeypatch.setattr(pause, "_manage_cmd", lambda scope: ["systemctl", bus])
    monkeypatch.setattr(pause, "_RUN", pause._empty_token())
    monkeypatch.setattr(pause, "_escape_failed", False)
    monkeypatch.setattr(pause, "_stop_gateways", lambda *args: action.append("quiesce"))
    monkeypatch.setattr(update_cmd_commit, "arm_commit_obligations", lambda *args: None)
    monkeypatch.setattr(update_cmd_commit, "arm_tree_move", lambda *args, **kw: action.append("switch"))
    from hermes_cli import update_pause_record

    for name in ("record_pause", "mark_stop_requested", "sync"):
        monkeypatch.setattr(update_pause_record, name, lambda *args, **kw: None)
    monkeypatch.setattr(update_pause_record, "identity", lambda pid: {"ct": "ct:1"})

    def manager(argv, **kwargs):
        assert argv[0] == "busctl" and bus in argv
        data = None
        if "GetId" in argv:
            data = {"type": "s", "data": ["fixture-bus-id"]}
        elif "GetNameOwner" in argv:
            data = {"type": "s", "data": [":1.2"]}
        elif "StartTransientUnit" in argv:
            args = argv[argv.index("StartTransientUnit") + 2:]
            requested.append(args)
            moved_to.append("/app.slice/" + args[0])
            target = cgroup_root / moved_to[0][1:]
            target.mkdir(parents=True)
            if policy != "unreadable membership":
                membership = "" if policy == "missing member" else f"{os.getpid()}\n"
                (target / "cgroup.procs").write_text(membership, encoding="ascii")
            data = {"type": "o", "data": ["/org/freedesktop/systemd1/job/42"]}
        elif "GetUnit" in argv:
            data = {"type": "o", "data": ["/org/freedesktop/systemd1/unit/fixture"]}
        elif argv[-1] == "InvocationID":
            data = {"type": "ay", "data": next(invocation_reads)}
        elif argv[-2:] == ["org.freedesktop.systemd1.Scope", "ControlGroup"]:
            data = {"type": "s", "data": moved_to[0]}
        elif argv[-1] == "ActiveState":
            data = {"type": "s", "data": "active"}
        elif argv[-1] == "OOMPolicy" and policy is not None:
            if policy == "replaced cgroup":
                target = cgroup_root / moved_to[0][1:]
                target.rename(target.with_name("original.scope"))
                target.mkdir()
            value = "continue" if policy in {
                "changed invocation", "replaced cgroup", "missing member", "unreadable membership",
            } else policy
            data = {"type": "s", "data": value}
        return subprocess.CompletedProcess(argv, 0 if data else 1, json.dumps(data), "")

    monkeypatch.setattr(pause.subprocess, "run", manager)
    result = update_cmd_commit.arm_commit_point(["git"], tmp_path, "a" * 40)
    assert len(requested) == 1
    if policy != "continue":
        assert result and "OOMPolicy" in result, "movement alone must not authorize the update"
        assert action == []
        assert not pause._escape_cgroup(unit), "an unverified escape stays refused after movement"
    else:
        assert result is None
        assert action == ["quiesce", "switch"]
    at = requested[0].index("OOMPolicy")
    assert requested[0][at:at + 3] == ["OOMPolicy", "s", "continue"]
    assert str(os.getpid()) in requested[0]
