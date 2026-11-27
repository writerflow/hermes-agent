"""Exercise real pipe and PTY children with a substituted scope manager."""

import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest

from tools import process_registry, work_scope


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("use_pty", [False, True], ids=["pipe", "pty"])
@pytest.mark.parametrize("verified", [False, True], ids=["refused", "admitted"])
def test_background_payload_waits_for_scope_verification(monkeypatch, tmp_path, use_pty, verified):
    marker = tmp_path / "payload"
    manager_log = tmp_path / "manager"
    manager = tmp_path / "systemd-run"
    manager.write_text(f"#!{sys.executable}\n" + f"""
import os
import runpy
import sys
from pathlib import Path

options, command = sys.argv[1:sys.argv.index('--')], sys.argv[sys.argv.index('--') + 1:]
log = Path({str(manager_log)!r})
log.write_text('requested\\n' if 'OOMPolicy=continue' in options else 'missing policy\\n')
if len(command) > 2 and command[2] == {str(Path(work_scope.__file__).resolve())!r}:
    entry = runpy.run_path(command[2])['_run_scope_launcher']
    def verify(*args, **kwargs):
        assert not Path({str(marker)!r}).exists()
        assert 'INVOCATION_ID' not in os.environ
        with log.open('a') as stream:
            stream.write('checked\\n')
        if not {verified!r}:
            raise entry.__globals__['ScopeVerificationError']('scope policy refused')
    entry.__globals__['_verify_scope_launch'] = verify
    sys.exit(entry(command[3], command[4:]))
os.execv(command[0], command)
""", encoding="utf-8")
    manager.chmod(0o700)
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: True)
    original_which = shutil.which
    monkeypatch.setattr("shutil.which", lambda name, **kw: str(manager) if name == "systemd-run" else original_which(name, **kw))
    monkeypatch.setattr(process_registry, "worker_output_log", lambda _: None)
    monkeypatch.setattr(process_registry.ProcessRegistry, "_spawn_env", staticmethod(lambda _: dict(os.environ)))
    monkeypatch.setenv("INVOCATION_ID", "parent")
    stops = []
    monkeypatch.setattr(process_registry, "_stop_systemd_unit", lambda name: stops.append(name))
    registry = process_registry.ProcessRegistry()
    command = f"printf x >> {shlex.quote(str(marker))}; printf 'output\\n'; exit 23"

    if verified:
        session = registry.spawn_local(command, cwd=str(tmp_path), use_pty=use_pty)
        session._reader_thread.join(timeout=5)
        assert not session._reader_thread.is_alive()
        assert session.exit_code == 23
        assert "output" in session.output_buffer
        assert marker.read_text() == "x"
    else:
        try:
            session = registry.spawn_local(command, cwd=str(tmp_path), use_pty=use_pty)
        except process_registry.RestartSafeScopeUnavailable:
            pass
        else:
            session._reader_thread.join(timeout=5)
            assert not marker.exists(), "payload ran without verified placement"
            pytest.fail("launch did not report the refusal")
        assert not marker.exists()
        assert not registry._running
    assert manager_log.read_text() == "requested\nchecked\n"
    assert not stops
