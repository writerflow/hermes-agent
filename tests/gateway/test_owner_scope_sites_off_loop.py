"""Runtime audit: gateway sites that enter ``owner_scope`` must not read profile files on the loop.

On a multiplexed gateway the sync ``owner_scope`` reads the profile ``.env``, ``config.yaml`` and
terminal policy. On the asyncio loop thread that read once stalled the loop for minutes on a swapping
host. Each case drives the real public entry of one audited site with a profile home of its own and
records, per ``agent.secret_scope.load_env_file`` call, the thread and the path read. Reads are
attributed to a case by the profile home the path sits under, never by call stack, so the oracle
survives a thread hop and a fix that moves the read into a new function.

Per case: reach (the site read the expected number of times, so a case that never reaches its site
fails instead of passing vacuously) and location (no read for the case's home on the loop thread).
"""

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def _make_home(root, name):
    home = root / "profiles" / name
    home.mkdir(parents=True)
    (home / ".env").write_text("TERMINAL_ENV=local\nLOOPSTALL_PROBE_KEY=from-profile\n", encoding="utf-8")
    (home / "config.yaml").write_text("terminal:\n  backend: local\n", encoding="utf-8")
    return home


def _authority(home, **extra):
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
    return SimpleNamespace(profile_id=str(home), runner=runner, **extra)


def _bound():
    from agent.secret_scope import current_secret_scope, current_secret_scope_home
    from hermes_constants import get_hermes_home_override
    from tools.terminal_scope import get_terminal_scope
    secrets = current_secret_scope()
    terminal = get_terminal_scope()
    return {"home": get_hermes_home_override(),
            "secrets": dict(secrets) if secrets is not None else None,
            "secret_home": current_secret_scope_home(),
            "terminal": dict(terminal) if terminal is not None else None}


class _EnvReads:
    """Records ``(thread ident, thread name, path)`` of every ``load_env_file`` call."""

    def __init__(self, monkeypatch):
        from agent import secret_scope

        self.reads = []
        real = secret_scope.load_env_file

        def wrapper(path, *args, **kwargs):
            self.reads.append((threading.get_ident(), threading.current_thread().name, Path(str(path))))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(secret_scope, "load_env_file", wrapper)

    def reads_for(self, home):
        home = Path(str(home))
        return [r for r in self.reads if r[2].parent == home]

    def on_loop(self, home, loop_ident):
        return [r for r in self.reads_for(home) if r[0] == loop_ident]

    def describe(self, home):
        return [(name, str(path)) for _, name, path in self.reads_for(home)]


@pytest.fixture
def reads(monkeypatch):
    return _EnvReads(monkeypatch)


@pytest.fixture
def n0(tmp_path, reads):
    """Reads one scope entry makes (pins the argument shape), measured on a calibration home."""
    from gateway.session_authorities import owner_scope
    home = _make_home(tmp_path, "calibration")
    with owner_scope(_authority(home)):
        pass
    count = len(reads.reads_for(home))
    assert count >= 1, "load_env_file wrapper patched at the wrong import point or scope read nothing"
    reads.reads.clear()
    return count


def _assert_reach(reads, home, expected):
    assert len(reads.reads_for(home)) == expected, (
        f"site reached {len(reads.reads_for(home))} reads, expected {expected}: {reads.describe(home)}")


def _assert_off_loop(reads, home, loop_ident):
    on_loop = reads.on_loop(home, loop_ident)
    assert on_loop == [], f"profile files for {home.name} read on the event loop: {on_loop}"


# --- Calibration: the oracle sees on-loop reads and worker reads -----------------------------------

@pytest.mark.asyncio
async def test_c1_sync_owner_scope_in_a_coroutine_is_seen_on_the_loop(tmp_path, reads, n0):
    from gateway.session_authorities import owner_scope
    home = _make_home(tmp_path, "c1")
    with owner_scope(_authority(home)):
        pass
    assert len(reads.on_loop(home, threading.get_ident())) == n0


@pytest.mark.asyncio
async def test_c2_async_owner_scope_reads_in_a_worker(tmp_path, reads, n0):
    from gateway.session_authorities import async_owner_scope
    home = _make_home(tmp_path, "c2")
    async with async_owner_scope(_authority(home)):
        pass
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())


@pytest.mark.asyncio
async def test_c3_another_authoritys_home_is_not_attributed(tmp_path, reads, n0):
    from gateway.session_authorities import owner_scope
    mine, other = _make_home(tmp_path, "c3-mine"), _make_home(tmp_path, "c3-other")
    with owner_scope(_authority(other)):
        pass
    assert reads.reads_for(mine) == []
    assert len(reads.reads_for(other)) == n0


# --- Row 1: AuthorityConnection.close ---------------------------------------------------------------

def _close_fixture(tmp_path, monkeypatch, seen):
    from gateway import session_controls, session_llm_oneshot, session_local_migration
    home = _make_home(tmp_path, "close")
    subscription = object()
    live = SimpleNamespace(subscribers=[subscription])

    async def detach(actor, sub, orphaned=False):
        seen.append(_bound())

    authority = _authority(home, sessions={"s1": live}, events={"t": 1}, detach=detach)
    monkeypatch.setattr(session_llm_oneshot, "forget", lambda *_a, **_k: None)
    monkeypatch.setattr(session_local_migration, "unbind_native_transport", lambda *_a, **_k: None)
    conn = object.__new__(session_controls.AuthorityConnection)
    conn.authority = authority
    conn.actor = SimpleNamespace(transport_id="t")
    conn._siblings = {}
    conn.subscriptions = {"s1": subscription}
    conn.handed_back = ()
    return home, conn


@pytest.mark.asyncio
async def test_site_session_controls_close_reads_the_profile_off_the_loop(tmp_path, reads, n0, monkeypatch):
    seen = []
    home, conn = _close_fixture(tmp_path, monkeypatch, seen)
    await conn.close()
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())


@pytest.mark.asyncio
async def test_site_session_controls_close_detaches_under_the_owners_scope(tmp_path, reads, n0, monkeypatch):
    from gateway.session_authorities import owner_scope
    seen = []
    home, conn = _close_fixture(tmp_path, monkeypatch, seen)
    with owner_scope(conn.authority):
        expected = _bound()
    await conn.close()
    assert seen == [expected]
    assert expected["home"] == str(home)


# --- Row 11: API-turn recovery on adapter publication (11a reconnect, 11b startup) -------------------

class _PublishRunner:
    """Only the state the real publish, reconnect-install and startup-aggregate bodies read."""

    def __init__(self, authorities, order):
        from gateway.config import Platform
        from gateway.run_adapters import GatewayAdapterLifecycleMixin
        from gateway.run_startup import GatewayStartupMixin

        self.order = order
        self.adapters = {}
        self.delivery_router = SimpleNamespace(adapters=None)
        self._failed_platforms = {Platform.API_SERVER: {"predecessor": None}}
        self.session_authorities = SimpleNamespace(serving=lambda: list(authorities))
        self.status_updates = []
        self.safe_disconnects = []
        cls = type(self)
        for name in ("_publish_primary_adapter", "_install_reconnected_adapter", "_prepare_primary_adapter"):
            if hasattr(GatewayAdapterLifecycleMixin, name):
                setattr(cls, name, getattr(GatewayAdapterLifecycleMixin, name))
        for name in ("_start_aggregate_connect_results", "_startup_queue_transient_failure", "_startup_retry_entry"):
            setattr(cls, name, getattr(GatewayStartupMixin, name))

    def _sync_voice_mode_state_to_adapter(self, adapter):
        self.order.append("voice_sync")

    def _bind_voice_input_callback(self, adapter):
        self.order.append("voice_bind")

    def _update_platform_runtime_status(self, name, **kw):
        self.status_updates.append((name, kw))

    def _schedule_restart_notice_replay(self):
        pass

    async def _redeliver_failed_obligations_for_platform(self, platform):
        pass

    def _schedule_resume_pending_sessions(self, platform=None):
        return 0

    async def _safe_adapter_disconnect(self, adapter, platform):
        self.safe_disconnects.append(adapter)
        await adapter.disconnect()


class _ApiAdapter:
    send_path_degraded = False
    has_fatal_error = False
    fatal_error_retryable = True
    DEGRADED_STATUS_MESSAGE = ""

    def __init__(self):
        self.disconnects = 0

    async def disconnect(self):
        self.disconnects += 1


def _publish_world(tmp_path, monkeypatch, names=("alpha", "beta")):
    from gateway import run_adapters, session_api_turn
    from gateway import channel_directory
    homes = [_make_home(tmp_path, name) for name in names]
    authorities = [_authority(home) for home in homes]
    order, recovered, handed = [], [], []

    def recover(adapter, authority):
        order.append("recover")
        recovered.append((authority, threading.get_ident(), _bound()["home"]))

    monkeypatch.setattr(session_api_turn, "recover_authority_api_turns", recover)
    monkeypatch.setattr(run_adapters, "hand_over_held_inbound", lambda *a: handed.append(a))

    async def no_directory(_adapters):
        return None

    monkeypatch.setattr(channel_directory, "build_channel_directory", no_directory)
    runner = _PublishRunner(authorities, order)
    adapter = _ApiAdapter()
    adapter.gateway_runner = runner
    return runner, adapter, homes, authorities, order, recovered, handed


@pytest.mark.asyncio
async def test_site_api_recover_reconnect_reads_each_profile_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from gateway.config import Platform
    runner, adapter, homes, _auth, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
    loop_ident = threading.get_ident()
    await runner._install_reconnected_adapter(Platform.API_SERVER, adapter)
    for home in homes:
        _assert_reach(reads, home, n0)
        _assert_off_loop(reads, home, loop_ident)
    assert runner.adapters[Platform.API_SERVER] is adapter
    assert [(ident == loop_ident) for _a, ident, _h in recovered] == [True, True]
    assert [h for _a, _i, h in recovered] == [str(home) for home in homes]


@pytest.mark.asyncio
async def test_site_api_recover_startup_reads_each_profile_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from gateway.config import Platform
    runner, adapter, homes, _auth, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
    loop_ident = threading.get_ident()
    count = await runner._start_aggregate_connect_results(
        [(Platform.API_SERVER, adapter, object(), "ok", None)], [], [])
    assert count == 1 and runner.adapters[Platform.API_SERVER] is adapter
    for home in homes:
        _assert_reach(reads, home, n0)
        _assert_off_loop(reads, home, loop_ident)
    assert [(ident == loop_ident) for _a, ident, _h in recovered] == [True, True]
