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
from contextlib import asynccontextmanager, contextmanager
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
        recovered.append((authority, threading.get_ident(), _bound()))

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
    runner, adapter, homes, _auth, _order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
    loop_ident = threading.get_ident()
    await runner._install_reconnected_adapter(Platform.API_SERVER, adapter)
    for home in homes:
        _assert_reach(reads, home, n0)
        _assert_off_loop(reads, home, loop_ident)
    assert runner.adapters[Platform.API_SERVER] is adapter
    assert [(ident == loop_ident) for _a, ident, _h in recovered] == [True, True]
    assert [b["home"] for _a, _i, b in recovered] == [str(home) for home in homes]


@pytest.mark.asyncio
async def test_site_api_recover_startup_reads_each_profile_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from gateway.config import Platform
    runner, adapter, homes, _auth, _order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
    loop_ident = threading.get_ident()
    count = await runner._start_aggregate_connect_results(
        [(Platform.API_SERVER, adapter, object(), "ok", None)], [], [])
    assert count == 1 and runner.adapters[Platform.API_SERVER] is adapter
    for home in homes:
        _assert_reach(reads, home, n0)
        _assert_off_loop(reads, home, loop_ident)
    assert [(ident == loop_ident) for _a, ident, _h in recovered] == [True, True]


# --- Publication protocol: the profile read is the only await, and it precedes installation ----------

class _Gate:
    """Holds ``prepare_owner_scope`` open at its suspension and releases it on demand."""

    def __init__(self, monkeypatch, fail_for=()):
        from gateway import session_authorities
        self.started, self.release = asyncio.Event(), asyncio.Event()
        real = session_authorities.prepare_owner_scope
        self.fail_for = tuple(fail_for)

        async def gated(authority):
            if authority.profile_id in self.fail_for:
                raise OSError("profile unreadable")
            self.started.set()
            await self.release.wait()
            return await real(authority)

        monkeypatch.setattr(session_authorities, "prepare_owner_scope", gated)

    async def blocked(self):
        try:
            await asyncio.wait_for(self.started.wait(), timeout=2)
        except TimeoutError:
            pytest.fail("adapter publication never suspended in prepare_owner_scope: "
                        "the profile read is not a separate step before installation")


@pytest.mark.asyncio
async def test_publish_never_partial_reconnect(tmp_path, reads, n0, monkeypatch):
    from gateway.config import Platform
    runner, adapter, _homes, _auth, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
    gate = _Gate(monkeypatch)
    task = asyncio.create_task(runner._install_reconnected_adapter(Platform.API_SERVER, adapter))
    await gate.blocked()
    assert runner.adapters.get(Platform.API_SERVER) is not adapter
    assert Platform.API_SERVER in runner._failed_platforms
    assert order == []
    gate.release.set()
    await task
    assert runner.adapters[Platform.API_SERVER] is adapter
    assert order == ["recover", "recover", "voice_sync", "voice_bind"]
    assert all(ident == threading.get_ident() for _a, ident, _b in recovered)


@pytest.mark.asyncio
async def test_publish_never_partial_startup(tmp_path, reads, n0, monkeypatch):
    from gateway.config import Platform
    runner, adapter, _homes, _auth, order, _recovered, _handed = _publish_world(tmp_path, monkeypatch)
    gate = _Gate(monkeypatch)
    task = asyncio.create_task(runner._start_aggregate_connect_results(
        [(Platform.API_SERVER, adapter, object(), "ok", None)], [], []))
    await gate.blocked()
    assert runner.adapters.get(Platform.API_SERVER) is not adapter
    assert order == []
    gate.release.set()
    assert await task == 1
    assert runner.adapters[Platform.API_SERVER] is adapter
    assert order == ["recover", "recover", "voice_sync", "voice_bind"]


@pytest.mark.asyncio
async def test_cancel_during_prepare_reconnect_disposes_the_unpublished_adapter(tmp_path, reads, n0, monkeypatch):
    from gateway.config import Platform
    runner, adapter, _homes, _auth, order, _recovered, handed = _publish_world(tmp_path, monkeypatch)
    gate = _Gate(monkeypatch)
    task = asyncio.create_task(runner._install_reconnected_adapter(Platform.API_SERVER, adapter))
    await gate.blocked()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner.adapters.get(Platform.API_SERVER) is not adapter
    assert adapter.disconnects == 1
    assert handed == [(adapter, None)]
    assert Platform.API_SERVER in runner._failed_platforms
    assert order == []


@pytest.mark.asyncio
async def test_cancel_during_prepare_startup_disconnects_the_unpublished_adapter(tmp_path, reads, n0, monkeypatch):
    from gateway.config import Platform
    runner, adapter, _homes, _auth, order, _recovered, _handed = _publish_world(tmp_path, monkeypatch)
    gate = _Gate(monkeypatch)
    task = asyncio.create_task(runner._start_aggregate_connect_results(
        [(Platform.API_SERVER, adapter, object(), "ok", None)], [], []))
    await gate.blocked()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner.adapters.get(Platform.API_SERVER) is not adapter
    assert runner.safe_disconnects == [adapter] and adapter.disconnects == 1
    assert runner.status_updates == [] and order == []


@pytest.mark.asyncio
async def test_prepare_failure_degrades_one_authority(tmp_path, reads, n0, monkeypatch, caplog):
    import logging
    from gateway.config import Platform
    runner, adapter, homes, _authorities, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
    gate = _Gate(monkeypatch, fail_for=[str(homes[1])])
    gate.release.set()
    with caplog.at_level(logging.WARNING):
        await runner._install_reconnected_adapter(Platform.API_SERVER, adapter)
    assert runner.adapters[Platform.API_SERVER] is adapter and order[-2:] == ["voice_sync", "voice_bind"]
    assert [b["home"] for _a, _i, b in recovered] == [str(h) for h in homes]
    assert any(homes[1].name in rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING)
    # alpha was prepared in a worker; beta fell back to the sync scope (on the loop, accepted and logged)
    assert reads.on_loop(homes[0], threading.get_ident()) == []
    assert len(reads.reads_for(homes[1])) == n0


@pytest.mark.asyncio
async def test_authority_added_or_retired_during_prepare(tmp_path, reads, n0, monkeypatch, caplog):
    import logging
    from gateway.config import Platform
    runner, adapter, homes, authorities, _order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
    live = list(authorities)
    runner.session_authorities = SimpleNamespace(serving=lambda: list(live))
    gate = _Gate(monkeypatch)
    task = asyncio.create_task(runner._install_reconnected_adapter(Platform.API_SERVER, adapter))
    await gate.blocked()
    added = _authority(_make_home(tmp_path, "gamma"))
    live.append(added)
    live.remove(authorities[0])  # retiring: no longer served
    with caplog.at_level(logging.WARNING):
        gate.release.set()
        await task
    got = [b["home"] for _a, _i, b in recovered]
    assert got == [str(homes[1]), added.profile_id]
    assert any("gamma" in rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING)


def test_publish_is_synchronous_between_install_and_voice_wiring(tmp_path, monkeypatch):
    """A caller that does not await sees the adapter installed, recovered and wired at once."""
    from gateway.config import Platform
    runner, adapter, _homes, _auth, order, _recovered, _handed = _publish_world(tmp_path, monkeypatch)
    runner._publish_primary_adapter(Platform.API_SERVER, adapter)
    assert runner.adapters[Platform.API_SERVER] is adapter
    assert order == ["recover", "recover", "voice_sync", "voice_bind"]


@pytest.mark.asyncio
async def test_unrouted_owner_binds_its_prepared_scope_without_reading_or_warning(tmp_path, reads, n0, monkeypatch, caplog):
    """A symbolic-id owner has no profile files: its prepared entry is ``None`` and recovery binds exactly
    what the sync ``owner_scope`` binds, with no fallback warning (it is not a failed preparation)."""
    from agent import secret_scope
    from tui_gateway import launch_profile_policy
    # Multiplex is off in tests, so ``owner_scope`` of an unrouted owner would bind nothing and the case
    # could not tell a recovery that binds the launch scope from one that binds nothing. Host several
    # profiles (as tests/tui_gateway/test_legacy_fallback_launch_explicit_profile.py does) so it binds
    # the launch profile's real scope, while the authority's runner stays ``multiplex_profiles=False``.
    launch = _make_home(tmp_path, "launch")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    previous = secret_scope.is_multiplex_active()
    launch_profile_policy.activate_multi_profile_hosting()
    try:
        await _unrouted_case(tmp_path, reads, monkeypatch, caplog, launch)
    finally:
        secret_scope.set_multiplex_active(previous)


async def _unrouted_case(tmp_path, reads, monkeypatch, caplog, launch):
    import logging
    from gateway import session_api_turn
    from gateway.session_authorities import owner_scope
    runner, adapter, homes, authorities, _order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
    unrouted = SimpleNamespace(profile_id="fixture", runner=SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False)))
    runner.session_authorities = SimpleNamespace(serving=lambda: [unrouted, authorities[0]])
    with owner_scope(unrouted):
        expected = _bound()
    assert expected["home"] == str(launch), "owner_scope of an unrouted owner must bind the launch scope here"
    assert expected["secrets"], "the expected binding must not be empty"
    prepared = await session_api_turn.prepare_api_recovery(adapter)
    assert prepared[id(unrouted)] is None
    with caplog.at_level(logging.WARNING):
        session_api_turn.recover_api_turns(adapter, prepared)
    assert recovered[0][0] is unrouted and recovered[0][2] == expected
    assert recovered[1][2]["home"] == str(homes[0])
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert reads.reads_for(Path("fixture")) == []


# --- Row 13: detached-pin follow mounts a session that is not mounted -------------------------------

def _detached_world(tmp_path, monkeypatch, *, mounted=False, restore=None):
    from gateway import session_detached, session_route_recovery
    home = _make_home(tmp_path, "detached")
    live = SimpleNamespace(route="agent:main:chat")
    authority = _authority(home, sessions={"sid": live} if mounted else {})
    runner = SimpleNamespace(session_authority=authority,
                             session_store=SimpleNamespace(_named_profile_for_key=lambda route: None))
    restored = []

    def restore_routed_session(auth, session_id):
        restored.append((session_id, _bound()["home"]))
        if restore is not None:
            restore()

    monkeypatch.setattr(session_route_recovery, "restore_routed_session", restore_routed_session)
    key = f"agent:main:{session_detached._SLOT}:sid"
    evt = {"session_key": key, "text": "result"}
    return home, runner, evt, restored


@pytest.mark.asyncio
async def test_site_detached_mounted_reads_the_profile_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from gateway.session_detached import follow_detached_pin
    home, runner, evt, restored = _detached_world(tmp_path, monkeypatch)
    assert await follow_detached_pin(runner, evt) == evt
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())
    assert restored == [("sid", str(home))]  # the mount still runs under the owner's scope


@pytest.mark.asyncio
async def test_site_detached_mounted_enters_no_scope_for_a_session_already_mounted(tmp_path, reads, n0, monkeypatch):
    from gateway.session_detached import follow_detached_pin
    home, runner, evt, restored = _detached_world(tmp_path, monkeypatch, mounted=True)
    assert (await follow_detached_pin(runner, evt))["session_key"] == "agent:main:chat"
    assert reads.reads_for(home) == [] and restored == []


@pytest.mark.asyncio
async def test_site_detached_mounted_a_draining_store_still_leaves_the_event_unchanged(tmp_path, reads, n0, monkeypatch):
    from gateway.session_detached import follow_detached_pin
    from hermes_state_runtime import RuntimeStoreError

    def draining():
        raise RuntimeStoreError("runtime_draining")

    _home, runner, evt, _restored = _detached_world(tmp_path, monkeypatch, restore=draining)
    before = _bound()
    assert await follow_detached_pin(runner, evt) == evt
    assert _bound() == before, "the scope leaked past the mount"


# --- Off-loop rows: regression assertions (no product change expected) --------------------------------

def _session_authority(home, **extra):
    live = SimpleNamespace(source=None, subscribers=[])
    return _authority(
        home, sessions={"sid": live}, agent=lambda ref: None,
        authorize=lambda actor, ref, capability: None, **extra)


def _conn(authority):
    return SimpleNamespace(authority=authority, actor=SimpleNamespace(capabilities={"session:read"}, transport_id="t"))


@pytest.mark.asyncio
async def test_site_session_ancillary_show_tools_reads_off_the_loop(tmp_path, reads, n0, monkeypatch):
    import model_tools
    from gateway import session_ancillary, session_policy
    from gateway.session_contract import SessionRef
    home = _make_home(tmp_path, "show-tools")
    authority = _session_authority(home)
    monkeypatch.setattr(session_policy, "policy_for_source", lambda runner, source: None)
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: [])
    await session_ancillary.show_tools(_conn(authority), SessionRef(str(home), "sid"), {"session_id": "sid"})
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())


@pytest.mark.asyncio
async def test_site_session_ancillary_reload_skills_reads_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from agent import skill_commands
    from gateway import session_ancillary, session_policy
    from gateway.session_contract import SessionRef
    home = _make_home(tmp_path, "reload-skills")
    authority = _session_authority(home)
    monkeypatch.setattr(session_policy, "policy_for_source", lambda runner, source: None)
    monkeypatch.setattr(session_policy, "policy_scope", lambda policy, authority=None: __import__("contextlib").nullcontext())
    monkeypatch.setattr(skill_commands, "reload_skills", lambda: {"added": [], "removed": [], "total": 0})
    await session_ancillary.reload_skills(_conn(authority), SessionRef(str(home), "sid"), {"session_id": "sid"})
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())


@pytest.mark.asyncio
async def test_site_session_tui_reads_run_tui_read_reads_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from contextlib import nullcontext
    from gateway import session_commands, session_tui_reads
    from gateway.session_contract import SessionRef
    home = _make_home(tmp_path, "tui-read")
    authority = _session_authority(home)
    monkeypatch.setattr(session_commands, "_session_workspace", lambda runner, source: nullcontext())
    out = await session_tui_reads.run_tui_read(authority, SessionRef(str(home), "sid"), authority.sessions["sid"],
                                               None, lambda *a: "rendered", "")
    assert out["output"] == "rendered"
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())


@pytest.mark.asyncio
async def test_site_session_transcript_subagent_reads_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from gateway import session_transcript_subagent
    from gateway.session_contract import SessionRef
    home = _make_home(tmp_path, "transcript")
    authority = _session_authority(home)
    monkeypatch.setattr(session_transcript_subagent, "_child", lambda *a: (None, False))
    result = await session_transcript_subagent.subagent_transcript(
        _conn(authority), SessionRef(str(home), "sid"), {"session_id": "sid"}, ("sid", "sub"), None, 10, False)
    assert result["available"] is False
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())


@pytest.mark.asyncio
async def test_site_session_llm_oneshot_work_runs_on_the_owners_pool_not_the_loop(tmp_path, reads, n0, monkeypatch):
    """Through the real ``DeferredReply.start`` and the owner's pool: no test-created executor."""
    from agent import oneshot as oneshot_module
    from gateway import session_llm_oneshot
    from gateway.session_contract import SessionRef
    home = _make_home(tmp_path, "oneshot")
    authority = _authority(home)
    connection = SimpleNamespace(authority=authority, transport=SimpleNamespace(),
                                 actor=SimpleNamespace(capabilities={"session:read"}, transport_id="t"))
    monkeypatch.setattr(oneshot_module, "run_oneshot", lambda **kw: "title")
    monkeypatch.setattr("gateway.session_authorities.require_serving", lambda a: None)
    reply = await session_llm_oneshot.oneshot(connection, SessionRef(str(home), ""), {"input": "x"})
    reply.start(connection, 1)  # the production scheduling path
    for _ in range(1000):
        if len(reads.reads_for(home)) >= n0:
            break
        await asyncio.sleep(0.01)
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())
    assert reads.reads_for(home)[0][1] != threading.current_thread().name


@pytest.mark.asyncio
async def test_site_session_usage_account_usage_reads_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from agent import account_usage as account_usage_module, account_usage_nous, billing_usage
    from gateway import session_llm_oneshot, session_usage
    from gateway.session_contract import SessionRef
    home = _make_home(tmp_path, "usage")

    async def resolve(actor, ref):
        return None

    authority = _session_authority(home, resolve=resolve)
    monkeypatch.setattr(session_llm_oneshot, "_session_runtime", lambda auth, ref: (dict))
    monkeypatch.setattr(account_usage_module, "account_limit_lines", lambda *a, **k: [])
    monkeypatch.setattr(account_usage_nous, "nous_usage_view", lambda **k: ([], None))
    monkeypatch.setattr(billing_usage, "usage_model_wire", lambda usage: None)
    await session_usage.account_usage(_conn(authority), SessionRef(str(home), "sid"), {"session_id": "sid"})
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())


@pytest.mark.asyncio
async def test_site_session_tui_reads_context_breakdown_reads_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from contextlib import nullcontext
    from agent import context_breakdown, context_file_sources
    from gateway import session_commands, session_tui_reads
    from gateway.session_contract import SessionRef
    home = _make_home(tmp_path, "breakdown")
    agent = SimpleNamespace()
    authority = _session_authority(home, db=SimpleNamespace(conversation_revision=lambda tip: 1))
    authority.agent = lambda ref: agent
    authority.sessions["sid"].context_breakdown = None
    monkeypatch.setattr(session_commands, "_session_workspace", lambda runner, source: nullcontext())
    monkeypatch.setattr(session_tui_reads, "_context_id", lambda auth, ref: "tip")
    monkeypatch.setattr(session_tui_reads, "_agent_context_marks", lambda a: ())
    monkeypatch.setattr(session_tui_reads, "_model_history", lambda db, tip: [])
    monkeypatch.setattr(context_breakdown, "compute_session_context_breakdown", lambda a, c: {"categories": []})
    monkeypatch.setattr(context_file_sources, "context_file_sources_for_agent", lambda a: [])
    await session_tui_reads.context_breakdown(_conn(authority), SessionRef(str(home), "sid"), {"session_id": "sid"})
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())


@pytest.mark.asyncio
async def test_site_session_hosted_transport_admission_check_runs_through_the_claim_hop(tmp_path, reads, n0, monkeypatch):
    """The real ``SessionAuthority._claim_next``: the admission check's ``owner_scope`` must land in the
    ``to_thread`` hop the claim makes, not on the loop. The test never calls ``to_thread`` itself."""
    import sqlite3
    from contextlib import closing, contextmanager
    from gateway import (session_authority, session_deferred_pick, session_local_recovery,
                         session_setting_commands)
    from gateway.config import Platform
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import RuntimeStoreError
    home = _make_home(tmp_path, "admission")
    row = {"status": "queued", "request_id": "hosted:r1", "admission_id": "a1", "payload": {}}

    @contextmanager
    def read_ctx():
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("CREATE TABLE state_meta (key TEXT, value TEXT)")
            yield conn

    async def no_op(*a, **k):
        return None

    authority = _authority(home, db=SimpleNamespace(_read_ctx=read_ctx))
    live = SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL), claim_gate=asyncio.Lock(),
                           mutation_lock=asyncio.Lock())
    monkeypatch.setattr(session_authority, "list_session_admissions", lambda db, session_id=None: [row])
    monkeypatch.setattr(session_setting_commands, "apply_pending_reasoning", no_op)
    monkeypatch.setattr(session_deferred_pick, "apply_deferred_pick", no_op)
    monkeypatch.setattr(session_local_recovery, "restore_local_session", lambda *a: None)
    with pytest.raises(RuntimeStoreError) as raised:
        await session_authority.SessionAuthority._claim_next(authority, SessionRef(str(home), "sid"), live)
    assert raised.value.reason == "permission_denied"
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())


async def _hosted_dispatch(tmp_path, verb, params_extra, attest):
    """Install the real hosted transport and send ``verb`` through the real dispatcher's executors."""
    import json
    from gateway.control_dispatch import ControlRequestDispatcher
    from gateway.session_hosted_transport import install_hosted_transport
    home = _make_home(tmp_path, "hosted-" + verb)
    authority = _authority(home, epoch=1, db=SimpleNamespace())
    server = SimpleNamespace(private_handlers={})
    install_hosted_transport(server, authority, asyncio.get_running_loop(), attest=attest)
    handler_fn = server.private_handlers[verb]

    def handler(raw, peer):
        return json.dumps(handler_fn(json.loads(raw)["params"], peer)).encode()

    dispatcher = ControlRequestDispatcher()
    request = json.dumps({"verb": verb, "params": {"profile_id": str(home), **params_extra}}).encode()
    try:
        return home, await dispatcher.dispatch(handler, request, "peer")
    finally:
        dispatcher.close()


@pytest.mark.asyncio
async def test_site_session_hosted_transport_source_handler_runs_on_a_control_worker(tmp_path, reads, n0):
    seen = []

    def attest(selector, operation, params):
        seen.append(threading.current_thread().name)
        return {"owner": "o"}

    home, _out = await _hosted_dispatch(tmp_path, "hosted-attest", {
        "selector": {"room_id": "r", "member_id": "m", "profile": "p"}, "operation": "submit", "params": {}}, attest)
    assert seen and seen[0].startswith("gateway-control")
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())


@pytest.mark.asyncio
async def test_site_session_hosted_transport_target_handler_runs_on_a_producer_worker(tmp_path, reads, n0):
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError) as raised:  # a malformed envelope: refused right after the scope is entered
        await _hosted_dispatch(tmp_path, "hosted-producer", {"bogus": 1}, lambda *a: {"owner": "o"})
    assert raised.value.reason == "invalid_params"
    homes = {r[2].parent for r in reads.reads}
    assert len(homes) == 1
    home = next(iter(homes))
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())
    assert reads.reads_for(home)[0][1].startswith("gateway-hosted-producer")


# --- The 9 sync ``with owner_scope`` sites in coroutines (t_9328163d) -------------------------------------
# Each case drives the real public entry, passes the reach assertion on the unchanged tree (the site read
# the profile files, so the fixture reached it) and fails location there until the site is moved off the loop.

def _expected_scope(authority):
    from gateway.session_authorities import owner_scope
    with owner_scope(authority):
        return _bound()


class _HopGate:
    """Holds the worker-thread profile read (``_prepare_owner_scope``) open until released."""

    def __init__(self, monkeypatch):
        from gateway import session_authorities
        self.entered, self.release = threading.Event(), threading.Event()
        real = session_authorities._prepare_owner_scope

        def gated(home):
            self.entered.set()
            assert self.release.wait(10), "hop gate never released"
            return real(home)

        monkeypatch.setattr(session_authorities, "_prepare_owner_scope", gated)

    async def wait_entered(self):
        for _ in range(500):
            if self.entered.is_set():
                return
            await asyncio.sleep(0.01)
        pytest.fail("the site never made the worker-thread profile read (no hop)")


# --- Site 1: HostedRoomAuthorityRPC._dispatch (session_hosted_rpc.py) ---------------------------------

def _hosted_rpc(tmp_path, monkeypatch, *, timeout=30, home_name="hosted-rpc"):
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    home = _make_home(tmp_path, home_name)
    authority = _authority(home)
    principal = SimpleNamespace(subject="alice", profile_id=str(home))
    rpc = HostedRoomAuthorityRPC(authority, asyncio.get_running_loop(), room_id="room", member_id="m",
                                 profile="default", principal=principal, authorize=lambda *a: True,
                                 timeout=timeout)
    owned = []

    async def dispatch_owned(operation, params):
        owned.append((operation, threading.get_ident(), _bound()))
        return {"ok": operation}

    monkeypatch.setattr(rpc, "_dispatch_owned", dispatch_owned)
    return home, authority, rpc, owned


@pytest.mark.asyncio
async def test_site_hosted_rpc_dispatch_reads_the_profile_off_the_loop(tmp_path, reads, n0, monkeypatch):
    home, _auth, rpc, owned = _hosted_rpc(tmp_path, monkeypatch)
    assert await asyncio.to_thread(rpc._call, "submit") == {"ok": "submit"}
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())
    assert len(owned) == 1 and owned[0][1] == threading.get_ident()  # the guarded work stays on the loop


@pytest.mark.asyncio
async def test_site_hosted_rpc_dispatch_binds_what_owner_scope_binds(tmp_path, reads, n0, monkeypatch):
    home, authority, rpc, owned = _hosted_rpc(tmp_path, monkeypatch)
    expected = _expected_scope(authority)
    assert expected["home"] == str(home)
    await asyncio.to_thread(rpc._call, "submit")
    assert [b for _o, _t, b in owned] == [expected]
    assert _bound()["home"] != str(home), "the scope leaked past the dispatch"


@pytest.mark.asyncio
async def test_site_hosted_rpc_timeout_during_the_hop_never_cancels_the_dispatch(tmp_path, reads, n0, monkeypatch):
    """A caller timeout is ambiguous: the dispatch is not cancelled, so it may still admit, once, under the owner."""
    home, authority, rpc, owned = _hosted_rpc(tmp_path, monkeypatch, timeout=0.2)
    expected = _expected_scope(authority)
    reads.reads.clear()
    gate = _HopGate(monkeypatch)
    call = asyncio.create_task(asyncio.to_thread(rpc._call, "submit"))
    await gate.wait_entered()
    with pytest.raises(TimeoutError):
        await call
    assert owned == []  # still inside the hop
    gate.release.set()
    for _ in range(500):
        if owned:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    assert [b for _o, _t, b in owned] == [expected], "the timed-out dispatch must still complete once, under the owner"
    _assert_off_loop(reads, home, threading.get_ident())


@pytest.mark.asyncio
async def test_site_hosted_rpc_task_cancellation_during_the_hop_runs_nothing(tmp_path, reads, n0, monkeypatch):
    home, _auth, rpc, owned = _hosted_rpc(tmp_path, monkeypatch)
    before = _bound()
    gate = _HopGate(monkeypatch)
    task = asyncio.create_task(rpc._dispatch("submit", {}))
    await gate.wait_entered()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.release.set()
    await asyncio.sleep(0.1)
    assert owned == [] and _bound() == before
    assert reads.on_loop(home, threading.get_ident()) == []


@pytest.mark.asyncio
async def test_site_hosted_rpc_unrouted_owner_reads_nothing_and_binds_the_launch_scope(tmp_path, reads, n0, monkeypatch):
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    launch = _make_home(tmp_path, "launch")
    with _launch_hosting(launch, monkeypatch):
        authority = SimpleNamespace(profile_id="fixture", runner=SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False)))
        expected = _expected_scope(authority)
        assert expected["home"] == str(launch) and expected["secrets"]
        principal = SimpleNamespace(subject="alice", profile_id="fixture")
        rpc = HostedRoomAuthorityRPC(authority, asyncio.get_running_loop(), room_id="room", member_id="m",
                                     profile="default", principal=principal, authorize=lambda *a: True)
        got = []

        async def dispatch_owned(operation, params):
            got.append(_bound())

        monkeypatch.setattr(rpc, "_dispatch_owned", dispatch_owned)
        reads.reads.clear()
        await asyncio.to_thread(rpc._call, "submit")
    assert got == [expected]
    assert reads.reads_for(Path("fixture")) == []


@contextmanager
def _launch_hosting(launch, monkeypatch):
    """Host several profiles (as the unrouted case above does) so an unrouted owner binds the launch scope."""
    from agent import secret_scope
    from tui_gateway import launch_profile_policy
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    previous = secret_scope.is_multiplex_active()
    launch_profile_policy.activate_multi_profile_hosting()
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(previous)


# --- Site 2: GatewayRuntimeAPI._serve (run_api.py), one scope per WebSocket connection ---------------

def _attach_world(tmp_path, monkeypatch, name="ws-serve"):
    """Attach over the real ``/api/ws`` admission for a routed (multiplexed) owner with its own home.

    Returns ``(home, authority, runner, handled, sent, start)``; ``start()`` makes the connection.
    """
    from gateway import run_api
    from gateway.run_api import GatewayRuntimeAPI
    from gateway.runtime_bootstrap import TicketStore
    from hermes_cli import web_server as web
    from hermes_cli import web_server_chat
    from tests.gateway.test_session_socket_preload import _attach_scope

    home = _make_home(tmp_path, name)
    tickets = TicketStore("owner", [str(home)])
    ticket = tickets.mint(profile_id=str(home), subject="human", purpose="interactive")
    authority = SimpleNamespace(db=SimpleNamespace(db_path=home / "state.db"), profile_id=str(home),
                                runner=SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True)))
    runner = SimpleNamespace(_draining=False, session_runtime_descriptor={"state": "ready"},
                             session_authority=authority, session_ticket_store=tickets)
    handled, sent = [], []

    async def handler_factory():
        async def handle(ws, **kwargs):
            handled.append((threading.get_ident(), _bound()))
            await ws.close()
        return handle

    monkeypatch.setattr(run_api, "session_socket_handler", handler_factory)
    monkeypatch.setattr(web, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(web_server_chat, "_ws_request_is_allowed", lambda ws: True)

    async def fallback(*args):
        pytest.fail("a native ticket must not enter the legacy fallback")

    incoming = iter([{"type": "websocket.connect"}])

    async def receive():
        return next(incoming)

    async def send(value):
        sent.append(value)

    def start():
        return asyncio.create_task(GatewayRuntimeAPI(fallback, runner, web.app)(_attach_scope(ticket), receive, send))

    return home, authority, runner, handled, sent, start


@pytest.mark.asyncio
async def test_site_ws_serve_reads_the_profile_off_the_loop(tmp_path, reads, n0, monkeypatch):
    home, _auth, _runner, handled, _sent, start = _attach_world(tmp_path, monkeypatch)
    await start()
    assert len(handled) == 1
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())


@pytest.mark.asyncio
async def test_site_ws_serve_handler_sees_the_owners_binding(tmp_path, reads, n0, monkeypatch):
    home, authority, _runner, handled, _sent, start = _attach_world(tmp_path, monkeypatch)
    expected = _expected_scope(authority)
    assert expected["home"] == str(home)
    await start()
    assert [b for _t, b in handled] == [expected]
    assert handled[0][0] == threading.get_ident()


@pytest.mark.asyncio
async def test_site_ws_serve_drain_during_the_hop_refuses_with_1013_and_never_calls_the_handler(tmp_path, reads, n0, monkeypatch):
    from gateway import run_idle_exit
    home, _auth, runner, handled, sent, start = _attach_world(tmp_path, monkeypatch)
    gate = _HopGate(monkeypatch)
    task = start()
    await gate.wait_entered()
    assert run_idle_exit._client_busy(runner, window=0.0) == "1 client(s) attached"  # counted through the hop
    runner._draining = True  # a drain that begins while the profile read is in flight
    gate.release.set()
    await task
    assert handled == []
    assert sent[-1] == {"type": "websocket.close", "code": 1013, "reason": ""}
    assert _bound()["home"] != str(home)


# --- Site 3: serve_profile_runtime (run_runtime.py), one scope per profile served ---------------------

class _ServeRegistry:
    def __init__(self):
        self.removed = []

    def for_home(self, home):
        return None

    def remove(self, home):
        self.removed.append(Path(home).resolve())


def _serve_world(tmp_path, monkeypatch):
    """A real ``serve_profile_runtime`` over a routed home; every recovery step records its binding."""
    from gateway import run_runtime
    from gateway import session_hosted_service
    from gateway.platforms import webhook_ingress
    from gateway import session_local_recovery

    home = _make_home(tmp_path, "served").resolve()
    authority = _authority(home)
    registry = _ServeRegistry()
    runner = SimpleNamespace(session_authorities=registry, session_control_server=None)
    steps, retired = [], []

    def record(step):
        steps.append((step, threading.get_ident(), _bound()))

    async def build(*args, **kwargs):
        return authority

    async def retire(auth, timeout=None):
        retired.append(auth)
        return True

    async def mailbox(auth):
        record("bot_mailbox")

    async def webhook(auth):
        record("webhook")

    monkeypatch.setattr(run_runtime, "_build_profile_authority", build)
    monkeypatch.setattr(run_runtime, "_retire_profile_authority", retire)
    monkeypatch.setattr(run_runtime, "_publish_served_set", lambda r: None)
    monkeypatch.setattr(run_runtime, "_drop_retired_local_adapter", lambda a: record("drop_adapter"))
    monkeypatch.setattr(run_runtime, "recover_bot_mailbox", mailbox)
    monkeypatch.setattr(run_runtime, "_recover_published_api_turns", lambda r, a: record("api_turns"))
    monkeypatch.setattr(session_local_recovery, "recover_local_sessions", lambda a, schedule: record("local_sessions"))
    monkeypatch.setattr(webhook_ingress, "recover_webhook_finalizations", webhook)
    monkeypatch.setattr(session_hosted_service, "start_ready_hosted_services", lambda r: None)
    return home, authority, runner, registry, steps, retired


_SERVE_ORDER = ["drop_adapter", "bot_mailbox", "local_sessions", "webhook", "api_turns"]


@pytest.mark.asyncio
async def test_site_serve_profile_runtime_reads_the_profile_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from gateway import run_runtime
    home, authority, runner, _registry, steps, _retired = _serve_world(tmp_path, monkeypatch)
    expected = _expected_scope(authority)
    reads.reads.clear()
    assert await run_runtime.serve_profile_runtime(runner, "served", home) is authority
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())
    assert [s for s, _t, _b in steps] == _SERVE_ORDER
    assert all(b == expected and t == threading.get_ident() for _s, t, b in steps)


@pytest.mark.asyncio
async def test_site_serve_profile_runtime_a_failing_prepare_retires_the_authority(tmp_path, reads, n0, monkeypatch):
    from gateway import run_runtime, session_authorities
    home, authority, runner, registry, steps, retired = _serve_world(tmp_path, monkeypatch)

    async def unreadable(auth):
        raise OSError("profile unreadable")

    monkeypatch.setattr(session_authorities, "prepare_owner_scope", unreadable)
    with pytest.raises(OSError, match="profile unreadable"):
        await run_runtime.serve_profile_runtime(runner, "served", home)
    assert retired == [authority] and registry.removed == [home] and steps == []


@pytest.mark.asyncio
async def test_site_serve_profile_runtime_cancel_during_prepare_retires_the_authority(tmp_path, reads, n0, monkeypatch):
    from gateway import run_runtime
    home, authority, runner, registry, steps, retired = _serve_world(tmp_path, monkeypatch)
    gate = _HopGate(monkeypatch)
    task = asyncio.create_task(run_runtime.serve_profile_runtime(runner, "served", home))
    await gate.wait_entered()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.release.set()
    assert retired == [authority] and registry.removed == [home] and steps == []


# --- Site 4: _recover_profile_bot_mailbox (run_runtime.py), startup Bot Chat recovery per profile ----

class _MailboxRegistry(_ServeRegistry):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def profile_name(self, authority):
        return self.name


def _mailbox_world(tmp_path, monkeypatch, *, name):
    """``name`` is the profile's name, or None for the launch profile."""
    from gateway import run_runtime
    home = _make_home(tmp_path, "mailbox").resolve()
    authority = _authority(home)
    registry = _MailboxRegistry(name)
    runner = SimpleNamespace(session_authorities=registry)
    recovered, retired, parked = [], [], []

    async def mailbox(auth):
        recovered.append((threading.get_ident(), _bound()))

    async def retire(auth, timeout=None):
        retired.append(auth)
        return True

    monkeypatch.setattr(run_runtime, "recover_bot_mailbox", mailbox)
    monkeypatch.setattr(run_runtime, "_retire_profile_authority", retire)
    monkeypatch.setattr(run_runtime, "_publish_served_set", lambda r: None)
    monkeypatch.setattr(run_runtime, "release_profile_home", lambda r, h: None)
    monkeypatch.setattr(run_runtime, "park_profile", lambda r, n, reason: parked.append((n, reason)))
    return home, authority, runner, registry, recovered, retired, parked


@pytest.mark.asyncio
async def test_site_bot_mailbox_recovery_reads_the_profile_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from gateway import run_runtime
    home, authority, runner, _registry, recovered, _retired, parked = _mailbox_world(tmp_path, monkeypatch, name="beta")
    expected = _expected_scope(authority)
    reads.reads.clear()
    await run_runtime._recover_profile_bot_mailbox(runner, authority)
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())
    assert recovered == [(threading.get_ident(), expected)] and parked == []


@pytest.mark.asyncio
async def test_site_bot_mailbox_recovery_a_failing_prepare_parks_a_secondary(tmp_path, reads, n0, monkeypatch):
    from gateway import run_runtime, session_authorities
    home, authority, runner, registry, recovered, retired, parked = _mailbox_world(tmp_path, monkeypatch, name="beta")

    async def unreadable(auth):
        raise OSError("profile unreadable")

    monkeypatch.setattr(session_authorities, "prepare_owner_scope", unreadable)
    await run_runtime._recover_profile_bot_mailbox(runner, authority)
    assert recovered == [] and retired == [authority] and registry.removed == [home]
    assert [n for n, _r in parked] == ["beta"] and "profile unreadable" in parked[0][1]


@pytest.mark.asyncio
async def test_site_bot_mailbox_recovery_a_failing_prepare_aborts_the_launch_profile(tmp_path, reads, n0, monkeypatch):
    from gateway import run_runtime, session_authorities
    _home, authority, runner, _registry, recovered, retired, parked = _mailbox_world(tmp_path, monkeypatch, name=None)

    async def unreadable(auth):
        raise OSError("profile unreadable")

    monkeypatch.setattr(session_authorities, "prepare_owner_scope", unreadable)
    with pytest.raises(OSError, match="profile unreadable"):
        await run_runtime._recover_profile_bot_mailbox(runner, authority)
    assert recovered == [] and retired == [] and parked == []


# --- Sites 5 and 6: native-session recovery (run_runtime.py) -----------------------------------------
# Site 5 (``recover_gateway_native_sessions``) enters one scope per authority and then calls site 6
# (``recover_profile_native_sessions``), which enters its own: each home sees two scope entries.

def _recovery_world(tmp_path, monkeypatch, names=("rec-a", "rec-b")):
    from gateway import session_detached, session_hosted_service, session_local_recovery
    from gateway.platforms import webhook_ingress
    homes = [_make_home(tmp_path, name).resolve() for name in names]
    steps, native = [], []
    authorities = []
    for home in homes:
        async def recover_native(bindings, _home=home):
            native.append((str(_home), threading.get_ident(), _bound()))
            return {}

        authorities.append(_authority(
            home, db=SimpleNamespace(_read_all=lambda sql: []), logical_owner=lambda sid: sid,
            recover_native_sessions=recover_native))
    runner = SimpleNamespace(session_authorities=SimpleNamespace(serving=lambda: list(authorities)),
                             session_store=SimpleNamespace(list_sessions=list))

    async def ensure(_runner):
        return None

    async def webhook(auth):
        steps.append(("webhook", auth.profile_id, _bound()["home"]))

    monkeypatch.setattr(session_hosted_service, "ensure_hosted_service", ensure)
    monkeypatch.setattr(session_local_recovery, "recover_local_sessions",
                        lambda auth, schedule: steps.append(("local", auth.profile_id, _bound()["home"])))
    monkeypatch.setattr(webhook_ingress, "recover_webhook_finalizations", webhook)
    monkeypatch.setattr(session_detached, "recover_detached_sessions", lambda auth, pending: [])
    return homes, authorities, runner, steps, native


@pytest.mark.asyncio
async def test_site_recover_profile_native_sessions_reads_the_profile_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from gateway import run_runtime
    homes, authorities, runner, _steps, native = _recovery_world(tmp_path, monkeypatch, names=("rec-a",))
    expected = _expected_scope(authorities[0])
    reads.reads.clear()
    assert await run_runtime.recover_profile_native_sessions(runner, authorities[0]) == {}
    _assert_reach(reads, homes[0], n0)
    _assert_off_loop(reads, homes[0], threading.get_ident())
    assert native == [(str(homes[0]), threading.get_ident(), expected)]


@pytest.mark.asyncio
async def test_site_recover_gateway_native_sessions_reads_each_profile_off_the_loop(tmp_path, reads, n0, monkeypatch):
    from gateway import run_runtime
    homes, authorities, runner, steps, native = _recovery_world(tmp_path, monkeypatch)
    expected = {a.profile_id: _expected_scope(a) for a in authorities}
    reads.reads.clear()
    assert await run_runtime.recover_gateway_native_sessions(runner) == {}
    loop_ident = threading.get_ident()
    for home in homes:
        # Two entries per home: site 5's own, then site 6's (``recover_profile_native_sessions`` is not stubbed).
        _assert_reach(reads, home, 2 * n0)
        _assert_off_loop(reads, home, loop_ident)
    # Per-authority isolation and order: each home's steps ran under that home's binding.
    assert steps == [(step, str(home), str(home)) for home in homes for step in ("local", "webhook")]
    assert [(h, b) for h, _t, b in native] == [(str(home), expected[str(home)]) for home in homes]


# --- Site 7: _admit_local_delivery (session_cron.py), one scope per cron delivery to a local chat -----

class _RecordingSessions(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gets = []

    def get(self, key, default=None):
        self.gets.append(key)
        return super().get(key, default)


def _delivery_world(tmp_path, monkeypatch, *, known=True):
    from gateway import session_automation, session_cron, session_local_recovery, session_route_recovery
    from gateway.config import Platform
    home = _make_home(tmp_path, "delivery").resolve()
    source = SimpleNamespace(platform=Platform.TELEGRAM)
    live = SimpleNamespace(source=source)
    sessions = _RecordingSessions({"chat1": live} if known else {})
    adapter = object()
    authority = _authority(home, sessions=sessions, db=SimpleNamespace(db_path=home / "state.db"))
    authority.runner._adapter_for_source = lambda src: adapter
    admitted = []

    async def admit_route_wake(runner, adp, src, text, kind, metadata):
        admitted.append((threading.get_ident(), _bound(), adp is adapter, text, kind))
        return True

    def restore_local_session(auth, chat_id):
        raise KeyError(chat_id)

    monkeypatch.setattr(session_automation, "admit_route_wake", admit_route_wake)
    monkeypatch.setattr(session_local_recovery, "restore_local_session", restore_local_session)
    monkeypatch.setattr(session_route_recovery, "mountable", lambda auth, chat_id: False)
    session_cron.bind_owner(authority)
    return home, authority, sessions, admitted, session_cron


@pytest.mark.asyncio
async def test_site_cron_local_delivery_reads_the_profile_off_the_loop(tmp_path, reads, n0, monkeypatch):
    home, authority, _sessions, admitted, session_cron = _delivery_world(tmp_path, monkeypatch)
    try:
        expected = _expected_scope(authority)
        reads.reads.clear()
        assert await asyncio.to_thread(session_cron.deliver_to_local_chat, home, "chat1", "fire") is None
    finally:
        session_cron.unbind_owner(authority)
    _assert_reach(reads, home, n0)
    _assert_off_loop(reads, home, threading.get_ident())
    assert admitted == [(threading.get_ident(), expected, True, "fire", "cron")]


@pytest.mark.asyncio
async def test_site_cron_local_delivery_looks_the_chat_up_only_after_the_hop(tmp_path, reads, n0, monkeypatch):
    home, authority, sessions, _admitted, session_cron = _delivery_world(tmp_path, monkeypatch, known=False)
    try:
        gate = _HopGate(monkeypatch)
        call = asyncio.create_task(asyncio.to_thread(session_cron.deliver_to_local_chat, home, "gone", "fire"))
        await gate.wait_entered()
        assert sessions.gets == []  # nothing read from the authority while the profile read is in flight
        gate.release.set()
        message = await call
    finally:
        session_cron.unbind_owner(authority)
    assert sessions.gets == ["gone"]
    assert message is not None and "KeyError" in message  # an unknown chat still surfaces as before
