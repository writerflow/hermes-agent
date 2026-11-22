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
    runner, adapter, homes, _auth, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
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
    runner, adapter, homes, _auth, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
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
        except asyncio.TimeoutError:
            pytest.fail("adapter publication never suspended in prepare_owner_scope: "
                        "the profile read is not a separate step before installation")


@pytest.mark.asyncio
async def test_publish_never_partial_reconnect(tmp_path, reads, n0, monkeypatch):
    from gateway.config import Platform
    runner, adapter, homes, _auth, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
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
    runner, adapter, homes, _auth, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
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
    runner, adapter, homes, _auth, order, recovered, handed = _publish_world(tmp_path, monkeypatch)
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
    runner, adapter, homes, _auth, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
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
    runner, adapter, homes, authorities, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
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
    runner, adapter, homes, authorities, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
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
    runner, adapter, homes, _auth, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
    runner._publish_primary_adapter(Platform.API_SERVER, adapter)
    assert runner.adapters[Platform.API_SERVER] is adapter
    assert order == ["recover", "recover", "voice_sync", "voice_bind"]


@pytest.mark.asyncio
async def test_unrouted_owner_binds_its_prepared_scope_without_reading_or_warning(tmp_path, reads, n0, monkeypatch, caplog):
    """A symbolic-id owner has no profile files: its prepared entry is ``None`` and recovery binds exactly
    what the sync ``owner_scope`` binds, with no fallback warning (it is not a failed preparation)."""
    import logging
    from gateway import session_api_turn
    from gateway.session_authorities import owner_scope
    runner, adapter, homes, authorities, order, recovered, _handed = _publish_world(tmp_path, monkeypatch)
    unrouted = SimpleNamespace(profile_id="fixture", runner=SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False)))
    runner.session_authorities = SimpleNamespace(serving=lambda: [unrouted, authorities[0]])
    with owner_scope(unrouted):
        expected = _bound()
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

    home, runner, evt, restored = _detached_world(tmp_path, monkeypatch, restore=draining)
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
    monkeypatch.setattr(session_llm_oneshot, "_session_runtime", lambda auth, ref: (lambda: {}))
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

    home, out = await _hosted_dispatch(tmp_path, "hosted-attest", {
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
