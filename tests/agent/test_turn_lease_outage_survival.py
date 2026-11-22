"""A worker's session turn lease outlives an owner outage up to the persist budget.

A worker renews its lease through ``RuntimeSessionStore``, whose renewal queues behind the persist
retry on ``store.lock`` and ignores ``patience_s``. With the 300 s lease a renewal blocked for about
230 s left the next tick no patience and ``DurableTurnLease`` interrupted a healthy turn. The worker
store therefore advertises its own, longer ``turn_lease_ttl_seconds``; ``SessionDB`` keeps 300 s.
"""
import subprocess
import sys
import threading
import types

import pytest

import agent.turn_facade_lease as lease_module
from agent.turn_facade_lease import DurableTurnLease, admit_durable_turn_lease


class _Clock:
    def __init__(self):
        self.t = 0.0

    def time(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(lease_module, "time", types.SimpleNamespace(time=fake.time))
    return fake


def _blocking_db(clock, ttl, outage_start, recover, refreshed):
    """A db that models the worker path: a renewal requested while the owner is silent returns only
    once the persist retry ends (``store.lock``), whatever ``patience_s`` says."""

    class Db:
        turn_lease_ttl_seconds = ttl

        def refresh_session_turn_lease(self, session_id, holder, ttl_seconds, patience_s):
            refreshed.append(ttl_seconds)
            if outage_start <= clock.t < recover:
                clock.t = recover
            return True

    return Db()


def _survives(clock, *, ttl, outage_start, outage_len, overshoot, interval=60.0, horizon=2000.0):
    """True when the turn is never interrupted. ``outage_start`` counts from the last renewal (t=0)."""
    clock.t = 0.0
    agent = types.SimpleNamespace(_session_turn_lease_refresh_interval=interval, session_id="s")
    interrupted = []
    db = _blocking_db(clock, ttl, outage_start, outage_start + outage_len + overshoot, [])
    lease = DurableTurnLease(agent, db, "s", "h", expires_at=ttl)
    lease._interrupt_turn = interrupted.append
    next_tick = interval
    while next_tick < horizon:
        clock.t = next_tick
        if lease.refresh_tick() is False or interrupted:
            return False
        next_tick = clock.t + interval
    return True


@pytest.mark.parametrize("outage_s", [180, 230, 284, 420])
def test_store_lease_survives_owner_outage_in_every_phase(clock, outage_s):
    killed = [
        (start, overshoot)
        for start in range(61) for overshoot in (0.0, 8.0)
        if not _survives(clock, ttl=600.0, outage_start=float(start), outage_len=float(outage_s), overshoot=overshoot)
    ]
    assert killed == []


def test_default_db_keeps_the_300_second_lease(clock):
    class SessionDbLike:  # no ``turn_lease_ttl_seconds``: the CLI / in-owner path
        calls = []

        def refresh_session_turn_lease(self, session_id, holder, ttl_seconds, patience_s):
            self.calls.append((ttl_seconds, patience_s))
            return True

    db = SessionDbLike()
    clock.t = 1000.0
    lease = DurableTurnLease(types.SimpleNamespace(session_id="s"), db, "s", "h")
    assert lease._authority_deadline == 1300.0
    clock.t = 1060.0
    assert lease.refresh_tick() is None
    assert db.calls == [(300.0, 20.0)]
    assert lease._authority_deadline == 1360.0


def test_non_numeric_db_attribute_is_ignored(clock):
    """MagicMock-style shims answer any attribute; only a real number may set the TTL."""
    from unittest.mock import MagicMock

    db = MagicMock()
    db.refresh_session_turn_lease.return_value = True
    clock.t = 1000.0
    lease = DurableTurnLease(types.SimpleNamespace(session_id="s"), db, "s", "h")
    assert lease._authority_deadline == 1300.0
    clock.t = 1060.0
    lease.refresh_tick()
    assert db.refresh_session_turn_lease.call_args.kwargs["ttl_seconds"] == 300.0


def test_store_backed_lease_uses_its_ttl_at_every_site(clock, monkeypatch):
    refreshed = []

    class StoreLike:
        turn_lease_ttl_seconds = 600.0
        acquired_with = []

        def get_session(self, session_id):
            return {"id": session_id}

        def acquire_session_turn_lease(self, session_id, holder, *, ttl_seconds, **kwargs):
            self.acquired_with.append(ttl_seconds)
            return True

        def refresh_session_turn_lease(self, session_id, holder, ttl_seconds, patience_s):
            refreshed.append(ttl_seconds)
            return True

        def release_session_turn_lease(self, session_id, holder):
            return None

    db = StoreLike()
    agent = types.SimpleNamespace(
        _session_db=db, session_id="s1", _persist_disabled=False, _interrupt_requested=False,
        _session_turn_lease_refresh_interval=60.0, _touch_activity=lambda *a, **k: None,
        _liveness_activity_lock=lambda: threading.Lock(), _emit_status=lambda *_: None,
    )
    monkeypatch.setattr(lease_module, "LEASE_WAIT_SECONDS", 1.0)
    clock.t = 1000.0
    admission = admit_durable_turn_lease(
        agent, session_id="s1", relay_turn_id="s1:t:abcd",
        task_context={"session_id": "s1", "task_id": "t", "platform": "cli"}, conversation_history=None,
    )
    lease = admission.lease
    assert db.acquired_with == [600.0]
    # No committed expiry readable from the store: the pre-acquire floor plus ITS ttl.
    assert lease._authority_deadline == 1600.0
    clock.t = 1060.0
    assert lease.refresh_tick() is None
    assert refreshed == [600.0]
    assert lease._authority_deadline == 1660.0
    # The agent attribute feeds append_messages_batch's `!= 300.0` reroute and must stay 300.
    assert agent._active_session_turn_lease_ttl_seconds == 300.0


def test_lease_ttl_is_read_at_call_time(clock, monkeypatch):
    monkeypatch.setattr(lease_module, "LEASE_TTL_SECONDS", 123.0)
    clock.t = 10.0
    lease = DurableTurnLease(types.SimpleNamespace(session_id="s"), object(), "s", "h")
    assert lease._authority_deadline == 133.0


def test_worker_store_advertises_a_longer_lease_than_session_db():
    from agent.runtime_session_store import RuntimeSessionStore
    from hermes_state import SessionDB

    assert getattr(RuntimeSessionStore, "turn_lease_ttl_seconds", lease_module.LEASE_TTL_SECONDS) == 600.0
    assert getattr(SessionDB, "turn_lease_ttl_seconds", lease_module.LEASE_TTL_SECONDS) == 300.0


def test_default_worker_append_stays_on_transcript_append(tmp_path):
    """A 600 s store TTL must not reach `turn_lease_ttl_seconds`: anything but 300 reroutes every
    append to `compression.append`, a different receipt kind that filters fields."""
    from agent.runtime_session_store import RuntimeSessionStore

    seen = []

    def rpc(method, **params):
        seen.append(params["operation"])
        return {"count": 1, "annotations": [{}]}

    store = RuntimeSessionStore(rpc, {"execution_id": "e", "session_id": "s", "epoch": 1, "generation": 1}, tmp_path)
    try:
        store.append_messages_batch("s", [{"role": "user", "content": "x"}], turn_lease_holder="h")
    finally:
        store.close()
    assert seen == ["transcript.append"]


def test_ttl_covers_budget_and_every_clock_that_must_outlive_it():
    """The executable coupling: change the budget or a clock without the others and this fails."""
    from agent import turn_liveness
    from agent.runtime_session_store import OWNER_SILENCE_BUDGET_S, RECEIPT_ATTEMPT_TIMEOUT_S, RuntimeSessionStore
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    refresh_interval, max_backoff, margin, slack = 60.0, 8.0, lease_module._REFRESH_EXPIRY_MARGIN_S, 60.0
    worst_renewal = OWNER_SILENCE_BUDGET_S + RECEIPT_ATTEMPT_TIMEOUT_S + max_backoff
    store_ttl = getattr(RuntimeSessionStore, "turn_lease_ttl_seconds", lease_module.LEASE_TTL_SECONDS)
    assert store_ttl >= refresh_interval + worst_renewal + margin + slack
    # Third clock: the liveness watchdog must not fire on a turn still inside the persist budget.
    liveness_s, _ = turn_liveness.resolve_turn_liveness_settings(DEFAULT_CONFIG)
    assert liveness_s >= OWNER_SILENCE_BUDGET_S + 28


def test_blocked_renewal_does_not_delay_other_scheduled_callbacks():
    """A renewal parked on store.lock for the whole retry must not starve the liveness watchdog's timer."""
    from agent.periodic_scheduler import schedule

    release = threading.Event()
    entered = threading.Event()
    other_ran = threading.Event()
    running, peak = [0], [0]
    guard = threading.Lock()

    def blocked_renewal():
        with guard:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        entered.set()
        release.wait(10)
        with guard:
            running[0] -= 1
        return False

    blocked = schedule(blocked_renewal, 0.01)
    other = schedule(lambda: other_ran.set() or False, 0.01)
    try:
        assert entered.wait(10)
        assert other_ran.wait(10), "a second timer starved behind the blocked renewal"
        assert peak[0] == 1
    finally:
        release.set()
        blocked.cancel(wait=5)
        other.cancel(wait=5)


def _dead_pid():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _session_db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="cli")
    return db


def test_dead_local_holder_is_reclaimed_at_once_despite_a_600s_row(tmp_path):
    from hermes_state_pidns import holder_namespace_token

    db = _session_db(tmp_path)
    try:
        dead = f"pid={_dead_pid()}{holder_namespace_token()}:turn=old"
        assert db.try_acquire_session_turn_lease("s", dead, ttl_seconds=600.0)
        assert db.try_acquire_session_turn_lease("s", f"pid={_live_pid()}:turn=new", ttl_seconds=600.0)
    finally:
        db.close()


def test_holder_in_a_foreign_pid_namespace_waits_for_ttl(tmp_path):
    """The documented residual: no kernel proof, so a 600 s row is held for its whole TTL."""
    db = _session_db(tmp_path)
    try:
        foreign = f"pid={_dead_pid()}:pidns=1:turn=old"
        assert db.try_acquire_session_turn_lease("s", foreign, ttl_seconds=600.0)
        assert not db.try_acquire_session_turn_lease("s", f"pid={_live_pid()}:turn=new", ttl_seconds=600.0)
    finally:
        db.close()


def _live_pid():
    import os

    return os.getpid() + 1_000_000  # a different pid than ours; the new holder is never probed
