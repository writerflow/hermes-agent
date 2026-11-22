"""A worker's lease renewal, persist retry and refresher cooperate across an owner stall.

``DurableTurnLease`` renews through ``RuntimeSessionStore``, whose renewal queues behind the persist
retry on ``store.lock``. These tests run the real store, the real ``WorkerRPC`` over a loopback
websocket owner and the real scheduler, with every constant scaled down.
"""
import json
import threading
import time
from types import SimpleNamespace

import pytest

import agent.runtime_session_store as store_module
import agent.turn_facade_lease as lease_module
from agent.turn_facade_lease import DurableTurnLease
from tests.waiting import wait_until

SCOPE = {"execution_id": "exec", "session_id": "s", "epoch": 1, "generation": 1}


def _agent():
    interrupts = []

    def interrupt(message, **kwargs):
        interrupts.append(message)

    return SimpleNamespace(
        session_id="s", interrupt=interrupt, _session_turn_lease_refresh_interval=0.2,
        _interrupt_requested=False, _interrupt_message=None,
    ), interrupts


def test_budget_exhaustion_fails_the_renewal_and_interrupts_the_turn_once(tmp_path, monkeypatch):
    """The owner never answers: the renewal raises when the persist budget is spent, the refresher
    interrupts the turn, and the scheduler then stops it, so the interrupt fires exactly once."""
    from agent.periodic_scheduler import schedule

    clock = [0.0]
    monkeypatch.setattr(store_module, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(store_module, "_sleep_or_cancel", lambda cancel, delay: clock.__setitem__(0, clock[0] + delay) or False)

    def rpc(method, **params):
        clock[0] += store_module.RECEIPT_ATTEMPT_TIMEOUT_S
        raise TimeoutError("worker_receipt_timeout")

    store = store_module.RuntimeSessionStore(rpc, SCOPE, tmp_path)
    agent, interrupts = _agent()
    lease = DurableTurnLease(agent, store, "s", "holder")
    lease.turn_active = True
    handle = schedule(lease.refresh_tick, 0.01)
    try:
        wait_until(lambda: interrupts and handle.cancelled, what="the refresher interrupted the turn and stopped")
    finally:
        handle.cancel(wait=5)
    assert len(interrupts) == 1
    assert "could not be refreshed" in interrupts[0]
    # The renewal itself raised the budget error, and the journal keeps the unacknowledged receipt.
    assert store.failure == "worker_receipt_timeout"
    assert store.journal["pending"]
    with pytest.raises(store_module.WorkerPersistenceError, match="worker_receipt_timeout"):
        store.refresh_session_turn_lease("s", "holder", ttl_seconds=300.0)


@pytest.fixture
def owner(monkeypatch):
    """A loopback websocket owner that answers every ``worker.persist`` unless held silent."""
    from websockets.sync.server import serve
    from hermes_cli import gateway_client, gateway_runtime
    import hermes_cli.gateway_runtime_discovery as discovery

    state = SimpleNamespace(silent=threading.Event(), go=threading.Event(), replies=[], tickets=0)

    def handler(ws):
        try:
            for raw in ws:
                request = json.loads(raw)
                operation = request["params"]["operation"]
                while state.silent.is_set() and not state.go.wait(0.05):
                    pass
                reply = {"count": 1, "annotations": [{}]} if operation == "transcript.append" else {"value": True}
                ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": reply}))
                state.replies.append((operation, time.monotonic()))
        except Exception:  # health: allow BLE001 -- the worker drops a silent attempt's socket by design
            return

    def ticket(home, endpoint, purpose):
        state.tickets += 1
        return f"ticket-{state.tickets}"

    with serve(handler, "127.0.0.1", 0, subprotocols=["hermes-gateway-v1"]) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.socket.getsockname()[1]
        endpoint = SimpleNamespace(api_origin=f"http://127.0.0.1:{port}", control_home=None)
        monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint",
                            lambda home, timeout: SimpleNamespace(state="ready", endpoint=endpoint))
        monkeypatch.setattr(discovery, "query_identify", lambda home, timeout: {"pid": -1})
        monkeypatch.setattr(gateway_client, "_session_ticket", ticket)
        try:
            yield state
        finally:
            state.go.set()
            server.shutdown()


def test_renewal_queued_behind_a_persist_retry_keeps_the_turn(owner, tmp_path, monkeypatch):
    """R7, scaled 1:30 (outage 3 s for 90 s): the append is retried through the silence, the refresher's
    renewal waits on ``store.lock`` until it lands, the turn is not interrupted, and the owner then
    accepts ``turn.renew``. The default TTL is scaled too, so a lease that ignored the store's
    longer one would die here exactly as the 300 s lease does at about 230 s."""
    from agent.periodic_scheduler import schedule

    budget, attempt, interval, silence = 8.0, 1.0, 0.2, 3.0
    monkeypatch.setattr(store_module, "OWNER_SILENCE_BUDGET_S", budget)
    monkeypatch.setattr(store_module, "RECEIPT_ATTEMPT_TIMEOUT_S", attempt)
    monkeypatch.setattr(store_module.WorkerRPC, "RECEIPT_ATTEMPT_TIMEOUT_S", attempt)
    monkeypatch.setattr(lease_module, "LEASE_TTL_SECONDS", 4.0)  # stands in for SessionDB's 300 s
    # The coupling invariant at test scale: interval + budget + attempt + max backoff (8) + margin + slack.
    monkeypatch.setattr(store_module.RuntimeSessionStore, "turn_lease_ttl_seconds",
                        interval + budget + attempt + 8.0 + lease_module._REFRESH_EXPIRY_MARGIN_S + 2.0, raising=False)

    store = store_module.RuntimeSessionStore(store_module.WorkerRPC(tmp_path), SCOPE, tmp_path / "outbox")
    agent, interrupts = _agent()
    lease = DurableTurnLease(agent, store, "s", "holder")
    lease.turn_active = True
    renewals = []
    real_refresh = store.refresh_session_turn_lease

    def timed_refresh(*args, **kwargs):
        started = time.monotonic()
        try:
            return real_refresh(*args, **kwargs)
        finally:
            renewals.append((started, time.monotonic()))

    store.refresh_session_turn_lease = timed_refresh
    outcome = {}

    def append():
        try:
            outcome["count"] = store.append_messages_batch("s", [{"role": "user", "content": "x"}])
        except Exception as exc:  # health: allow BLE001 -- handed to the asserting thread
            outcome["error"] = exc

    owner.silent.set()
    released = threading.Timer(silence, lambda: (owner.silent.clear(), owner.go.set()))
    began = time.monotonic()
    appender = threading.Thread(target=append)
    handle = schedule(lease.refresh_tick, interval)
    released.start()
    appender.start()
    try:
        appender.join(timeout=60)
        assert not appender.is_alive()
        # Past the recovery, keep the refresher running: a lease sized too short dies on the tick AFTER the
        # blocked renewal returns, not during it.
        wait_until(lambda: len(renewals) >= 3 or interrupts, what="renewals after the recovery")
    finally:
        released.cancel()
        handle.cancel(wait=5)
        owner.go.set()
    assert outcome == {"count": 1}
    assert time.monotonic() - began >= silence
    assert interrupts == []
    assert lease.stop.is_set() is False
    # The first renewal really queued behind the retry: it returned only after the append recovered.
    first_start, first_end = renewals[0]
    assert first_end - first_start >= silence - 1.0
    assert any(op == "turn.renew" for op, _ in owner.replies)
    assert store.failure is None
