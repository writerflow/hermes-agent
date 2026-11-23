"""Read-location regressions for hosted, cron and recovery owner scopes."""
import asyncio
import concurrent.futures
import threading
from pathlib import Path

import pytest

from tests.gateway.test_hosted_mux_runtime import mux  # noqa: F401


def test_hosted_create_config(mux, monkeypatch):
    from gateway import run
    from gateway.session_contract import SessionActor
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC

    runner, homes, loop, call = mux
    authority = runner.session_authorities.require(homes['beta'])
    actor = SessionActor('alice', authority.profile_id,
                         frozenset({'session:create', 'session:read'}), 'private-owner')
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='probe', member_id='helper',
                                 profile='beta', principal=actor, authorize=lambda *args: True)
    async def current_thread():
        return threading.get_ident()
    loop_ident = call(current_thread())
    real = run._load_gateway_config
    observed = []

    def read(config_path=None):
        home = Path(config_path).parent if config_path is not None else run._gateway_config_home()
        observed.append((Path(home).resolve(), threading.get_ident()))
        return real(config_path)

    monkeypatch.setattr(run, '_load_gateway_config', read)
    result = rpc.create(profile='beta', source='bot_room', title='Group: probe')
    assert result['session_id'] == rpc.ref.session_id
    assert authority.db.get_session(rpc.ref.session_id) is not None
    assert observed, 'hosted create did not reach the config reader'
    assert all(home == homes['beta'].resolve() for home, _ in observed), observed
    assert all(ident != loop_ident for _, ident in observed), 'hosted config read on owner loop'


def test_hosted_create_rechecks_membership_after_config(mux, monkeypatch):
    from gateway import run
    from gateway.session_contract import SessionActor
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from hermes_state_runtime import RuntimeStoreError

    runner, homes, loop, _ = mux
    authority = runner.session_authorities.require(homes['beta'])
    actor = SessionActor('alice', authority.profile_id,
                         frozenset({'session:create', 'session:read'}), 'private-owner')
    allowed = threading.Event()
    allowed.set()
    entered, release = threading.Event(), threading.Event()
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='revoked', member_id='helper',
                                 profile='beta', principal=actor, authorize=lambda *args: allowed.is_set())
    real = run._load_gateway_config

    def held_read(*args, **kwargs):
        entered.set()
        if not release.wait(30):
            raise TimeoutError('config test release')
        return real(*args, **kwargs)

    monkeypatch.setattr(run, '_load_gateway_config', held_read)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rpc.create, profile='beta', source='bot_room', title='Group: revoked')
        try:
            assert entered.wait(20), 'hosted create did not reach config reader'
            allowed.clear()
        finally:
            release.set()
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            future.result(timeout=30)
    assert authority.db.get_session(rpc.ref.session_id) is None


def test_hosted_cancel_retains_physical_reader_until_settled(mux, monkeypatch):
    from gateway import run
    from gateway.session_contract import SessionActor
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.session_runtime_workers import mutation_tasks

    runner, homes, loop, call = mux
    authority = runner.session_authorities.require(homes['beta'])
    actor = SessionActor('alice', authority.profile_id,
                         frozenset({'session:create', 'session:read'}), 'private-owner')
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='cancel', member_id='helper',
                                 profile='beta', principal=actor, authorize=lambda *args: True)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real = run._load_gateway_config

    def held_read(*args, **kwargs):
        entered.set()
        try:
            if not release.wait(30):
                raise TimeoutError('config test release')
            return real(*args, **kwargs)
        finally:
            finished.set()

    async def begin():
        return asyncio.create_task(rpc._dispatch('create', {
            'profile': 'beta', 'source': 'bot_room', 'title': 'Group: cancel'}))

    async def cancel_and_count(task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return len(mutation_tasks(authority))

    monkeypatch.setattr(run, '_load_gateway_config', held_read)
    task = call(begin())
    async def settle_reads():
        await asyncio.gather(*mutation_tasks(authority))

    try:
        assert entered.wait(20), 'hosted create did not reach config reader'
        assert call(cancel_and_count(task)) >= 1, 'retirement lost the physical read'
    finally:
        release.set()
        assert finished.wait(30)
        call(settle_reads())
    assert authority.db.get_session(rpc.ref.session_id) is None


def test_hosted_cancel_consumes_late_read_error(mux, monkeypatch):
    from gateway import run
    from gateway.session_contract import SessionActor
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.session_runtime_workers import mutation_tasks

    runner, homes, loop, call = mux
    authority = runner.session_authorities.require(homes['beta'])
    actor = SessionActor('alice', authority.profile_id,
                         frozenset({'session:create', 'session:read'}), 'private-owner')
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='late-error', member_id='helper',
                                 profile='beta', principal=actor, authorize=lambda *args: True)
    entered, release = threading.Event(), threading.Event()

    def failing_read(*args, **kwargs):
        entered.set()
        if not release.wait(30):
            raise TimeoutError('config test release')
        raise ValueError('late synthetic read failure')

    async def begin():
        return asyncio.create_task(rpc._dispatch('create', {
            'profile': 'beta', 'source': 'bot_room', 'title': 'Group: late-error'}))

    async def cancel_and_capture(task):
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        tasks = mutation_tasks(authority)
        assert tasks, 'retirement lost the physical reader'
        return tasks

    async def settle(tasks):
        return await asyncio.gather(*tasks, return_exceptions=True)

    monkeypatch.setattr(run, '_load_gateway_config', failing_read)
    task = call(begin())
    try:
        assert entered.wait(20), 'hosted create did not reach config reader'
        tasks = call(cancel_and_capture(task))
    finally:
        release.set()
    results = call(settle(tasks))
    assert any(isinstance(value, ValueError) and str(value) == 'late synthetic read failure'
               for value in results)
    assert authority.db.get_session(rpc.ref.session_id) is None


def test_hosted_exact_retry_avoids_broken_config(mux, monkeypatch):
    from gateway import run
    from gateway.session_contract import SessionActor
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC

    runner, homes, loop, _ = mux
    authority = runner.session_authorities.require(homes['beta'])
    actor = SessionActor('alice', authority.profile_id,
                         frozenset({'session:create', 'session:read'}), 'private-owner')
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='retry', member_id='helper',
                                 profile='beta', principal=actor, authorize=lambda *args: True)
    params = {'profile': 'beta', 'source': 'bot_room', 'title': 'Group: retry'}
    created = rpc.create(**params)

    def broken(*args, **kwargs):
        raise AssertionError('exact retry read config')

    monkeypatch.setattr(run, '_load_gateway_config', broken)
    assert rpc.create(**params) == created


def test_hosted_create_refuses_retirement_during_config(mux, monkeypatch):
    from gateway import run
    from gateway.session_contract import SessionActor
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from hermes_state_runtime import RuntimeStoreError

    runner, homes, loop, _ = mux
    authority = runner.session_authorities.require(homes['beta'])
    actor = SessionActor('alice', authority.profile_id,
                         frozenset({'session:create', 'session:read'}), 'private-owner')
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='retire', member_id='helper',
                                 profile='beta', principal=actor, authorize=lambda *args: True)
    entered, release = threading.Event(), threading.Event()
    real = run._load_gateway_config

    def held_read(*args, **kwargs):
        entered.set()
        if not release.wait(30):
            raise TimeoutError('config test release')
        return real(*args, **kwargs)

    async def retire():
        authority.retiring = True

    monkeypatch.setattr(run, '_load_gateway_config', held_read)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rpc.create, profile='beta', source='bot_room', title='Group: retire')
        try:
            assert entered.wait(20), 'hosted create did not reach config reader'
            asyncio.run_coroutine_threadsafe(retire(), loop).result(15)
        finally:
            release.set()
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            future.result(timeout=30)
    assert authority.db.get_session(rpc.ref.session_id) is None


def test_hosted_timeout_does_not_cancel_creation(mux, monkeypatch):
    from gateway import run
    from gateway.session_contract import SessionActor
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC

    runner, homes, loop, call = mux
    authority = runner.session_authorities.require(homes['beta'])
    actor = SessionActor('alice', authority.profile_id,
                         frozenset({'session:create', 'session:read'}), 'private-owner')
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='timeout', member_id='helper',
                                 profile='beta', principal=actor, authorize=lambda *args: True,
                                 timeout=0.5)
    entered, release = threading.Event(), threading.Event()
    real = run._load_gateway_config

    def held_read(*args, **kwargs):
        entered.set()
        if not release.wait(30):
            raise TimeoutError('config test release')
        return real(*args, **kwargs)

    monkeypatch.setattr(run, '_load_gateway_config', held_read)
    params = {'profile': 'beta', 'source': 'bot_room', 'title': 'Group: timeout'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rpc.create, **params)
        try:
            assert entered.wait(20), 'hosted create did not reach config reader'
            with pytest.raises(concurrent.futures.TimeoutError):
                future.result(timeout=10)
        finally:
            release.set()

    async def wait_for_creation():
        for _ in range(300):
            if authority.db.get_session(rpc.ref.session_id) is not None:
                return
            await asyncio.sleep(0.1)
        raise AssertionError('timed-out create did not commit')

    call(wait_for_creation())
    rpc.timeout = 30
    assert rpc.create(**params)['session_id'] == rpc.ref.session_id


def test_hosted_concurrent_create_converges(mux, monkeypatch):
    from gateway import run
    from gateway.session_contract import SessionActor
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC

    runner, homes, loop, _ = mux
    authority = runner.session_authorities.require(homes['beta'])
    actor = SessionActor('alice', authority.profile_id,
                         frozenset({'session:create', 'session:read'}), 'private-owner')
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='duplicate', member_id='helper',
                                 profile='beta', principal=actor, authorize=lambda *args: True)
    arrived, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    count = 0
    real = run._load_gateway_config

    def held_read(*args, **kwargs):
        nonlocal count
        with lock:
            count += 1
            if count == 2:
                arrived.set()
        if not release.wait(30):
            raise TimeoutError('config test release')
        return real(*args, **kwargs)

    monkeypatch.setattr(run, '_load_gateway_config', held_read)
    params = {'profile': 'beta', 'source': 'bot_room', 'title': 'Group: duplicate'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(rpc.create, **params)
        second = pool.submit(rpc.create, **params)
        try:
            assert arrived.wait(20), 'both creates did not reach config reader'
        finally:
            release.set()
        assert first.result(timeout=30) == second.result(timeout=30)
    assert authority.db.get_session(rpc.ref.session_id) is not None
