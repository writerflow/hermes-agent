"""Read-location regressions for hosted, cron and recovery owner scopes."""
import asyncio
import concurrent.futures
import threading
from pathlib import Path

import pytest

from tests.gateway.test_hosted_mux_runtime import mux  # noqa: F401


@pytest.mark.parametrize('site', ['session', 'config'])
def test_cron_create_other_reads_are_off_loop(mux, monkeypatch, site):
    from cron import jobs
    from gateway import run, session_cron

    runner, homes, loop, call = mux
    authority = runner.session_authorities.require(homes['beta'])
    with jobs.use_cron_store(homes['beta']):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local', model='fixture')
    observed = []
    if site == 'session':
        import inspect
        from gateway.session_local_recovery import local_identity
        from hermes_state_keys import CRON_PRINCIPAL_ID, cron_request_id
        expected_sid = local_identity(authority.profile_id, CRON_PRINCIPAL_ID,
                                      cron_request_id(job['id'], 'fresh'))
        real = authority.db._read_one
        def probe(sql, params=()):
            # Include owner-side identity checks, not just the first scout read.
            # Restore's independent reads are not the identity decision.
            stack = {frame.function for frame in inspect.stack()}
            direct_create_check = '_create' in stack and 'restore_local_session' not in stack
            if (not observed or direct_create_check) and 'FROM sessions' in sql and expected_sid in params:
                observed.append((Path(authority.db.db_path).resolve().parent, threading.get_ident()))
            return real(sql, params)
        monkeypatch.setattr(authority.db, '_read_one', probe)
    else:
        real = run._load_gateway_config
        def config_probe(config_path=None):
            home = Path(config_path).parent if config_path is not None else run._gateway_config_home()
            observed.append((Path(home).resolve(), threading.get_ident()))
            return real(config_path)
        monkeypatch.setattr(run, '_load_gateway_config', config_probe)

    async def loop_thread():
        return threading.get_ident()

    loop_ident = call(loop_thread())
    result = call(session_cron.operation(authority, 'submit', {
        'job_id': job['id'], 'request_id': 'fresh', 'extra_prompt': None}))
    assert result['session_id']
    assert observed, f'cron submit did not reach {site} reader'
    assert all(home == homes['beta'].resolve() for home, _ in observed), observed
    assert all(ident != loop_ident for _, ident in observed), f'cron {site} read on owner loop'


def test_cron_create_exact_retry_reads_identity_off_loop_without_job(mux, monkeypatch):
    from cron import jobs
    from gateway import run, session_cron
    from gateway.session_local_recovery import local_identity
    from hermes_state_keys import CRON_PRINCIPAL_ID, cron_request_id

    runner, homes, loop, call = mux
    authority = runner.session_authorities.require(homes['beta'])
    with jobs.use_cron_store(homes['beta']):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local', model='fixture')
    params = {'job_id': job['id'], 'request_id': 'retry', 'extra_prompt': None}
    original = call(session_cron.operation(authority, 'submit', params))
    sid = local_identity(authority.profile_id, CRON_PRINCIPAL_ID, cron_request_id(job['id'], 'retry'))
    assert original['session_id'] == sid
    reads = []
    real = authority.db._read_one

    def probe(sql, params=()):
        import inspect
        stack = {frame.function for frame in inspect.stack()}
        if '_create' in stack and 'restore_local_session' not in stack and 'FROM sessions' in sql and sid in params:
            reads.append(threading.get_ident())
        return real(sql, params)

    def unexpected(*args, **kwargs):
        raise AssertionError('exact retry read job or config')

    async def loop_thread():
        return threading.get_ident()

    loop_ident = call(loop_thread())
    monkeypatch.setattr(authority.db, '_read_one', probe)
    monkeypatch.setattr(jobs, 'get_job', unexpected)
    monkeypatch.setattr(run, '_load_gateway_config', unexpected)
    retry = call(session_cron.operation(authority, 'submit', params))
    assert retry == original
    assert reads, 'exact retry did not reach session identity read'
    assert all(ident != loop_ident for ident in reads), 'exact retry identity read on owner loop'


def test_cron_create_refuses_drain_during_job_read(mux, monkeypatch):
    from cron import jobs
    from gateway import session_cron
    from gateway.session_local_recovery import local_identity
    from hermes_state_keys import CRON_PRINCIPAL_ID, cron_request_id
    from hermes_state_runtime import RuntimeStoreError

    runner, homes, loop, _ = mux
    authority = runner.session_authorities.require(homes['beta'])
    with jobs.use_cron_store(homes['beta']):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local', model='fixture')
    sid = local_identity(authority.profile_id, CRON_PRINCIPAL_ID, cron_request_id(job['id'], 'drain'))
    entered, release = threading.Event(), threading.Event()
    real = jobs.get_job

    def held_read(job_id):
        entered.set()
        if not release.wait(30):
            raise TimeoutError('cron job test release')
        return real(job_id)

    monkeypatch.setattr(jobs, 'get_job', held_read)
    future = asyncio.run_coroutine_threadsafe(session_cron.operation(authority, 'submit', {
        'job_id': job['id'], 'request_id': 'drain', 'extra_prompt': None}), loop)
    try:
        assert entered.wait(20), 'cron submit did not reach job reader'
        runner._draining = True
    finally:
        release.set()
    try:
        with pytest.raises(RuntimeStoreError, match='runtime_draining'):
            future.result(timeout=30)
        assert authority.db.get_session(sid) is None, 'draining attempt created a session'
    finally:
        runner._draining = False


def test_cron_create_job_read_is_off_loop(mux, monkeypatch):
    from cron import jobs
    from gateway import session_cron

    runner, homes, loop, call = mux
    authority = runner.session_authorities.require(homes['beta'])
    with jobs.use_cron_store(homes['beta']):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local', model='fixture')
    observed = []
    real = jobs.get_job

    def probe(job_id):
        observed.append((jobs._current_cron_store().cron_dir.resolve(), threading.get_ident()))
        return real(job_id)

    async def loop_thread():
        return threading.get_ident()

    loop_ident = call(loop_thread())
    monkeypatch.setattr(jobs, 'get_job', probe)
    result = call(session_cron.operation(authority, 'submit', {
        'job_id': job['id'], 'request_id': 'fresh', 'extra_prompt': None}))
    assert authority.db.get_session(result['session_id']) is not None
    assert observed, 'cron submit did not reach jobs reader'
    assert all(path == (homes['beta'] / 'cron').resolve() for path, _ in observed), observed
    assert all(ident != loop_ident for _, ident in observed), 'cron job read on owner loop'


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
