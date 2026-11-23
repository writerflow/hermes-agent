"""Read-location regressions for hosted, cron and recovery owner scopes."""
import asyncio
import threading
from pathlib import Path

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
