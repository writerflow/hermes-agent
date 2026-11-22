"""Internal systemd work-scope policy and readback helpers."""

from __future__ import annotations

import json
import subprocess
import time

SYSTEMD = "org.freedesktop.systemd1"
MANAGER = SYSTEMD + ".Manager"
MANAGER_PATH = "/org/freedesktop/systemd1"
UNIT = SYSTEMD + ".Unit"
SCOPE = SYSTEMD + ".Scope"
OOM_POLICY = "continue"


class ScopeVerificationError(RuntimeError):
    """A systemd scope could not be verified; do not release the held work."""


class ScopeBus:
    """Bounded reads on one selected bus, addressing the manager by its unique name."""

    def __init__(self, route: str, deadline: float):
        self.route = route
        self.deadline = deadline
        self.bus_id = self._bus_call("GetId")
        self.owner = self._bus_call("GetNameOwner", "s", SYSTEMD)
        if not isinstance(self.owner, str) or not self.owner.startswith(":"):
            raise ScopeVerificationError("systemd did not have a unique bus owner")
        if not isinstance(self.bus_id, str) or not self.bus_id:
            raise ScopeVerificationError("systemd bus identity was unreadable")

    def _read(self, signature: str, *args: str):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ScopeVerificationError("systemd scope verification timed out")
        try:
            result = subprocess.run(
                ["busctl", self.route, "--json=short", *args],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=remaining,
                check=False,
            )
            reply = json.loads(result.stdout) if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise ScopeVerificationError("systemd reply was unreadable") from exc
        if not isinstance(reply, dict) or reply.get("type") != signature or "data" not in reply:
            raise ScopeVerificationError("systemd reply was refused or malformed")
        return reply["data"]

    def _call(self, signature: str, *args: str):
        data = self._read(signature, "call", *args)
        if not isinstance(data, list) or len(data) != 1:
            raise ScopeVerificationError("systemd method reply did not have one value")
        return data[0]

    def _bus_call(self, method: str, *args: str):
        return self._call("s", "org.freedesktop.DBus", "/org/freedesktop/DBus",
                          "org.freedesktop.DBus", method, *args)

    def manager_call(self, signature: str, method: str, *args: str):
        return self._call(signature, self.owner, MANAGER_PATH, MANAGER, method, *args)

    def property(self, signature: str, path: str, interface: str, name: str):
        return self._read(signature, "get-property", self.owner, path, interface, name)

    def verify_owner(self):
        if self._bus_call("GetId") != self.bus_id or self._bus_call("GetNameOwner", "s", SYSTEMD) != self.owner:
            raise ScopeVerificationError("systemd manager changed during scope verification")

    def scope_identity(self, name: str, membership: str) -> tuple[str, bytes]:
        path = self.manager_call("o", "GetUnit", "s", name)
        if not isinstance(path, str) or not path.startswith(MANAGER_PATH + "/unit/"):
            raise ScopeVerificationError("systemd scope object path was invalid")
        invocation = self.property("ay", path, UNIT, "InvocationID")
        if (not isinstance(invocation, list) or len(invocation) != 16 or not any(invocation)
                or any(type(n) is not int or not 0 <= n <= 255 for n in invocation)):
            raise ScopeVerificationError("systemd scope invocation was unreadable")
        if self.property("s", path, UNIT, "ControlGroup") != membership:
            raise ScopeVerificationError("systemd scope membership did not match")
        if self.property("s", path, UNIT, "ActiveState") != "active":
            raise ScopeVerificationError("systemd scope was not active")
        return path, bytes(invocation)

    def verify_scope(self, name: str, membership: str):
        path, invocation = self.scope_identity(name, membership)
        if self.property("s", path, SCOPE, "OOMPolicy") != OOM_POLICY:
            raise ScopeVerificationError("systemd scope OOMPolicy did not equal continue")
        if self.scope_identity(name, membership) != (path, invocation):
            raise ScopeVerificationError("systemd scope invocation changed during OOMPolicy verification")
        self.verify_owner()
