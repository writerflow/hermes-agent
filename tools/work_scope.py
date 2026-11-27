"""Internal systemd work-scope policy and readback helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path

SYSTEMD = "org.freedesktop.systemd1"
MANAGER = SYSTEMD + ".Manager"
MANAGER_PATH = "/org/freedesktop/systemd1"
UNIT = SYSTEMD + ".Unit"
SCOPE = SYSTEMD + ".Scope"
OOM_POLICY = "continue"
_CGROUP_ROOT = Path("/sys/fs/cgroup")


class ScopeVerificationError(RuntimeError):
    """A systemd scope could not be verified; do not release the held work."""


class ScopeProbeFailure(Enum):
    UNVERIFIED_POLICY = "scope probe could not verify OOMPolicy=continue"


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
        if self.property("s", path, SCOPE, "ControlGroup") != membership:
            raise ScopeVerificationError("systemd scope membership did not match")
        if self.property("s", path, UNIT, "ActiveState") != "active":
            raise ScopeVerificationError("systemd scope was not active")
        return path, bytes(invocation)

    def verify_scope(self, name: str, membership: str, *, expected_invocation: str | None = None) -> None:
        """Check the current process and the manager against the same cgroup."""
        parts = membership.split("/")
        if parts[0] != "" or any(part in {"", ".", ".."} for part in parts[1:]) or parts[-1] != name:
            raise ScopeVerificationError("invalid cgroup path for scope policy readback")
        directory = _CGROUP_ROOT.joinpath(*parts[1:])
        # Pin the directory while reading the unit. The invocation and its
        # path text alone cannot distinguish a replaced kernel directory.
        try:
            fd = os.open(directory, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                path, invocation = self.scope_identity(name, membership)
                if expected_invocation is not None and invocation.hex() != expected_invocation:
                    raise ScopeVerificationError("systemd scope invocation did not match the launcher")
                if self.property("s", path, SCOPE, "OOMPolicy") != OOM_POLICY:
                    raise ScopeVerificationError("systemd scope OOMPolicy did not equal continue")
                if self.scope_identity(name, membership) != (path, invocation):
                    raise ScopeVerificationError("systemd scope invocation changed during OOMPolicy verification")
                self.verify_owner()
                if not os.path.samestat(os.fstat(fd), directory.stat(follow_symlinks=False)):
                    raise ScopeVerificationError("systemd scope cgroup changed during OOMPolicy verification")
                _verify_self_in_cgroup(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            raise ScopeVerificationError("systemd scope cgroup identity was unreadable") from exc


def _verify_self_in_cgroup(directory_fd: int) -> None:
    # A path string from /proc/self/cgroup doesn't establish which directory
    # contained us. Read membership relative to the object we verified.
    fd = os.open("cgroup.procs", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory_fd)
    try:
        pid = f"{os.getpid()}\n".encode("ascii")
        with open(fd, "rb", closefd=False) as members:
            if not any(line == pid for line in members):
                raise ScopeVerificationError("current process was not in the verified scope")
    finally:
        os.close(fd)


def probe_payload(name: str) -> list[str]:
    return [sys.executable, "-I", str(Path(__file__).resolve()), name]


def scoped_payload(name: str, command: list[str]) -> list[str]:
    return [*probe_payload(name), *command]


def _verify_scope_launch(name: str, *, expected_invocation: str | None = None) -> None:
    lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    membership = next((line[3:] for line in lines if line.startswith("0::")), "")
    ScopeBus("--user", time.monotonic() + 2).verify_scope(
        name, membership, expected_invocation=expected_invocation,
    )


def _run_scope_launcher(name: str, command: list[str]) -> int:
    try:
        # systemd-run v255 waits for the create job and passes the invocation
        # ID of its own unit. The caller clears any inherited ID before launch.
        _verify_scope_launch(name, expected_invocation=os.environ.get("INVOCATION_ID", "") if command else None)
        if command:
            os.execvpe(command[0], command, os.environ)  # windows-footgun: ok — Linux systemd entry point
    except (ScopeVerificationError, OSError):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_run_scope_launcher(sys.argv[1], sys.argv[2:]) if len(sys.argv) >= 2 else 1)
