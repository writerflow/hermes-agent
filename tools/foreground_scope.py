"""Foreground terminal commands in their own capped ``hermes-worker-fg-*`` systemd scope (seam)."""
from __future__ import annotations


def foreground_scope_host() -> bool:
    return False
