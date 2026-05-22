"""Subprocess spawn helpers shared across docgraph.

The host runs without a console (telecode launches `docgraph host` with
`CREATE_NO_WINDOW`). Any console-subsystem child we shell out to — `git`
above all — would otherwise get a fresh console allocated and torn down
immediately, producing a visible terminal flash on every call. Passing
`CREATE_NO_WINDOW` suppresses that. No-op (0) off Windows.
"""
from __future__ import annotations

import sys

# CREATE_NO_WINDOW on Windows; 0 (ignored) elsewhere so call sites can pass
# `creationflags=NO_WINDOW` unconditionally.
NO_WINDOW: int = 0x08000000 if sys.platform.startswith("win") else 0
