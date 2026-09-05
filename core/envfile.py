"""Reading `.env` into the process — MASTER_PLAN §21.

`Settings` loads `.env` through pydantic, but only for the fields it declares.
Anything read with a plain `os.environ` lookup — the Mongo URI, the credential
encryption key — would miss it and report "not configured" while the value sat
in the file, which is a confusing way to spend an afternoon.

Loaded once, and never over the top of a variable the process was actually
started with: an explicit environment beats a file, always.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_env_file"]

#: Modification time of the file as last read. `None` means never.
#:
#: A plain "already loaded" flag latched forever, so a process that started
#: before a value was added to `.env` could never see it — the console reported
#: NEUTRON_SECRET_KEY as unset while the file plainly contained it, and the only
#: cure was a restart nobody knew to perform. Keyed on mtime instead, so editing
#: the file is noticed by anything still running.
_LOADED_MTIME: float | None = None


def _env_path(path: Path | None = None) -> Path:
    return path or Path(__file__).resolve().parent.parent / ".env"


def load_env_file(path: Path | None = None) -> None:
    """Apply `.env` to the environment. Cheap to call, and re-reads on change.

    Never overrides a variable the process was started with: an explicit
    environment beats a file, always.
    """
    global _LOADED_MTIME  # noqa: PLW0603 - one cache for a process-wide side effect

    env_file = _env_path(path)
    try:
        mtime = env_file.stat().st_mtime
    except OSError:
        return
    if mtime == _LOADED_MTIME:
        return
    _LOADED_MTIME = mtime

    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            # `setdefault`, so a variable exported into the process wins over
            # the file. A deployment that pins something explicitly should not
            # have it quietly replaced.
            os.environ.setdefault(key.strip(), value.strip())
    except OSError:
        return
