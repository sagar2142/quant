"""Command-line entry points.

**Console encoding is forced to UTF-8 here, once, for every CLI.** Windows
consoles default to cp1252, and every command in this package prints section
marks (§), em dashes and box rules — so `python -m apps.cli.readiness` died with
a UnicodeEncodeError partway through its own output. That is a bad failure for
any command and an unacceptable one for the pre-live checklist, which is
supposed to be the last thing you read before risking money.

Fixed at import of the package rather than in each module: a fix that has to be
remembered in every new CLI is a fix that will be forgotten in one of them.

`errors="replace"` rather than `"strict"`: on a console that genuinely cannot
render a glyph, a substituted character is a cosmetic flaw, while a traceback
loses the whole report.
"""

from __future__ import annotations

import contextlib
import sys


def _force_utf8() -> None:
    """Re-encode stdout and stderr as UTF-8 where the stream allows it.

    Guarded by `hasattr`: pytest and some runners replace these streams with
    objects that have no `reconfigure`, and a crash while setting up output
    formatting would be a worse bug than the one this prevents.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # A stream that refuses re-encoding still works; it just prints
        # replacement characters. Never fatal, so never raised.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


_force_utf8()
