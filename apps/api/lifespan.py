"""What the API does before it answers anything — MASTER_PLAN §13.6.

Split from `main` because starting a process and defining its routes are
different jobs, and the first one is where a slow dependency shows up.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

__all__ = ["warm_backends"]


#: uvicorn's own logger, not this module's.
#:
#: A module logger propagates to root, which uvicorn does not configure, so
#: every message from it is dropped — and a warm-up that reports neither
#: success nor failure is indistinguishable from one that never ran. This is
#: the stream the operator is already reading.
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def warm_backends(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001 - lifespan signature
    """Open the account database before the first request needs it.

    The first connection to Atlas costs seconds — DNS and SRV resolution, TCP,
    TLS, SCRAM and a verifying ping — and it is paid by whoever triggers it.
    Left to the first request, that is the operator opening the console, who
    sees a blank card for five seconds on a system that is otherwise answering
    in milliseconds.

    **On a thread, so the port binds immediately.** A synchronous warm here
    would move the delay rather than remove it: uvicorn would not accept
    connections until it finished, and a console that refuses to connect for
    five seconds is worse than one that is briefly slow. The request path is
    unchanged either way — it opens the cached client if this has finished, and
    opens its own if it has not.

    Failure is logged and not raised. An unreachable database is a state this
    API is built to render, and refusing to start would remove the screen that
    would have said so.
    """

    def warm() -> None:
        started = time.perf_counter()
        try:
            from ops.mongo import mongo_available, warm_mongo_client  # noqa: PLC0415

            if not mongo_available():
                return
            ok = warm_mongo_client()
            logger.info(
                "account database %s in %.2fs",
                "ready" if ok else "unreachable",
                time.perf_counter() - started,
            )
        except Exception:
            logger.exception("could not warm the account database")

    threading.Thread(target=warm, name="warm-accounts", daemon=True).start()
    yield
