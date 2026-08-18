"""Dataset versioning — MASTER_PLAN §M2, §M3 gate.

An experiment records *which data it ran on*. Without that, "this backtest is
reproducible" is a claim rather than a fact: the lake gets backfilled, a
corrected bhavcopy replaces a bad one, and last month's Sharpe quietly becomes
unreproducible with no error anywhere.

A version is the content hash of the actual bytes on disk. Recompute it and you
either get the same hash — in which case the data is identical — or you do not,
in which case the experiment must be re-run rather than trusted.

Hashing is over file contents, not mtimes or paths, so copying the lake to
another machine preserves versions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.clock import utc_now

__all__ = ["CHUNK_BYTES", "DatasetVersion", "compute_version"]

#: 1 MiB read chunks — large enough to be fast, small enough to bound memory.
CHUNK_BYTES = 1024 * 1024

#: Bump when the hashing scheme changes, so old versions are never silently
#: compared against new ones.
HASH_SCHEME = "sha256-v1"


@dataclass(frozen=True)
class DatasetVersion:
    """An immutable fingerprint of a set of files."""

    content_hash: str
    file_count: int
    total_bytes: int
    computed_at: datetime
    scheme: str = HASH_SCHEME
    #: Relative path -> per-file digest. Makes "which file changed?" answerable.
    files: dict[str, str] = field(default_factory=dict)

    @property
    def short(self) -> str:
        return self.content_hash[:12]

    def differs_from(self, other: DatasetVersion) -> list[str]:
        """Relative paths that changed, were added, or were removed."""
        mine, theirs = self.files, other.files
        changed = [p for p in mine.keys() & theirs.keys() if mine[p] != theirs[p]]
        added = list(mine.keys() - theirs.keys())
        removed = list(theirs.keys() - mine.keys())
        return sorted(changed + added + removed)

    def to_json(self) -> str:
        return json.dumps(
            {
                "content_hash": self.content_hash,
                "scheme": self.scheme,
                "file_count": self.file_count,
                "total_bytes": self.total_bytes,
                "computed_at": self.computed_at.isoformat(),
                "files": self.files,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> DatasetVersion:
        data = json.loads(payload)
        return cls(
            content_hash=data["content_hash"],
            file_count=data["file_count"],
            total_bytes=data["total_bytes"],
            computed_at=datetime.fromisoformat(data["computed_at"]),
            scheme=data.get("scheme", HASH_SCHEME),
            files=data.get("files", {}),
        )


def _digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def compute_version(root: Path, pattern: str = "**/*.parquet") -> DatasetVersion:
    """Fingerprint every file under `root` matching `pattern`.

    The combined hash folds in each file's *relative path* as well as its
    contents, so moving a file between partitions changes the version. That is
    intended: the partition layout is part of what a reader depends on.

    Raises:
        FileNotFoundError: if `root` does not exist or matches nothing. An
            empty dataset must not silently produce a valid-looking version.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    paths = sorted(p for p in root.glob(pattern) if p.is_file())
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} under {root}")

    combined = hashlib.sha256()
    files: dict[str, str] = {}
    total = 0

    # Sorted order makes the combined hash independent of filesystem iteration
    # order, which differs between platforms (§14.1.1).
    for path in paths:
        file_hash, size = _digest_file(path)
        relative = path.relative_to(root).as_posix()
        files[relative] = file_hash
        total += size
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(file_hash.encode("ascii"))
        combined.update(b"\0")

    return DatasetVersion(
        content_hash=combined.hexdigest(),
        file_count=len(paths),
        total_bytes=total,
        computed_at=utc_now(),
        files=files,
    )
