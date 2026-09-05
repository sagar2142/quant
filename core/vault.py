"""Encrypting credentials at rest — MASTER_PLAN §21.

**This exists because broker keys can move money.** Everything else this system
stores is a price or a verdict; these are the one class of value where a stolen
database row is a stolen account. So they are encrypted before they are
written, with a key that is not in the database.

**The key lives in the environment, deliberately.** `NEUTRON_SECRET_KEY` sits
in `.env` beside the other secrets, which means a dumped database — a backup,
a replica, a screenshot of a collection — is useless on its own. It also means
losing that key loses the stored credentials, which is the correct trade: the
alternative is a key stored next to what it protects, which protects nothing.

**Fernet, not something invented here.** AES-128-CBC with an HMAC-SHA256
authentication tag and a random IV per message, from the `cryptography`
library. Authenticated, so a tampered ciphertext fails loudly rather than
decrypting to something else; and versioned, so the format can change later.

A key derived from a passphrase uses scrypt with a fixed salt. A fixed salt is
normally wrong — it is right here because the derivation must be reproducible
across restarts from the passphrase alone, and the passphrase is not a user
password chosen from a small space but a machine secret.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass

from core.envfile import load_env_file

__all__ = ["VaultError", "VaultUnavailableError", "decrypt", "encrypt", "vault_configured"]

#: Environment variable holding the key, either a Fernet key or a passphrase.
KEY_VARIABLE = "NEUTRON_SECRET_KEY"

#: Salt for deriving a key from a passphrase. Fixed on purpose — see the module
#: docstring. Not a secret; its job is domain separation, not surprise.
_DERIVATION_SALT = b"neutron.vault.v1"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1

#: A Fernet key is 32 bytes, urlsafe-base64 encoded.
_FERNET_KEY_BYTES = 32


class VaultError(RuntimeError):
    """A credential could not be encrypted or decrypted."""


class VaultUnavailableError(VaultError):
    """No key is configured, so nothing can be stored.

    Raised rather than falling back to plaintext. A system that silently stored
    an unencrypted broker key when the key was missing would be at its least
    protected exactly when nobody was watching.
    """


@dataclass(frozen=True)
class Vault:
    """Encrypt and decrypt with the configured key."""

    key: bytes

    def _fernet(self) -> object:
        from cryptography.fernet import Fernet  # noqa: PLC0415 - optional dependency

        return Fernet(self.key)

    def encrypt(self, plaintext: str) -> str:
        if not plaintext:
            return ""
        token = self._fernet().encrypt(plaintext.encode("utf-8"))  # type: ignore[attr-defined]
        return str(token.decode("ascii"))

    def decrypt(self, token: str) -> str:
        if not token:
            return ""
        from cryptography.fernet import InvalidToken  # noqa: PLC0415

        try:
            plain = self._fernet().decrypt(token.encode("ascii"))  # type: ignore[attr-defined]
        except InvalidToken as exc:
            # Authentication failed: either the key changed or the ciphertext
            # was altered. Both are refusals, never a best-effort guess.
            raise VaultError(
                "stored credential could not be decrypted — the key has changed, "
                "or the value was tampered with"
            ) from exc
        return str(plain.decode("utf-8"))


def _normalise(raw: str) -> bytes:
    """Accept either a real Fernet key or a passphrase.

    A 44-character urlsafe-base64 string is used as-is; anything else is run
    through scrypt to produce one. Both are supported because a generated key
    is better and a passphrase is what people actually paste.
    """
    candidate = raw.strip()
    try:
        decoded = base64.urlsafe_b64decode(candidate)
        if len(decoded) == _FERNET_KEY_BYTES:
            return candidate.encode("ascii")
    except (ValueError, TypeError):
        pass

    derived = hashlib.scrypt(
        candidate.encode("utf-8"),
        salt=_DERIVATION_SALT,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=32,
        maxmem=64 * 1024 * 1024,
    )
    return base64.urlsafe_b64encode(derived)


def _configured_key() -> bytes | None:
    load_env_file()
    raw = os.environ.get(KEY_VARIABLE, "").strip()
    return _normalise(raw) if raw else None


def vault_configured() -> bool:
    """Whether credentials can be stored at all."""
    return _configured_key() is not None


def _vault() -> Vault:
    key = _configured_key()
    if key is None:
        raise VaultUnavailableError(
            f"{KEY_VARIABLE} is not set, so broker credentials cannot be stored. "
            "Generate one with: python -m apps.cli.vault --generate"
        )
    return Vault(key)


def encrypt(plaintext: str) -> str:
    """Encrypt a credential for storage."""
    return _vault().encrypt(plaintext)


def decrypt(token: str) -> str:
    """Decrypt a stored credential."""
    return _vault().decrypt(token)


def generate_key() -> str:
    """A fresh Fernet key, for putting in `.env`."""
    from cryptography.fernet import Fernet  # noqa: PLC0415

    return str(Fernet.generate_key().decode("ascii"))
