"""Generate the key that protects stored credentials — MASTER_PLAN §21.

    python -m apps.cli.vault --generate
    python -m apps.cli.vault --check

**The key does not live in the database it protects.** It goes in `.env`, so a
dumped collection — a backup, a replica, a screenshot — is useless on its own.
That also means losing the key loses the stored credentials, which is the right
trade: a key kept beside what it protects protects nothing.
"""

from __future__ import annotations

import argparse
import sys

from core.vault import KEY_VARIABLE, generate_key, vault_configured

__all__ = ["run"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the credential encryption key")
    parser.add_argument("--generate", action="store_true", help="Print a fresh key for .env")
    parser.add_argument("--check", action="store_true", help="Report whether a key is configured")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.generate:
        print("\nAdd this line to .env, then restart the API:\n")
        print(f"  {KEY_VARIABLE}={generate_key()}\n")
        print("Keep it. Credentials encrypted with this key cannot be read without it.")
        return 0

    if args.check:
        if vault_configured():
            print(f"{KEY_VARIABLE} is set — broker credentials can be stored encrypted.")
            return 0
        print(f"{KEY_VARIABLE} is not set. Storing broker credentials is refused until it is.")
        print("Generate one with: python -m apps.cli.vault --generate")
        return 1

    print("give --generate to make a key, or --check to test one")
    return 1


if __name__ == "__main__":
    sys.exit(run())
