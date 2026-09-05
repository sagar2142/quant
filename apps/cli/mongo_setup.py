"""Create the Neutron database in MongoDB — MASTER_PLAN §13.7.

    python -m apps.cli.mongo_setup --check
    python -m apps.cli.mongo_setup --create

**Mongo creates a database when something is first written to it**, so
"creating" one means creating its collections and, more importantly, its
indexes. The indexes are the part that matters: the unique index on the
lowercased email is what makes two accounts with the same address impossible
rather than merely unlikely, and the TTL index on sessions is what makes an
expired session actually disappear instead of lingering until something
happens to notice.

Postgres enforced both of those with a UNIQUE constraint and an explicit
expiry check. Mongo will not do it unless it is asked, so it is asked here, and
the check command reports whether it was.
"""

from __future__ import annotations

import argparse
import sys

from ops.mongo import DATABASE_NAME, describe_uri, mongo_available, mongo_database

__all__ = ["run"]

RULE = "─" * 78

#: Collections this system uses, and why each exists.
COLLECTIONS = {
    "users": "operator accounts and their preferences",
    "sessions": "signed-in sessions, keyed by a hash of the token",
}


def create() -> int:
    """Create the collections and indexes. Safe to run repeatedly."""
    with mongo_database() as database:
        if database is None:
            print("MongoDB is not configured or not reachable.")
            print("Set MONGODB_URI in .env, then run this again.")
            return 1

        existing = set(database.list_collection_names())  # type: ignore[attr-defined]
        for name, purpose in COLLECTIONS.items():
            if name in existing:
                print(f"  {name:<10} already present — {purpose}")
                continue
            database.create_collection(name)  # type: ignore[attr-defined]
            print(f"  {name:<10} created — {purpose}")

        users = database["users"]  # type: ignore[index]
        sessions = database["sessions"]  # type: ignore[index]

        # Unique on the lowercased email. Without it two accounts can hold the
        # same address, and "which one did I sign in to" has no answer.
        users.create_index("email_lower", unique=True, name="users_email_unique")

        # Sessions expire. A TTL index makes Mongo remove them rather than
        # leaving the check to whatever happens to read the row next — an
        # expired session that survives because nobody looked is a session
        # that still works.
        sessions.create_index("expires_at", expireAfterSeconds=0, name="sessions_ttl")
        sessions.create_index("user_id", name="sessions_by_user")

        print("\n  indexes:")
        for collection in (users, sessions):
            for index in collection.list_indexes():
                keys = ", ".join(index["key"].keys())
                unique = " unique" if index.get("unique") else ""
                ttl = " ttl" if "expireAfterSeconds" in index else ""
                print(f"    {collection.name}.{index['name']:<22} ({keys}){unique}{ttl}")
    return 0


def check() -> int:
    """Report what is there, without changing anything."""
    print(f"  uri       {describe_uri()}")
    print(f"  database  {DATABASE_NAME}")

    if not mongo_available():
        print("\n  MONGODB_URI is not set. Put it in .env:")
        # A generic host, not this install's. The example is committed and
        # the repository is public; naming the actual cluster would put an
        # address someone can point a login attempt at into it, for no gain
        # to the person reading the help.
        print("    MONGODB_URI=mongodb+srv://USER:PASSWORD@YOUR-CLUSTER.mongodb.net/")
        return 1

    with mongo_database() as database:
        if database is None:
            print("\n  Configured, but the cluster did not answer.")
            print("  Check the password, and that this machine's IP is allowed in Atlas.")
            return 1

        names = sorted(database.list_collection_names())  # type: ignore[attr-defined]
        print(f"  reachable, {len(names)} collection(s): {', '.join(names) or 'none yet'}")
        for name in COLLECTIONS:
            if name in names:
                count = database[name].count_documents({})  # type: ignore[index]
                print(f"    {name:<10} {count} document(s)")
            else:
                print(f"    {name:<10} missing — run --create")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or inspect the Neutron Mongo database")
    parser.add_argument("--create", action="store_true", help="Create collections and indexes")
    parser.add_argument("--check", action="store_true", help="Report what is there")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.create or args.check):
        print("give --check to inspect, or --create to set it up")
        return 1

    print()
    print(RULE)
    print(f"MONGODB — {DATABASE_NAME}")
    print(RULE)
    return create() if args.create else check()


if __name__ == "__main__":
    sys.exit(run())
