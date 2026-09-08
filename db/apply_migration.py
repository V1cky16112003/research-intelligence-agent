"""Apply a .sql migration file to DATABASE_URL without needing the psql client.

The documented way to run the files in db/migrations/ is `psql -f`. That assumes
a local libpq install, which the dev machine does not have. This runner uses the
psycopg driver the app already depends on, so a migration can be applied from
anywhere the app itself can run.

Statements are executed one at a time in autocommit mode rather than as one
multi-statement string, for two reasons: VACUUM cannot run inside a transaction
block (and 003 ends with one), and a failure part-way through leaves the earlier
statements committed instead of rolling back into a state where, say, the HNSW
index has been dropped and not rebuilt.

Usage:
    PYTHONPATH=. python3 -m db.apply_migration db/migrations/003_halfvec_embeddings.sql
    PYTHONPATH=. python3 -m db.apply_migration <file> --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_STATEMENT_TIMEOUT = "30min"


def split_statements(sql: str) -> list[str]:
    """Split a migration file into top-level statements.

    Semicolons inside single-quoted strings and inside dollar-quoted bodies
    ($$ ... $$, used by 001's DO block) do not terminate a statement.
    """
    statements: list[str] = []
    buf: list[str] = []
    i, n = 0, len(sql)
    in_single = False
    dollar_tag: str | None = None

    while i < n:
        ch = sql[i]

        if dollar_tag:
            if sql.startswith(dollar_tag, i):
                buf.append(dollar_tag)
                i += len(dollar_tag)
                dollar_tag = None
                continue
            buf.append(ch)
            i += 1
            continue

        if in_single:
            buf.append(ch)
            if ch == "'":
                in_single = False
            i += 1
            continue

        if ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
            continue

        if ch == "$":
            end = sql.find("$", i + 1)
            # A dollar quote tag is $$ or $tag$ where tag is an identifier.
            if end != -1 and (end == i + 1 or sql[i + 1:end].replace("_", "").isalnum()):
                dollar_tag = sql[i:end + 1]
                buf.append(dollar_tag)
                i = end + 1
                continue

        if ch == "-" and sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end == -1 else end
            continue

        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _first_line(stmt: str) -> str:
    line = " ".join(stmt.split())
    return line if len(line) <= 90 else line[:87] + "..."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="path to the .sql migration file")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the statements that would run, touch nothing")
    parser.add_argument("--statement-timeout", default=DEFAULT_STATEMENT_TIMEOUT,
                        help=f"per-statement timeout (default: {DEFAULT_STATEMENT_TIMEOUT})")
    args = parser.parse_args(argv)

    with open(args.path, encoding="utf-8") as fh:
        statements = split_statements(fh.read())

    if not statements:
        print(f"No statements found in {args.path}")
        return 1

    print(f"{args.path}: {len(statements)} statement(s)")
    for idx, stmt in enumerate(statements, 1):
        print(f"  {idx}. {_first_line(stmt)}")

    if args.dry_run:
        print("\nDry run — nothing executed. Re-run without --dry-run to apply.")
        return 0

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        # Fall back to .env so the connection string never has to be pasted onto
        # the command line, where it would land in shell history and logs.
        try:
            from dotenv import load_dotenv
        except ImportError:
            load_dotenv = None
        if load_dotenv is not None:
            load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
            dsn = os.getenv("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set and was not found in .env", file=sys.stderr)
        return 2

    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(f"SET statement_timeout = '{args.statement_timeout}'")
        for idx, stmt in enumerate(statements, 1):
            started = time.monotonic()
            print(f"\n-> [{idx}/{len(statements)}] {_first_line(stmt)}", flush=True)
            conn.execute(stmt)
            print(f"   ok ({time.monotonic() - started:.1f}s)", flush=True)

    print(f"\nApplied {len(statements)} statement(s) from {args.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
