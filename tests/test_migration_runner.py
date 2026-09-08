"""Tests for db/apply_migration.py's SQL splitter.

The splitter is the only non-trivial part of the runner: everything else is a
loop over psycopg. What it must not do is cut a statement in half at a semicolon
that lives inside a string literal or a dollar-quoted body — 001 ships a DO $$
block whose body contains several.
"""
from pathlib import Path

import pytest

from db.apply_migration import split_statements

MIGRATIONS = Path(__file__).resolve().parents[1] / "db" / "migrations"


def test_splits_plain_statements():
    sql = "DROP INDEX IF EXISTS a;\nCREATE INDEX a ON t (c);\n"
    assert split_statements(sql) == [
        "DROP INDEX IF EXISTS a",
        "CREATE INDEX a ON t (c)",
    ]


def test_strips_comments_and_blank_statements():
    sql = "-- a comment; with a semicolon\nSELECT 1;\n\n;\n-- trailing\n"
    assert split_statements(sql) == ["SELECT 1"]


def test_semicolon_inside_string_literal_does_not_split():
    sql = "INSERT INTO t (c) VALUES ('a;b');"
    assert split_statements(sql) == ["INSERT INTO t (c) VALUES ('a;b')"]


def test_dollar_quoted_body_stays_one_statement():
    sql = (
        "DO $$ BEGIN\n"
        "  IF NOT EXISTS (SELECT 1) THEN\n"
        "    ALTER TABLE t ADD COLUMN c TEXT;\n"
        "  END IF;\n"
        "END $$;\n"
        "SELECT 1;\n"
    )
    statements = split_statements(sql)
    assert len(statements) == 2
    assert statements[0].startswith("DO $$")
    assert statements[0].endswith("$$")
    assert statements[1] == "SELECT 1"


def test_tagged_dollar_quote_stays_one_statement():
    sql = "DO $body$ SELECT 'x;y'; $body$;"
    assert len(split_statements(sql)) == 1


@pytest.mark.parametrize("name,expected", [
    ("001_contextual_retrieval.sql", 3),
    ("002_llm_call_log.sql", 6),
    ("003_halfvec_embeddings.sql", 4),
])
def test_real_migrations_split_into_expected_statement_counts(name, expected):
    sql = (MIGRATIONS / name).read_text(encoding="utf-8")
    assert len(split_statements(sql)) == expected


def test_halfvec_migration_drops_index_before_rewriting_the_column():
    """Ordering is load-bearing: the DROP frees the headroom the rewrite needs."""
    statements = split_statements(
        (MIGRATIONS / "003_halfvec_embeddings.sql").read_text(encoding="utf-8")
    )
    assert statements[0].upper().startswith("DROP INDEX")
    assert "halfvec(768)" in statements[1]
    assert "halfvec_cosine_ops" in statements[2]
