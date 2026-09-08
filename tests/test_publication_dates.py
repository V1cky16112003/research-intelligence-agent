from __future__ import annotations

"""
Regression tests for the publication-date bug.

`papers.published_at` was loaded from the snapshot's `update_date` — the day the OAI
*metadata record* was last touched. Measured against the 50k-paper corpus in Neon,
that put 27.7% of papers in the wrong year and stamped 4,286 of them 2019-2026, years
in which this corpus (arXiv IDs 0704-1805) published nothing at all. Because
`papers_by_month` sorted by month descending, the rows it returned came entirely from
that phantom range: every temporal answer the agent gave described months in which
none of its papers were published.

The dates in the fixtures below are the real values for arXiv 0704.0001.
"""
from datetime import datetime, timezone

from ingestion.loader import parse_published_at, parse_record, parse_updated_at

REAL_RECORD = {
    "id": "0704.0001",
    "title": "Calculation of prompt diphoton production cross sections",
    "abstract": "  A fully differential calculation...  ",
    "categories": "hep-ph",
    "authors_parsed": [["Balazs", "C.", ""], ["Berger", "E. L.", ""]],
    "versions": [
        {"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"},
        {"version": "v2", "created": "Tue, 24 Jul 2007 20:10:27 GMT"},
    ],
    "update_date": "2008-11-26",
}


def test_published_at_comes_from_v1_not_update_date():
    published = parse_published_at(REAL_RECORD)
    assert published.year == 2007 and published.month == 4
    # The exact failure mode: update_date is a year and a half later.
    assert published.year != 2008


def test_published_at_ignores_later_versions():
    """v2 is a revision, not a publication — the paper appeared in April."""
    assert parse_published_at(REAL_RECORD).month == 4


def test_updated_at_carries_update_date():
    assert parse_updated_at(REAL_RECORD) == datetime(2008, 11, 26, tzinfo=timezone.utc)


def test_falls_back_to_arxiv_id_yymm_when_versions_missing():
    """New-style IDs encode the v1 submission month, so a missing `versions` list
    still yields a real date rather than silently falling back to update_date."""
    rec = {"id": "1505.04597", "update_date": "2021-06-11", "categories": "cs.CV"}
    published = parse_published_at(rec)
    assert (published.year, published.month) == (2015, 5)


def test_returns_none_when_no_date_is_recoverable():
    assert parse_published_at({"id": "hep-th/9901001"}) is None


def test_malformed_v1_timestamp_falls_back_rather_than_raising():
    rec = {"id": "1505.04597", "versions": [{"version": "v1", "created": "not a date"}]}
    assert parse_published_at(rec).year == 2015


def test_parse_record_wires_both_dates_through():
    paper = parse_record(__import__("json").dumps(REAL_RECORD))
    assert paper["published_at"].year == 2007
    assert paper["updated_at"].year == 2008
    assert paper["categories"] == ["hep-ph"]
