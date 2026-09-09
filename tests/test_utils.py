"""Unit tests for kalshi_bot.utils.parse_iso_dt.

The audit flagged that a *naive* (no-offset) timestamp coming out of this
helper can trip a TypeError when compared against a tz-aware `now` in the exit
loop. These tests document the current behavior explicitly, so that if we later
decide to normalize to aware-UTC inside parse_iso_dt, that change is a conscious
one caught by a failing test rather than a silent behavior shift.
"""
from datetime import datetime, timezone

import pytest

from kalshi_bot.utils import parse_iso_dt


def test_z_suffix_parses_as_utc_aware():
    dt = parse_iso_dt("2026-09-09T12:00:00Z")
    assert dt.tzinfo is not None
    assert dt.utcoffset() == timezone.utc.utcoffset(None)
    assert dt == datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)


def test_explicit_offset_is_aware():
    dt = parse_iso_dt("2026-09-09T12:00:00+00:00")
    assert dt.tzinfo is not None
    assert dt == datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)


def test_nonzero_offset_preserved():
    dt = parse_iso_dt("2026-09-09T08:00:00-04:00")
    # Same instant as 12:00Z.
    assert dt.astimezone(timezone.utc) == datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)


def test_microseconds_ok():
    dt = parse_iso_dt("2026-09-09T12:00:00.123456+00:00")
    assert dt.microsecond == 123456


def test_naive_input_stays_naive_documented_trap():
    # A string with NO offset yields a NAIVE datetime. This is the latent trap:
    # comparing it to a tz-aware datetime raises TypeError. The exit-loop guards
    # now catch TypeError; this test pins the naive-in/naive-out behavior so a
    # future normalization is deliberate.
    dt = parse_iso_dt("2026-09-09T12:00:00")
    assert dt.tzinfo is None
    with pytest.raises(TypeError):
        _ = dt < datetime.now(timezone.utc)


def test_malformed_raises_value_error():
    with pytest.raises(ValueError):
        parse_iso_dt("not-a-timestamp")
