"""Tests for the stealthcam capture_time timezone fix.

Before the fix, ``stealthcam/dashboard.py`` stored ``capture_time`` as naive
US/Mountain wall-clock, while ``watcher/browser.py::_fmt_time`` treats the
naive column as container-local UTC — so stealthcam events rendered 6h early.

After the fix, the writer stores naive UTC (matching the motion path) and the
browser renders the correct MDT wall-clock. These tests pin that contract:

  - ``_capture_time_naive`` returns naive UTC (no tzinfo).
  - ``_lighting`` derives daylight/night from the US/Mountain-aware hour, not
    the stored UTC-naive value (so a 10:39 MDT capture stays "daylight").
  - ``_fmt_time`` renders a naive-UTC value as the MDT wall-clock burned into
    the image (naive-UTC-in -> MDT-out).
  - the backfill's MDT->UTC conversion is DST-safe (per-row wall-tz aware).
"""
from __future__ import annotations

import unittest
from datetime import datetime
from unittest import mock

import pytz

from stealthcam.dashboard import DashboardWriter, CAMERA_TZ
from stealthcam.backfill_tz import _mdt_naive_to_utc_naive, backfill, MARKER_KEY


class TestCaptureTimeNaive(unittest.TestCase):
    def test_returns_naive_utc(self):
        # createdDateTime is UTC (naive or Z-suffixed); the stored value must be
        # naive UTC so the browser's _fmt_time (which assumes container-local
        # UTC) renders the correct MDT wall-clock.
        out = DashboardWriter._capture_time_naive(None, "2026-09-08T16:39:32")
        self.assertIsNone(out.tzinfo)
        self.assertEqual(out, datetime(2026, 9, 8, 16, 39, 32))

    def test_handles_z_suffixed_input(self):
        out = DashboardWriter._capture_time_naive(None, "2026-09-08T16:39:32Z")
        self.assertIsNone(out.tzinfo)
        self.assertEqual(out, datetime(2026, 9, 8, 16, 39, 32))

    def test_handles_offset_input(self):
        # An explicit +00:00 offset is also UTC.
        out = DashboardWriter._capture_time_naive(None, "2026-09-08T16:39:32+00:00")
        self.assertIsNone(out.tzinfo)
        self.assertEqual(out, datetime(2026, 9, 8, 16, 39, 32))


class TestLighting(unittest.TestCase):
    def test_lighting_uses_mountain_wall_hour(self):
        # 16:39 UTC == 10:39 MDT -> daylight (7 <= hour < 20).
        mountain = DashboardWriter._capture_time_mountain(None, "2026-09-08T16:39:32")
        self.assertEqual(mountain.hour, 10)
        self.assertEqual(DashboardWriter._lighting(None, mountain), "daylight")

    def test_lighting_night_boundary(self):
        # 03:00 UTC == 21:00 MDT (previous day) -> night.
        mountain = DashboardWriter._capture_time_mountain(None, "2026-09-08T03:00:00")
        self.assertEqual(mountain.hour, 21)
        self.assertEqual(DashboardWriter._lighting(None, mountain), "night")


class TestBrowserRenderRoundTrip(unittest.TestCase):
    """naive-UTC-in -> MDT-out, matching the image burn-in.

    ``_fmt_time`` does ``dt.astimezone(_tz())`` on a naive datetime, which
    Python interprets in the *system* local timezone. In the Docker container
    that is UTC (containers run ``date -> UTC``), so a naive-UTC value converts
    to MDT correctly. We pin the process TZ to UTC for the duration of the test
    to reproduce the container environment deterministically, regardless of the
    host's own timezone.
    """

    def setUp(self):
        import os
        import time
        self._old_tz = os.environ.get("TZ")
        os.environ["TZ"] = "UTC"
        time.tzset()

    def tearDown(self):
        import os
        import time
        if self._old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = self._old_tz
        time.tzset()

    def test_fmt_time_renders_mdt_from_naive_utc(self):
        from watcher.browser import _fmt_time

        # The stored value after the fix: naive UTC for 10:39:32 MDT.
        naive_utc = datetime(2026, 9, 8, 16, 39, 32)
        with mock.patch("watcher.browser._tz",
                        return_value=pytz.timezone(CAMERA_TZ)):
            rendered = _fmt_time(naive_utc)

        # The burn-in is "10:39AM 09/08/2026"; the browser must show 10:39 AM.
        self.assertIn("10:39 AM", rendered)
        self.assertIn("Sep 8, 2026", rendered)
        # And must NOT show the 6h-early 04:39 AM.
        self.assertNotIn("04:39", rendered)


class TestBackfillDstSafe(unittest.TestCase):
    def test_mdt_to_utc_summer(self):
        # 2026-09-08 is MDT (UTC-6): 10:39 MDT -> 16:39 UTC.
        out = _mdt_naive_to_utc_naive(datetime(2026, 9, 8, 10, 39, 32))
        self.assertIsNone(out.tzinfo)
        self.assertEqual(out, datetime(2026, 9, 8, 16, 39, 32))

    def test_mdt_to_utc_winter(self):
        # 2026-01-15 is MST (UTC-7): 10:39 MST -> 17:39 UTC (not a flat +6h).
        out = _mdt_naive_to_utc_naive(datetime(2026, 1, 15, 10, 39, 32))
        self.assertIsNone(out.tzinfo)
        self.assertEqual(out, datetime(2026, 1, 15, 17, 39, 32))


class _FakeCursor:
    """Records executed SQL and returns canned rows for the backfill's queries.

    Applies the ``id <= N`` cutoff to the canned rows so the conversion count
    reflects what a real DB would return for the WHERE clause.
    """

    def __init__(self, event_rows, telemetry_rows, marker_present=False):
        self.event_rows = event_rows
        self.telemetry_rows = telemetry_rows
        self.marker_present = marker_present
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def _last(self):
        return self.executed[-1]

    def fetchall(self):
        sql, params = self._last()
        if "event_observations" in sql:
            rows = self.event_rows
            if params:
                max_id = params[0]
                rows = [r for r in rows if r[0] <= max_id]
            return rows
        if "stealthcam_telemetry" in sql:
            rows = self.telemetry_rows
            if params:
                max_id = params[0]
                rows = [r for r in rows if r[0] <= max_id]
            return rows
        return []

    def fetchone(self):
        sql, _ = self._last()
        if "_backfill_markers" in sql and "SELECT" in sql:
            return (MARKER_KEY,) if self.marker_present else None
        return None

    def close(self):
        pass


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        pass


class TestBackfillGuard(unittest.TestCase):
    def _run(self, event_rows, telemetry_rows, marker_present=False,
             max_event_id=None, max_telemetry_id=None, force=False):
        cur = _FakeCursor(event_rows, telemetry_rows, marker_present)
        conn = _FakeConn(cur)
        with mock.patch("stealthcam.backfill_tz.psycopg2.connect",
                        return_value=conn):
            return backfill(
                "postgresql://fake/fake",
                max_event_id=max_event_id,
                max_telemetry_id=max_telemetry_id,
                force=force,
            ), cur, conn

    def test_refuses_rerun_when_marker_present(self):
        with self.assertRaises(RuntimeError):
            self._run([], [], marker_present=True)

    def test_force_overrides_marker(self):
        (n_cap, n_sync), cur, conn = self._run(
            [], [], marker_present=True, force=True)
        self.assertEqual((n_cap, n_sync), (0, 0))
        self.assertTrue(conn.committed)

    def test_cutoff_filters_rows(self):
        # Two event rows (ids 1, 2) and two telemetry rows (ids 1, 2).
        event_rows = [
            (1, datetime(2026, 9, 8, 10, 39, 32)),
            (2, datetime(2026, 9, 8, 11, 0, 0)),
        ]
        telemetry_rows = [
            (1, datetime(2026, 9, 8, 10, 0, 0)),
            (2, datetime(2026, 9, 8, 11, 0, 0)),
        ]
        (n_cap, n_sync), cur, conn = self._run(
            event_rows, telemetry_rows, max_event_id=1, max_telemetry_id=1)
        # Only id <= 1 converted in each table.
        self.assertEqual((n_cap, n_sync), (1, 1))
        # The SELECTs carried the cutoff predicate.
        event_select = [s for s, _ in cur.executed
                        if "event_observations" in s][0]
        telemetry_select = [s for s, _ in cur.executed
                            if "stealthcam_telemetry" in s][0]
        self.assertIn("id <= %s", event_select)
        self.assertIn("id <= %s", telemetry_select)

    def test_no_cutoff_converts_all(self):
        event_rows = [(1, datetime(2026, 9, 8, 10, 39, 32))]
        telemetry_rows = [(1, datetime(2026, 9, 8, 10, 0, 0))]
        (n_cap, n_sync), cur, conn = self._run(event_rows, telemetry_rows)
        self.assertEqual((n_cap, n_sync), (1, 1))
        self.assertTrue(conn.committed)

    def test_marker_written_on_success(self):
        (n_cap, n_sync), cur, conn = self._run([], [])
        marker_inserts = [s for s, _ in cur.executed
                          if "INSERT INTO _backfill_markers" in s]
        self.assertEqual(len(marker_inserts), 1)


if __name__ == "__main__":
    unittest.main()
