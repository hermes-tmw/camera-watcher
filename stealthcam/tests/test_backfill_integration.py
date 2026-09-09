"""Real-Postgres integration test for the backfill --force path.

The unit-test fake cursor historically did NOT model the ``_backfill_markers``
PRIMARY KEY constraint, so ``test_force_overrides_marker`` passed where the
real DB raised ``UniqueViolation`` and rolled back the whole transaction (the
``--force`` escape hatch was dead in its own documented scenario).

This test exercises the real ``backfill()`` against a live Postgres and pins
the fixed behaviour: a forced re-run with the marker present upserts the
marker (``ON CONFLICT DO UPDATE``) instead of raising, while the default
no-force path still refuses.

It is skipped unless ``BACKFILL_TEST_DATABASE_URL`` points at a disposable
Postgres (the test drops and recreates its tables). Example:

    BACKFILL_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/postgres \\
        python3 -m pytest stealthcam/tests/test_backfill_integration.py -q
"""
from __future__ import annotations

import os
import unittest
from datetime import datetime

import psycopg2

from stealthcam.backfill_tz import backfill, MARKER_KEY

DB_URL = os.environ.get("BACKFILL_TEST_DATABASE_URL")


@unittest.skipUnless(DB_URL, "BACKFILL_TEST_DATABASE_URL not set; "
                             "skipping real-Postgres integration test")
class TestBackfillForceIntegration(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg2.connect(DB_URL)
        self.conn.autocommit = True
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS event_observations, "
                    "stealthcam_telemetry, _backfill_markers")
        cur.execute("CREATE TABLE event_observations ("
                    "id SERIAL PRIMARY KEY, camera TEXT NOT NULL, "
                    "capture_time TIMESTAMP)")
        cur.execute("CREATE TABLE stealthcam_telemetry ("
                    "id SERIAL PRIMARY KEY, last_sync_at TIMESTAMP)")
        cur.execute("CREATE TABLE _backfill_markers ("
                    "key TEXT PRIMARY KEY, completed_at TIMESTAMP NOT NULL, "
                    "event_rows INTEGER NOT NULL, telemetry_rows INTEGER NOT NULL)")
        # One pre-deploy stealthcam row (naive MDT) and one telemetry row.
        cur.execute("INSERT INTO event_observations (camera, capture_time) "
                    "VALUES ('stealthcam', '2026-07-08 10:39:32')")
        cur.execute("INSERT INTO stealthcam_telemetry (last_sync_at) "
                    "VALUES ('2026-07-08 10:00:00')")
        cur.close()

    def tearDown(self):
        self.conn.close()

    def _marker(self):
        cur = self.conn.cursor()
        cur.execute("SELECT key, event_rows, telemetry_rows "
                    "FROM _backfill_markers WHERE key = %s", (MARKER_KEY,))
        row = cur.fetchone()
        cur.close()
        return row

    def test_force_with_marker_present_upserts(self):
        # Run 1: converts rows and writes the marker.
        n_cap, n_sync = backfill(DB_URL, max_event_id=1, max_telemetry_id=1)
        self.assertEqual((n_cap, n_sync), (1, 1))
        self.assertEqual(self._marker(), (MARKER_KEY, 1, 1))

        # Default re-run refuses (marker present, no --force).
        with self.assertRaises(RuntimeError):
            backfill(DB_URL, max_event_id=1, max_telemetry_id=1)

        # Forced re-run with the marker present must NOT raise UniqueViolation;
        # it upserts the marker and completes.
        n_cap, n_sync = backfill(DB_URL, max_event_id=1, max_telemetry_id=1,
                                 force=True)
        self.assertEqual((n_cap, n_sync), (1, 1))
        # Marker still exactly one row, upserted (not duplicated).
        self.assertEqual(self._marker(), (MARKER_KEY, 1, 1))


if __name__ == "__main__":
    unittest.main()
