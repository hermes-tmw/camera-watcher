"""Regression tests for the a718b85-deployment backfill path.

The round-2 review probes (P0-2) validated backfill against a FRESH in-memory
DB. This deployment (task t_60e8940a) exposed two live-DB gaps in the same
P0-2 area that the probes could not see:

1. worker.backfill() did not thread ``region_mapping`` into process_capture.
   This camera emits compass-bearing directions ("West_SouthWest"); with no
   mapping every frame resolves to an unclassified region, so record_monitored
   (denominator, gated on driveway/road) and entity resolution (driveway-only)
   never run — backfill seeds entities=0 even when a mapping IS configured
   for the live loop (silent config/behavior mismatch).

2. backfill() skips captures already in seen_guids. The pre-#36 worker has
   been marking captures seen for weeks, so on the LIVE pipeline DB the
   seen-set covers the whole backfill window (probe: 1524 seen_guids rows,
   0 unseen among the newest 8 API captures) and the default backfill would
   process 0 captures. Baseline seeding must be able to run over the
   already-seen history (``force_seen``).

These tests drive the REAL worker.backfill()/process_capture with stub
client/detector — same harness shape as Percy's probe_backfill_storm.py.
"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from stealthcam.worker import backfill
from stealthcam.memory import Memory
from stealthcam.reid import OpenCVFingerprinter


class _Det:
    """Minimal Detection stand-in (duck-typed: cls/bbox/confidence/category)."""

    def __init__(self, bbox, cls):
        self.bbox = bbox
        self.cls = cls

    @property
    def category(self):
        return "vehicle"

    confidence = 0.9


class _StubDetector:
    conf_threshold = 0.6

    def detect(self, image, conf_threshold=None):
        return [_Det((50.0, 110.0, 290.0, 270.0), "truck")]


class _StubClient:
    """Serves a deterministic truck crop; one page of 10 captures, newest-first."""

    def __init__(self):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (300, 300), (25, 30, 20))
        d = ImageDraw.Draw(img)
        d.rectangle([60, 150, 230, 260], fill=(70, 70, 75))
        d.rectangle([230, 120, 285, 260], fill=(95, 95, 100))
        self._img = img

    def download_subimage(self, guid, n, dest):
        self._img.save(dest, format="JPEG")
        return True

    def list_images(self, take_count=50, cursor=None):
        if cursor is not None:
            return []  # single page
        # 10 captures over ~3 hours; 16:00 UTC == 10:00 MDT (day window).
        t0 = datetime(2026, 9, 10, 16, 0, 0, tzinfo=timezone.utc)
        return [
            {"imageGuid": f"bf-{10 - i}",
             "createdDateTime": (t0 + timedelta(minutes=20 * i)).isoformat(),
             "imageDirections": ["West_SouthWest"] * 6}
            for i in range(10)
        ]


def _cfg(direction_regions: str = ""):
    cfg = mock.Mock()
    cfg.day_start_hour = 7
    cfg.day_end_hour = 20
    cfg.direction_regions = direction_regions
    return cfg


class TestBackfillSeedingOnLiveDb(unittest.TestCase):
    """P0-2 on a LIVE-like DB: seen-set full, compass directions."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sub = Path(self.tmp.name) / "sub"
        self.sub.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _memory(self):
        return Memory(":memory:", fingerprinter=OpenCVFingerprinter(),
                      entity_expected_presence_ratio=0.5,
                      entity_baseline_window_days=7.0)

    def _seed_seen(self, mem: Memory, n: int):
        """Pre-mark captures seen, as the pre-#36 worker did for weeks."""
        for i in range(1, n + 1):
            mem.mark_seen(f"bf-{i}")

    def test_backfill_seeds_entities_on_fully_seen_db_with_mapping(self):
        m = self._memory()
        self._seed_seen(m, 10)
        n = backfill(_cfg("west=driveway"), _StubClient(),
                     _StubDetector(), m, self.sub,
                     region_mapping={"west": "driveway"},
                     force_seen=True)
        self.assertEqual(n, 10)
        ents = m._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        pres = m._conn.execute(
            "SELECT COUNT(*) FROM entity_presence").fetchone()[0]
        mon = m._conn.execute(
            "SELECT COUNT(*) FROM monitored_captures").fetchone()[0]
        # P0-2 gate on the live DB: entities and presence rows must be seeded.
        self.assertGreater(ents, 0, "entities must be seeded during backfill")
        self.assertGreater(pres, 0, "entity_presence must be seeded")
        self.assertGreater(mon, 0, "monitored_captures denominator must exist")
        m.close()

    def test_backfill_without_mapping_seeds_zero_entities(self):
        # Compass directions + no mapping -> unclassified everywhere -> the
        # pre-fix behavior silently seeds nothing. PINNED as the failure shape
        # the empty-mapping warning guards against.
        m = self._memory()
        n = backfill(_cfg(""), _StubClient(), _StubDetector(), m, self.sub,
                     region_mapping=None, force_seen=True)
        self.assertEqual(n, 10)  # captures ARE processed...
        ents = m._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        mon = m._conn.execute(
            "SELECT COUNT(*) FROM monitored_captures").fetchone()[0]
        self.assertEqual(ents, 0, "no mapping -> no entity resolution")
        self.assertEqual(mon, 0, "no mapping -> no monitored denominator")
        m.close()

    def test_backfill_default_skips_seen(self):
        # Without force_seen, the seen-set skip is preserved (idempotent
        # re-runs must not double-seed).
        m = self._memory()
        self._seed_seen(m, 10)
        n = backfill(_cfg("west=driveway"), _StubClient(),
                     _StubDetector(), m, self.sub,
                     region_mapping={"west_southwest": "driveway"})
        self.assertEqual(n, 0)
        ents = m._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        self.assertEqual(ents, 0)
        m.close()

    def test_backfill_threaded_mapping_reaches_presence_denominator(self):
        # The mapping threaded through the WORKER entry (not process_capture
        # directly, as Percy's probe did) must reach record_monitored: the
        # denominator is what keeps the first live poll ratio sane.
        m = self._memory()
        n = backfill(_cfg("west=driveway"), _StubClient(),
                     _StubDetector(), m, self.sub,
                     region_mapping={"west": "driveway"},
                     force_seen=True)
        self.assertEqual(n, 10)
        mon = m._conn.execute(
            "SELECT COUNT(*) FROM monitored_captures "
            "WHERE direction='West_SouthWest' AND region='driveway' "
            "AND is_day=1").fetchone()[0]
        # 10 captures x 6 sub-images (every lens carries the bearing in the
        # stub) — the denominator counts per sub-image, per record_monitored.
        self.assertEqual(mon, 60)
        m.close()


if __name__ == "__main__":
    unittest.main()